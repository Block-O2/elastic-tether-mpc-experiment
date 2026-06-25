"""
GPR vs RLS 力预测对比实验

背景：大纲原始设计是"GPR 为主预测模型，RLS 为降级后备"，但当前
integrated_sim_3d.py 里力预测全程依赖 RLS 的线性外推(ks_eff*stretch+
b*vel+m*acc)，GPR 只用 σ 收紧距离约束，均值预测从未被使用——这和
大纲设计正好相反。

6月19日的早期记录里曾经短暂把 GPR 设为主预测模型，但当时遇到的问题
是：arc 阶段径向约束把 stretch 锁得很死，几乎没有激励信息，导致 GPR
（以及 RLS）在 stretch 维度上的估计都不稳定。这不是"GPR 方法本身
不行"，而是"输入没有激励"。当时这个问题没有走到验证结束就被搁置，
后续版本回退成了 RLS 主导的结构。

现在的 v15 架构已经不存在这个问题——settle 阶段的梯形扫描专门设计了
大范围往返运动来主动制造激励，theta 在 arc 开始时被冻结，不再是 6月19
日那种"arc 阶段几乎不动、没有新激励"的情况。这次对比就是要在这个已经
解决了激励问题的架构基础上，重新、诚实地验证一次：GPR 均值预测和 RLS
线性预测，对真实 arc 阶段力的预测误差，到底谁更小。

方法
----
对每个材料模型：
1. 跑一次完整 run_sim，拿到 settle 结束时冻结的 RLS theta，以及训练
   好的 GPR 对象(run_sim 返回字典新增的 'gpr_model' 字段)
2. 从返回的 hp(位置历史)反推 arc 阶段每一步的 (stretch, vel, theta_pos)
   三元组——这正是 GPR 的输入特征，也是 RLS 公式需要的输入
3. 分别用两种方法预测每一步的力大小，和真实力(hf)算 RMS 误差
4. 对比哪个更小

这是独立于 run_sim 决策逻辑之外的事后分析，不修改任何控制流程，
只读取 run_sim 已经训练好的 RLS/GPR 状态做预测对比。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ongoing_parts.integrated_sim_3d import run_sim
from material_models import LinearSpring, HardeningSpring, HysteresisSpring


def reconstruct_arc_features(result):
    """
    从 run_sim 返回的位置历史(hp)反推 arc 阶段每一步的
    (stretch, vel, theta_pos) 特征，以及对应的真实力(标量大小)。

    这些正是 GPR 的输入特征 [stretch, vel, theta]，也是 RLS 线性
    预测 ks*stretch+b*vel+m*acc 需要的输入。run_sim 内部已经算过
    这些量，但没有作为独立的历史数组返回，这里从 hp 重建，dt 用
    run_sim 内部相同的步长(0.05，从 integrated_sim_3d 导入的模块级
    常量，跟 run_sim 内部完全一致)。
    """
    from ongoing_parts.integrated_sim_3d import dt as DT

    hp = result['hp']
    anchor_est = result['anchor_est']
    dist_taut = result['dist_taut']
    arc_step = result['arc_step']
    hf = result['hf']

    hp_arc = hp[arc_step:]
    hf_arc = np.asarray(hf[arc_step:])
    # hp 和 hf 的长度可能不完全相等：主循环里 hf.append() 发生在每轮
    # 开头，hp.append() 发生在每轮末尾；任务完成时的 break 发生在两者
    # 之间，导致该轮的 hf 已经记了一笔但 hp 还没来得及记。这不是固定的
    # "差1"关系（取决于 run 是通过 break 正常完成还是 T_sim 耗尽），
    # 统一按两者长度的较小值截断，确保每个索引上 hp_arc[i] 和
    # hf_arc[i] 确实对应同一个仿真步，而不是错位的样本。
    n = min(len(hp_arc), len(hf_arc))
    hp_arc = hp_arc[:n]
    hf_arc = hf_arc[:n]

    stretches = np.zeros(n)
    vels = np.zeros(n)
    thetas = np.zeros(n)

    v_prev = np.zeros(3)
    for i in range(n):
        p = hp_arc[i]
        d = p - anchor_est
        dist = np.linalg.norm(d)
        stretches[i] = max(0., dist - dist_taut)
        thetas[i] = np.arctan2(d[1], d[0])
        if i == 0:
            v = np.zeros(3)
        else:
            v = (hp_arc[i] - hp_arc[i-1]) / DT
        vels[i] = np.linalg.norm(v)

    return stretches, vels, thetas, hf_arc


def predict_rls(theta_frozen, stretches, vels):
    """RLS 线性预测：ks*stretch + b*vel（acc 项忽略，因为重建的 vel
    序列已经是有限差分近似，二次差分做 acc 噪声会被进一步放大，且
    run_mpc_3d 内部的力引导代价项实际使用的 pf 函数本身权重上也是
    stretch/vel 项贡献主导，这里只对比这两项的预测能力，是公平的
    简化，不影响两种方法的相对比较）。"""
    ks, b, m = theta_frozen
    return ks * stretches + b * vels


def predict_gpr(gpr_model, stretches, vels, thetas):
    """GPR 均值预测，逐点调用 predict（GPR 没有向量化批量接口）。
    数据不足/未 fit 时返回 None，由调用方处理。"""
    if not gpr_model.fitted:
        return None
    preds = np.zeros(len(stretches))
    for i in range(len(stretches)):
        mu, _ = gpr_model.predict(stretches[i], vels[i], thetas[i])
        preds[i] = mu if mu is not None else 0.
    return preds


def compare_one(name, force_model, arc_deg=90, STRETCH_MAX=0.25, seed=42,
                 verbose_each=True):
    r = run_sim(force_model=force_model, arc_deg=arc_deg,
                STRETCH_MAX=STRETCH_MAX, f_max_safe=3.0, seed=seed,
                verbose=False)
    if r['status'] != 'OK':
        if verbose_each:
            print(f"[{name}] 仿真未成功(status={r['status']})，跳过")
        return None

    stretches, vels, thetas, f_true = reconstruct_arc_features(r)
    f_true = np.asarray(f_true).reshape(-1)

    rls_pred = predict_rls(r['theta_frozen'], stretches, vels)
    gpr_pred = predict_gpr(r['gpr_model'], stretches, vels, thetas)

    rls_rms = float(np.sqrt(np.mean((rls_pred - f_true) ** 2)))
    if gpr_pred is not None:
        gpr_rms = float(np.sqrt(np.mean((gpr_pred - f_true) ** 2)))
    else:
        gpr_rms = None

    if verbose_each:
        print(f"[{name}]  ks_frozen={r['theta_frozen'][0]:.2f}  "
              f"arc步数={len(f_true)}")
        print(f"    RLS预测 RMS = {rls_rms:.4f} N")
        if gpr_rms is not None:
            winner = "GPR更优" if gpr_rms < rls_rms else "RLS更优"
            diff_pct = abs(gpr_rms - rls_rms) / rls_rms * 100
            print(f"    GPR预测 RMS = {gpr_rms:.4f} N   → {winner} "
                  f"(差距 {diff_pct:.1f}%)")
        else:
            print(f"    GPR预测 RMS = N/A (GPR 未成功 fit)")
        print()

    return {'name': name, 'seed': seed, 'rls_rms': rls_rms, 'gpr_rms': gpr_rms}


if __name__ == "__main__":
    # 多种子重复实验：单次运行的 RMS 差距容易被具体轨迹细节(梯形扫描
    # 折返时机、settle收敛瞬间的具体位置等)主导，不同 numpy/scipy/
    # sklearn 版本下同一个 seed 也可能走出不同轨迹(优化器/随机数生成
    # 实现细节跨版本有差异)。要回答"GPR 是否系统性地比 RLS 准"，应该
    # 看多个种子下的统计趋势，而不是单次结果——这与 run_experiment_
    # matrix.py 用 24 组实验而非单次运行做结论是同一个道理。
    print("=" * 60)
    print("GPR vs RLS 力预测对比（多种子重复，v15 架构）")
    print("=" * 60)
    print()

    configs = [
        ("线性(ks=30)", LinearSpring(ks=30, b=0.5, m=0.05)),
        ("线性(ks=10)", LinearSpring(ks=10, b=0.5, m=0.05)),
        ("渐进硬化", HardeningSpring(ks_base=10, alpha=2.0, b=0.5, m=0.05)),
        ("迟滞", HysteresisSpring(ks=20, hyst_gain=0.3, b=0.5, m=0.05)),
    ]
    seeds = [42, 43, 44, 45, 46]

    all_results = []
    total = len(configs) * len(seeds)
    count = 0

    for name, model in configs:
        for seed in seeds:
            count += 1
            print(f"[{count:2d}/{total}] {name}  seed={seed} ...", end=' ', flush=True)
            res = compare_one(name, model, seed=seed, verbose_each=False)
            if res:
                all_results.append(res)
                if res['gpr_rms'] is not None:
                    tag = "GPR" if res['gpr_rms'] < res['rls_rms'] else "RLS"
                    print(f"RLS={res['rls_rms']:.4f}N  GPR={res['gpr_rms']:.4f}N  → {tag}更优")
                else:
                    print(f"RLS={res['rls_rms']:.4f}N  GPR=N/A")
            else:
                print("仿真未成功，跳过")

    print()
    print("=" * 60)
    print("汇总（按材料分组，多种子统计）")
    print("=" * 60)
    for name, _ in configs:
        grp = [r for r in all_results if r['name'] == name and r['gpr_rms'] is not None]
        if not grp:
            print(f"  {name:12s}  无有效数据")
            continue
        rls_vals = np.array([r['rls_rms'] for r in grp])
        gpr_vals = np.array([r['gpr_rms'] for r in grp])
        gpr_win_count = int(np.sum(gpr_vals < rls_vals))
        n = len(grp)
        avg_rls = rls_vals.mean()
        avg_gpr = gpr_vals.mean()
        avg_diff_pct = float(np.mean((gpr_vals - rls_vals) / rls_vals)) * 100
        print(f"  {name:12s}  GPR胜出 {gpr_win_count}/{n} 次  "
              f"平均RLS={avg_rls:.4f}N  平均GPR={avg_gpr:.4f}N  "
              f"平均差距={avg_diff_pct:+.1f}%(负=GPR更优)")

    print()
    print("=" * 60)
    print("汇总（整体，不分材料）")
    print("=" * 60)
    all_valid = [r for r in all_results if r['gpr_rms'] is not None]
    if all_valid:
        total_n = len(all_valid)
        total_gpr_win = sum(1 for r in all_valid if r['gpr_rms'] < r['rls_rms'])
        print(f"  全部 {total_n} 组实验中，GPR 更优 {total_gpr_win} 次 "
              f"({total_gpr_win/total_n*100:.0f}%)，"
              f"RLS 更优 {total_n-total_gpr_win} 次 "
              f"({(total_n-total_gpr_win)/total_n*100:.0f}%)")