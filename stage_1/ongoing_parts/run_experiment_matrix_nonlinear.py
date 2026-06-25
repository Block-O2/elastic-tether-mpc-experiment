"""
非线性材料实验矩阵 — 与 run_experiment_matrix.py(线性24组) 平行的独立
脚本，验证点不同：线性那份矩阵的核心问题是"系统在不同材料刚度下是否
都保持安全稳定"；这份矩阵的核心问题是"GPR 在确实存在非线性结构的材料
下，有没有被更多地、更有效地采用"。

背景：之前在 ks=30 这种相对温和的线性材料下定位到一个 bug——GPR 在
stretch 极小(接近原点)的时刻被采用，但这个区域本身任何连续曲线都近似
线性，GPR 没有施展空间，反而因为 mu/stretch 这个除法对小分母敏感而
给出离谱估计。修复(min_stretch_for_ratio 防线)之后，线性材料矩阵的
RMS 已经完全回归纯 RLS 基线——这是符合预期的：材料越线性，GPR 被
正确地采用得越少。

这份矩阵要验证的是反过来的情形：渐进硬化(HardeningSpring，stretch
接近 STRETCH_MAX 时才显现明显非线性)、迟滞(HysteresisSpring，加载/
卸载路径分歧，且强度依赖 vel 的符号)这类材料下，GPR 是否会在"stretch
确实较大、已经走出线性近似适用范围"的区域被更多采用，并且在这些区域
确实比单纯线性外推的 RLS 更准——这正是 GPR 这条路径存在的意义所在，
不是要求它处处都赢，而是要求它在它该擅长的地方真正发挥作用。

4×3×2 factorial design(material 维度从线性的"ks 数值"换成材料对象):
  - 材料: 线性(ks=30，对照组) / 渐进硬化 / 迟滞 / 分段各向异性
  - 弧角(arc_deg): 30°, 60°, 90°
  - 速度权重(w_time): 0.5, 2.0
共 24 组，结果写入 results_nonlinear.csv

PiecewiseAnisoSpring 已知架构边界：settle 阶段刚度突变会让 RLS 无法
真正收敛(此前在 integrated_sim_3d.py 单独验证过)，这份矩阵里它依然
被保留作为对照——目的不是指望它成功，而是确认"即使这个材料下游预测
质量不可信，距离硬约束这道独立防线是否依然守得住"，FAILED/高ks_err
本身就是这组材料的预期结果，不代表脚本或这次改动有问题。
"""
import sys
import os
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_sim_3d import run_sim
from material_models import (LinearSpring, HardeningSpring,
                              HysteresisSpring, PiecewiseAnisoSpring,
                              estimate_ks_max)

# ══════════════════════════════════════════════════════
# 实验矩阵参数
# ══════════════════════════════════════════════════════
arc_list     = [30, 60, 90]
w_time_list  = [0.5, 2.0]

STRETCH_MAX  = 0.25   # m，和线性矩阵保持一致，便于跨矩阵对比
F_MAX_SAFE   = 3.0    # N，仅作观测，不参与决策


def make_materials():
    """每次调用返回一组全新的材料对象。force_model 在 run_sim 内部
    会被多次调用 compute()，但不持有跨 run 的状态，理论上同一个对象
    可以在多组实验间复用；这里仍然选择每组实验单独构造，避免任何
    意外的隐式状态共享，写法上更安全、也和线性矩阵"每次 run_one 内部
    现场构造"的风格一致。"""
    return {
        '线性(ks=30,对照)': LinearSpring(ks=30, b=0.5, m=0.05),
        '渐进硬化':          HardeningSpring(ks_base=10, alpha=2.0, b=0.5, m=0.05),
        '迟滞':              HysteresisSpring(ks=20, hyst_gain=0.3, b=0.5, m=0.05),
        '分段各向异性':       PiecewiseAnisoSpring(ks_soft=8, ks_hard=40,
                                                  STRETCH_MAX=STRETCH_MAX,
                                                  aniso_amp=0.2),
    }


def run_one(material_name, force_model, arc_deg, w_time, seed=42):
    """跑一组实验。

    与线性矩阵的 run_one 的关键区别：非线性材料没有单一的"真值 ks"
    可以用来算 ks_final_err_pct——RLS 在非线性材料下拟合出的是"settle
    扫描区间的最佳全局折中直线"，不是某个唯一正确答案，强行算一个
    "误差百分比"会暗示存在一个它应该收敛到的真值，这是误导的。改为
    直接展示 ks_final 这个折中值本身，以及 ks_max_equiv(用
    estimate_ks_max 采样估计的材料保守等效最大刚度，run_sim 内部算
    viol 时已经在用同一个函数)作为参考量级，但不算"误差"。

    新增 gpr_usage_pct 字段：这份矩阵存在的核心目的就是回答"GPR 在
    非线性材料下是否被更多采用"，所以直接在 run_sim 跑完之后，对
    arc 阶段的轨迹重放一遍，统计 gpr_force_anchor 实际命中 GPR(而非
    返回 None)的步数占比。
    """
    r = run_sim(force_model=force_model, arc_deg=arc_deg,
                STRETCH_MAX=STRETCH_MAX, f_max_safe=F_MAX_SAFE,
                w_time=w_time, seed=seed, verbose=False)

    ks_max_equiv = round(estimate_ks_max(force_model, STRETCH_MAX), 2)

    if r['status'] == 'FAILED':
        return {
            'material': material_name, 'arc_deg': arc_deg, 'w_time': w_time,
            'status': 'FAILED',
            'ks_final': 'N/A', 'ks_max_equiv': ks_max_equiv,
            'rms': 'N/A', 'viol': 'N/A',
            'dist_viol_upper': 'N/A', 'dist_viol_lower': 'N/A',
            'gpr_usage_pct': 'N/A',
            'r_mean': 'N/A', 'r_std': 'N/A',
            'total_steps': r['total_steps'], 'arc_steps': 'N/A',
        }

    gpr_usage_pct = compute_gpr_usage_pct(r)

    return {
        'material': material_name, 'arc_deg': arc_deg, 'w_time': w_time,
        'status': r['status'],
        'ks_final': round(r['ks_final'], 2), 'ks_max_equiv': ks_max_equiv,
        'rms': round(r['rms'], 4), 'viol': round(r['viol'], 2),
        'dist_viol_upper': round(r['dist_viol_upper'], 2),
        'dist_viol_lower': round(r['dist_viol_lower'], 2),
        'gpr_usage_pct': gpr_usage_pct,
        'r_mean': round(r['R_fit'], 4) if r['R_fit'] else 'N/A',
        'r_std': round(r['R_std'], 5) if r['R_std'] else 'N/A',
        'total_steps': r['total_steps'], 'arc_steps': r['arc_steps_count'],
    }


def compute_gpr_usage_pct(result):
    """事后重放 arc 阶段轨迹，统计 local_ratio 实际命中 GPR 的步数
    占比。不在 run_sim 内部直接埋点计数，是为了不侵入已经验证稳定
    的主循环逻辑——这个统计纯粹是分析用途，用 run_sim 返回的
    gpr_model/theta_frozen/hp/anchor_est/dist_taut 在外部重新计算，
    和 compare_gpr_rls.py 里 reconstruct_arc_features 的做法一致。
    """
    from integrated_sim_3d import dt as DT

    hp = result['hp']
    anchor_est = result['anchor_est']
    dist_taut = result['dist_taut']
    arc_step = result['arc_step']
    gpr = result['gpr_model']
    theta_frozen = result['theta_frozen']

    hp_arc = hp[arc_step:]
    n = len(hp_arc)
    if n < 2:
        return 0.0

    gpr_used = 0
    for i in range(n):
        p = hp_arc[i]
        d = p - anchor_est
        stretch = max(0., np.linalg.norm(d) - dist_taut)
        theta = np.arctan2(d[1], d[0])
        if i == 0:
            vel = 0.
        else:
            vel = np.linalg.norm((hp_arc[i] - hp_arc[i-1]) / DT)
        mu_anchor, _ = gpr.gpr_force_anchor(stretch, vel, theta)
        if mu_anchor is not None:
            gpr_used += 1
    return round(gpr_used / n * 100, 1)


if __name__ == "__main__":
    materials = make_materials()
    results = []
    total = len(materials) * len(arc_list) * len(w_time_list)
    count = 0

    print("=" * 70)
    print(f"非线性材料实验矩阵：{total}组  复用 integrated_sim_3d.run_sim (v15架构)")
    print(f"STRETCH_MAX={STRETCH_MAX}m  f_max_safe={F_MAX_SAFE}N")
    print("验证目标：GPR 在非线性材料下是否被更多采用、是否真正发挥作用")
    print("=" * 70)

    for material_name, force_model in materials.items():
        for arc_deg in arc_list:
            for w_time in w_time_list:
                count += 1
                print(f"[{count:2d}/{total}] {material_name:14s} arc={arc_deg:2d}°  "
                      f"w_time={w_time} ...", end=' ', flush=True)
                r = run_one(material_name, force_model, arc_deg, w_time)
                results.append(r)
                if r['status'] == 'FAILED':
                    print(f"FAILED（未在限定步数内完成绷直，ks_max_equiv参考={r['ks_max_equiv']}）")
                else:
                    print(f"ks_final={r['ks_final']}(参考ks_max_equiv={r['ks_max_equiv']})  "
                          f"RMS={r['rms']}  viol={r['viol']}%  "
                          f"GPR采用率={r['gpr_usage_pct']}%  "
                          f"dist_viol={max(r['dist_viol_upper'], r['dist_viol_lower'])}%  "
                          f"[{r['status']}]")

    # ══════════════════════════════════════════════════════
    # 写入 CSV
    # ══════════════════════════════════════════════════════
    fields = ['material', 'arc_deg', 'w_time', 'status',
              'ks_final', 'ks_max_equiv', 'rms', 'viol',
              'dist_viol_upper', 'dist_viol_lower', 'gpr_usage_pct',
              'r_mean', 'r_std', 'total_steps', 'arc_steps']

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results_nonlinear.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已写入 {csv_path}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════
    # 按材料分组汇总
    # ══════════════════════════════════════════════════════
    print("\n[汇总] 按材料分组：")
    for material_name in materials:
        grp = [r for r in results if r['material'] == material_name and r['status'] == 'OK']
        if grp:
            avg_rms = np.mean([r['rms'] for r in grp])
            avg_gpr_usage = np.mean([r['gpr_usage_pct'] for r in grp])
            max_dist_viol = max([max(r['dist_viol_upper'], r['dist_viol_lower']) for r in grp])
            n_failed = len([r for r in results if r['material'] == material_name]) - len(grp)
            print(f"  {material_name:14s}: RMS={avg_rms:.4f}N  "
                  f"平均GPR采用率={avg_gpr_usage:.1f}%  "
                  f"max距离违反={max_dist_viol:.1f}%  "
                  f"({len(grp)}组成功, {n_failed}组失败)")
        else:
            print(f"  {material_name:14s}: 全部失败")