"""
实验矩阵脚本 v3 — 基于 v15 架构，直接复用 integrated_sim_3d.py 的 run_sim

与 v2 的关键区别：v2 在本文件内维护了一份独立的 BOCD/RLS/GPR/MPC/主循环
实现，随着 integrated_sim_3d.py 的迭代（v13→v15）逐渐与主文件脱节。
v3 改为直接 `from integrated_sim_3d import run_sim`，两个脚本共享同一份
逻辑——以后任何对核心算法的修改只需要改 integrated_sim_3d.py 一处，
本文件自动保持同步，不会再出现实验脚本验证的是过时架构的问题。

4×3×2 factorial design:
  - 材料(ks_true): 10, 30, 50, 100 N/m
  - 弧角(arc_deg): 30°, 60°, 90°
  - 速度权重(w_time): 0.5, 2.0
共 24 组，结果写入 results_v3.csv

STRETCH_MAX 的设定：
  之前(v2)的版本里，STRETCH_MAX 是从 f_max_safe/ks_true 反推的，让
  不同材料的"允许拉伸量"自动适配材料刚度。但 v15 的架构立场是
  STRETCH_MAX 应该是纯临床输入（治疗师根据关节活动度设定），与材料
  刚度无关 —— 同一个患者关节、同一次治疗，不会因为遇到的材料软硬不同
  就采用不同的拉伸上限。因此这里 STRETCH_MAX 在整组实验里固定为一个
  值，只随 ks 变化的是"同样拉伸量对应的力"，这正是这组实验想要观察的：
  系统在不依赖材料先验的前提下，能否在不同材料刚度下都保持安全和稳定。
"""
import sys
import os
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 直接复用 integrated_sim_3d.py 里的核心仿真逻辑，而不是重新实现一份
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_sim_3d import run_sim
from material_models import LinearSpring

# ══════════════════════════════════════════════════════
# 实验矩阵参数
# ══════════════════════════════════════════════════════
ks_list      = [10, 30, 50, 100]
arc_list     = [30, 60, 90]
w_time_list  = [0.5, 2.0]

# 临床参数：整组实验固定，不随材料变化（详见上方说明）
STRETCH_MAX  = 0.25   # m，治疗师设定的最大允许位移
F_MAX_SAFE   = 3.0    # N，安全力阈值（仅作观测，不参与决策）

# 关于结果里的 'viol' 字段：
# 不是和上面 F_MAX_SAFE 这个固定值比较——3N 这个绝对数字没有按材料
# 归一化，软材料(ks小)永远到不了，硬材料(ks大)轻松超过，会让这个字段
# 在不同 ks 组之间失去区分度，只是在重复"ks 大小"这个已知信息。
# run_sim 内部用 estimate_ks_max(force_model, STRETCH_MAX) 采样估计
# 材料的保守等效最大刚度，再算 f_static_max = 0.9 * STRETCH_MAX *
# ks_max_equiv，代表"材料被拉伸到 90% 允许范围时应产生的参照力"。
# viol 衡量的是实际力超过这个材料自身参照值的步数占比，能真实反映
# "力控制得好不好"，而不是和某个绝对数字比大小。这套归一化对线性/
# 非线性材料都适用（线性材料下采样结果精确收敛到 ks 本身）。


def run_one(ks, arc_deg, w_time, seed=42):
    """跑一组实验，提取关心的标量指标。

    run_sim 现在接受 force_model 对象而不是裸的 ks_true 标量，这里把
    ks 包成 LinearSpring(ks=ks)——线性弹簧是 run_sim 支持的众多材料
    模型里最简单的一种特例，这个矩阵脚本本身仍然只测线性材料（已经
    过 24 组实验验证的稳定基线），扩展到非线性材料矩阵是后续单独的
    工作，不在这次改动范围内。
    """
    model = LinearSpring(ks=ks, b=0.5, m=0.05)
    r = run_sim(force_model=model, arc_deg=arc_deg,
                STRETCH_MAX=STRETCH_MAX, f_max_safe=F_MAX_SAFE,
                w_time=w_time, seed=seed, verbose=False)

    if r['status'] == 'FAILED':
        return {
            'ks': ks, 'arc_deg': arc_deg, 'w_time': w_time,
            'status': 'FAILED',
            'ks_final': 'N/A', 'ks_final_err_pct': 'N/A',
            'rms': 'N/A', 'viol': 'N/A',
            'dist_viol_upper': 'N/A', 'dist_viol_lower': 'N/A',
            'r_mean': 'N/A', 'r_std': 'N/A',
            'total_steps': r['total_steps'], 'arc_steps': 'N/A',
        }

    ks_final = r['ks_final']
    ks_err_pct = round(abs(ks_final - ks) / ks * 100, 1)

    return {
        'ks': ks, 'arc_deg': arc_deg, 'w_time': w_time,
        'status': r['status'],
        'ks_final': round(ks_final, 2), 'ks_final_err_pct': ks_err_pct,
        'rms': round(r['rms'], 4), 'viol': round(r['viol'], 2),
        'dist_viol_upper': round(r['dist_viol_upper'], 2),
        'dist_viol_lower': round(r['dist_viol_lower'], 2),
        'r_mean': round(r['R_fit'], 4) if r['R_fit'] else 'N/A',
        'r_std': round(r['R_std'], 5) if r['R_std'] else 'N/A',
        'total_steps': r['total_steps'], 'arc_steps': r['arc_steps_count'],
    }



if __name__ == "__main__":
    # 只有直接运行本文件(python run_experiment_matrix.py)才会跑
    # 完整的24组实验。被 import 时不会触发，方便单独调用 run_one
    # 做小规模验证而不必每次都跑全部矩阵。
    # ══════════════════════════════════════════════════════
    # 执行实验矩阵
    # ══════════════════════════════════════════════════════
    results = []
    total = len(ks_list) * len(arc_list) * len(w_time_list)
    count = 0

    print("=" * 70)
    print(f"实验矩阵 v3：{total}组  直接复用 integrated_sim_3d.run_sim (v15架构)")
    print(f"STRETCH_MAX={STRETCH_MAX}m（固定，不随材料变化） f_max_safe={F_MAX_SAFE}N")
    print("=" * 70)

    for ks in ks_list:
        for arc_deg in arc_list:
            for w_time in w_time_list:
                count += 1
                print(f"[{count:2d}/{total}] ks={ks:3d}  arc={arc_deg:2d}°  w_time={w_time} ...",
                      end=' ', flush=True)
                r = run_one(ks, arc_deg, w_time)
                results.append(r)
                if r['status'] == 'FAILED':
                    print("FAILED（未在限定步数内完成绷直）")
                else:
                    print(f"ks_err={r['ks_final_err_pct']}%  RMS={r['rms']}  "
                          f"viol={r['viol']}%  dist_viol_up={r['dist_viol_upper']}%  "
                          f"dist_viol_lo={r['dist_viol_lower']}%  [{r['status']}]")

    # ══════════════════════════════════════════════════════
    # 写入 CSV
    # ══════════════════════════════════════════════════════
    fields = ['ks', 'arc_deg', 'w_time', 'status',
              'ks_final', 'ks_final_err_pct', 'rms', 'viol',
              'dist_viol_upper', 'dist_viol_lower',
              'r_mean', 'r_std', 'total_steps', 'arc_steps']

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results_v3.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已写入 {csv_path}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════
    # 按 ks 分组汇总
    # ══════════════════════════════════════════════════════
    print("\n[汇总] 按 ks 分组：")
    for ks in ks_list:
        grp = [r for r in results if r['ks'] == ks and r['status'] == 'OK']
        if grp:
            avg_rms = np.mean([r['rms'] for r in grp])
            avg_rstd = np.mean([r['r_std'] for r in grp if r['r_std'] != 'N/A'])
            avg_ks_err = np.mean([r['ks_final_err_pct'] for r in grp])
            max_dist_viol = max([max(r['dist_viol_upper'], r['dist_viol_lower']) for r in grp])
            n_failed = len([r for r in results if r['ks'] == ks]) - len(grp)
            print(f"  ks={ks:3d}: ks_err={avg_ks_err:.1f}%  RMS={avg_rms:.4f}N  "
                  f"R_std={avg_rstd:.5f}m  max距离违反={max_dist_viol:.1f}%  "
                  f"({len(grp)}组成功, {n_failed}组失败)")
        else:
            print(f"  ks={ks:3d}: 全部失败")