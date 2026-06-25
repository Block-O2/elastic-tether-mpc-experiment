"""
非线性材料模型 — 用于验证大纲第三节提出的核心要求：

    "标准MPC依赖精确模型，而本研究中的柔性对象可能呈现非线性、迟滞
    乃至各向异性，若简单视为线性弹簧，预测偏差将导致约束违反"

整个 v15 架构（BOCD/RLS/GPR/MPC/距离约束）目前只在线性弹簧材料下被
验证过。这里设计三种递进复杂度的非线性模型，对应大纲第五节列出的
材料谱系（橡皮筋、硅胶带、棉绳、弹力织物——"从线性弹性到显著非线性"）：

  1. HardeningSpring  — 渐进硬化，对应硅胶带类材料（应变硬化）
  2. HysteresisSpring — 加载/卸载路径不同，对应棉绳/织物类材料
  3. PiecewiseAnisoSpring — 分段刚度+方向依赖，对应橡皮筋类材料
     （初期软、接近极限时陡硬 + 不同牵引方向刚度不同）

接口设计
--------
所有模型实现统一的 compute(stretch, vel, acc, direction_angle=0.) 接口，
返回标量力大小（不含方向，方向由调用者乘上单位向量）。这个接口是无
状态的——即使是迟滞模型，"加载还是卸载"也完全由传入的 vel 符号瞬时
决定，不需要模型内部记忆上一步状态，因为 true_force_3d 本来就会在
每一步把当前 vel 传进来。

LinearSpring 作为对照组保留，和当前 integrated_sim_3d.py 内部硬编码
的线性模型行为完全一致，用于验证"非线性脚本在线性材料下退化回已知
结果"这件事（如果不一致，说明改造过程引入了问题）。

ks_max 估计
----------
非线性材料没有一个单一的"真实刚度"可以作为诊断指标(viol)的归一化
基准——线性模型里 ks_true 是个常数，但硬化/迟滞/各向异性模型里
等效刚度随 stretch、vel、方向变化。这里不为每个模型手写解析公式去
推最大刚度（容易出错，且每加一种新模型都要重新推导），而是用采样：
在 [0, STRETCH_MAX] 范围内扫描 stretch、vel 符号、牵引方向，找出
实际可能出现的最大力，反推一个保守的等效 ks_max。这个流程对任何
满足统一 compute 接口的模型都适用，不需要模型自己声明任何额外信息，
也更贴近现实——现实中我们同样只能通过有限次拉伸测量去估计材料的
力学边界，不会有一个解析公式直接告诉你"最大刚度是多少"。
"""
import numpy as np


class LinearSpring:
    """线性弹簧，对照组。与 integrated_sim_3d.py 内部硬编码的线性模型
    行为完全一致：f = ks*stretch + b*vel + m*acc"""

    def __init__(self, ks, b=0.5, m=0.05):
        self.ks = ks
        self.b = b
        self.m = m

    def compute(self, stretch, vel, acc, direction_angle=0.):
        if stretch <= 0:
            return 0.0
        return self.ks * stretch + self.b * vel + self.m * acc

    def __repr__(self):
        return f"LinearSpring(ks={self.ks})"


class HardeningSpring:
    """渐进硬化：刚度随 stretch 增大而增大（应变硬化），对应硅胶带类材料。
    ks_eff(stretch) = ks_base * (1 + alpha*stretch)，单调递增。"""

    def __init__(self, ks_base, alpha, b=0.5, m=0.05):
        self.ks_base = ks_base
        self.alpha = alpha
        self.b = b
        self.m = m

    def compute(self, stretch, vel, acc, direction_angle=0.):
        if stretch <= 0:
            return 0.0
        ks_eff = self.ks_base * (1.0 + self.alpha * stretch)
        return ks_eff * stretch + self.b * vel + self.m * acc

    def __repr__(self):
        return f"HardeningSpring(ks_base={self.ks_base}, alpha={self.alpha})"


class HysteresisSpring:
    """加载/卸载路径不同，对应棉绳/织物类材料的典型迟滞特性。
    同样的 stretch，正在拉伸(vel>0)时力偏高，正在回缩(vel<0)时力偏低，
    中间围出一个迟滞环。标准的 f=ks*stretch 线性模型无法表达这种
    "同一个 stretch 对应不同力" 的行为，是对 RLS 线性近似和 GPR
    非线性拟合能力的直接考验。"""

    def __init__(self, ks, hyst_gain, b=0.5, m=0.05):
        self.ks = ks
        self.hyst_gain = hyst_gain
        self.b = b
        self.m = m

    def compute(self, stretch, vel, acc, direction_angle=0.):
        if stretch <= 0:
            return 0.0
        base = self.ks * stretch
        if vel > 0:
            hyst = self.hyst_gain * stretch          # 加载：力偏高
        elif vel < 0:
            hyst = -self.hyst_gain * stretch * 0.5    # 卸载：力偏低（不对称）
        else:
            hyst = 0.0
        return max(0.0, base + hyst + self.b * vel + self.m * acc)

    def __repr__(self):
        return f"HysteresisSpring(ks={self.ks}, hyst_gain={self.hyst_gain})"


class PiecewiseAnisoSpring:
    """分段刚度 + 方向依赖，对应橡皮筋类材料：初期软、接近拉伸极限时
    刚度陡增（材料逼近极限阻力），且不同牵引方向（相对 anchor 的角度）
    刚度不同（各向异性，比如织物纹理方向）。这是大纲原文"非线性、
    迟滞乃至各向异性"里前两个模型没覆盖的最后一块。

    breakpoint_frac: 分段转折点占 STRETCH_MAX 的比例
    aniso_amp: 各向异性幅度，0表示无方向依赖，建议 < 0.5 避免出现负刚度
    """

    def __init__(self, ks_soft, ks_hard, STRETCH_MAX, breakpoint_frac=0.6,
                 aniso_amp=0.2, b=0.5, m=0.05):
        self.ks_soft = ks_soft
        self.ks_hard = ks_hard
        self.STRETCH_MAX = STRETCH_MAX
        self.breakpoint = STRETCH_MAX * breakpoint_frac
        self.aniso_amp = aniso_amp
        self.b = b
        self.m = m

    def compute(self, stretch, vel, acc, direction_angle=0.):
        if stretch <= 0:
            return 0.0
        if stretch <= self.breakpoint:
            ks_eff = self.ks_soft
        else:
            span = max(self.STRETCH_MAX - self.breakpoint, 1e-9)
            frac = (stretch - self.breakpoint) / span
            ks_eff = self.ks_soft + (self.ks_hard - self.ks_soft) * frac ** 2
        # 各向异性：乘性调制，cos(2*angle) 使其有两个等效方向的周期
        # （比如织物纹理沿/横方向），在 cos=1 时达到最大增益
        ks_eff *= (1.0 + self.aniso_amp * np.cos(2 * direction_angle))
        return ks_eff * stretch + self.b * vel + self.m * acc

    def __repr__(self):
        return (f"PiecewiseAnisoSpring(ks_soft={self.ks_soft}, "
                f"ks_hard={self.ks_hard}, aniso_amp={self.aniso_amp})")


def estimate_ks_max(force_model, STRETCH_MAX, n_stretch=50, v_probe=0.05,
                     n_angle=8):
    """
    采样估计材料模型在 [0, STRETCH_MAX] 内的保守等效最大刚度。

    不依赖任何解析形式，对线性/硬化/迟滞/各向异性模型统一适用——只要
    模型实现了 compute(stretch, vel, acc, direction_angle) 接口。

    扫描范围：
      - stretch: 0 到 STRETCH_MAX 线性采样 n_stretch 个点
      - vel: {+v_probe, -v_probe, 0}，覆盖加载/卸载/静止三种状态
             （迟滞模型的关键：同一 stretch 在不同 vel 符号下力不同）
      - direction_angle: 0 到 2π 采样 n_angle 个点（各向异性模型的关键）

    返回的 ks_max_equiv = max_force / STRETCH_MAX，是一个"等效线性
    刚度上界"——用最大可能出现的力，按线性弹簧的换算方式反推。这个
    值只用于事后诊断指标（如 viol）的归一化基准，不进入控制器决策。
    """
    if STRETCH_MAX <= 0:
        return 0.0
    stretches = np.linspace(0, STRETCH_MAX, n_stretch)
    v_signs = [v_probe, -v_probe, 0.0]
    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)

    max_f = 0.0
    for s in stretches:
        for v in v_signs:
            for ang in angles:
                f = force_model.compute(s, v, 0.0, direction_angle=ang)
                if f > max_f:
                    max_f = f

    return max_f / STRETCH_MAX


if __name__ == "__main__":
    # 自检：线性模型采样结果应该精确等于其 ks 本身（无论 b/m 取什么值，
    # 因为 v_probe 很小、acc=0，硬约束下采样误差应该可以忽略）
    STRETCH_MAX = 0.25
    lin = LinearSpring(ks=30, b=0.5, m=0.05)
    ks_max_est = estimate_ks_max(lin, STRETCH_MAX)
    print(f"LinearSpring(ks=30) 采样估计 ks_max = {ks_max_est:.3f} "
          f"(应接近 30 + b*v_probe/STRETCH_MAX 的微小偏移)")

    hard = HardeningSpring(ks_base=10, alpha=2.0)
    print(f"HardeningSpring(ks_base=10, alpha=2.0) 采样估计 ks_max = "
          f"{estimate_ks_max(hard, STRETCH_MAX):.3f}")

    hyst = HysteresisSpring(ks=20, hyst_gain=0.3)
    print(f"HysteresisSpring(ks=20, hyst_gain=0.3) 采样估计 ks_max = "
          f"{estimate_ks_max(hyst, STRETCH_MAX):.3f}")

    pw = PiecewiseAnisoSpring(ks_soft=8, ks_hard=40, STRETCH_MAX=STRETCH_MAX,
                               aniso_amp=0.2)
    print(f"PiecewiseAnisoSpring(soft=8,hard=40,aniso=0.2) 采样估计 ks_max = "
          f"{estimate_ks_max(pw, STRETCH_MAX):.3f}")