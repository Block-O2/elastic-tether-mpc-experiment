"""
Integrated 3D Simulation — 大纲四层架构 v15

═══════════════════════════════════════════════════════════════
架构总览（三阶段状态机）
═══════════════════════════════════════════════════════════════
  阶段0 探索(explore) → 阶段1 settle(梯形扫描) → 阶段2 弧线(arc, MPC)

  探索：固定速度径向外推，力接近安全网时自动减速。同时标定传感器
        噪声水平(noise_std_est)，不更新RLS（stretch恒为0，没有信息量）。

  settle：BOCD检测到绷直后触发。机器人在[dist_taut, dist_taut+0.95*
        STRETCH_MAX]之间做梯形速度往返扫描，激励RLS学习ks/b/m。
        收敛判据只看ks这一维的不确定度（P矩阵[0,0]元素），收敛后
        冻结theta，不再要求b、m也收敛（详见下方"收敛判据"一节）。

  弧线：MPC在距离硬约束内规划轨迹，代价函数用力引导轨迹质量
        （维持在f_taut附近），不再用力做安全判断。

═══════════════════════════════════════════════════════════════
安全约束：只用距离，不用力
═══════════════════════════════════════════════════════════════
唯一的安全边界是距离约束：dist_taut ≤ dist ≤ dist_taut + STRETCH_MAX
  - dist_taut：BOCD触发时一次性测量的绷直距离，之后固定不变，
    取代了L0_est（L0_est已从代码中完全删除）
  - STRETCH_MAX：临床输入（治疗师根据关节活动度设定），与材料
    刚度ks无关
  - f_max_safe：保留作为观测量（画图、记录用），不参与任何控制
    决策分支。settle的折返和arc的安全性完全由距离约束保证，
    不再有任何"力超过阈值就强制改变运动"的判断。

临床立场：安全标准是"绷紧到位"（距离），不是"受了多少力"——同样
的力对软组织可能已过度拉伸、对硬组织可能还没绷紧，力不是材料
无关的判据，距离才是。

═══════════════════════════════════════════════════════════════
力的角色：从"安全判据"降级为"任务引导"
═══════════════════════════════════════════════════════════════
arc阶段MPC代价函数里有一项 W_FORCE_GUIDE*(f_pred-f_taut)^2，把预测
力拉向f_taut（刚好绷直时记录的力）。这不是安全约束（不会硬性限制
运动），而是轨迹质量的引导信号——单靠角度推进(W_ANGLE)和距离硬约束，
MPC在可行域内没有偏好，容易导致轨迹质量差；力引导让MPC主动维持
"刚好绷紧"的状态去传导运动，这正是牵引任务本身想要的状态。

═══════════════════════════════════════════════════════════════
收敛判据：三个条件，且只要求ks收敛
═══════════════════════════════════════════════════════════════
settle→arc的转换需要同时满足：
  (a) 有激励：settle过程中走到过的最大stretch占STRETCH_MAX的比例
      够大（EXCITATION_FRAC=0.3），用历史最大值而非瞬时值，因为
      settle是往返扫描，瞬时值在折返点会回到≈0
  (b) 预测准：力预测误差 < PRED_ERR_MULT * noise_std_est（标定值，
      不是真值），且(a)排除了stretch≈0时任何theta都能拟合对的假阳性
  (c) 估计稳：RLS对ks的不确定度（P矩阵[0,0]，相对初始值归一化）
      已收缩到阈值以下

      为什么只看ks、不要求b/m收敛：导师的指导是"受力预测先用简单
      阈值法试试，不够再上MLP/GPR"——ks主导的线性近似就是这里的
      "简单方法"，b/m对应的是更精细的受力预测，属于下一级复杂度，
      不需要在settle阶段就做到位。这也避免了b/m因为运动模式下
      vs与ac存在相关性而难以收敛时，拖累整个判据。

三选一不满足时，靠SETTLE_STEPS步数上限超时退出（保底机制，正常
应靠真实收敛提前退出，超时只应是少数情况）。

═══════════════════════════════════════════════════════════════
无真值泄露：控制器可见的先验只有三个，全部是临床/传感器量
═══════════════════════════════════════════════════════════════
  - STRETCH_MAX：关节最大允许位移，治疗师评估确定
  - f_max_safe：安全力阈值，治疗师评估确定（仅作观测，不参与决策）
  - anchor_est：锚点位置，实验开始前测量
  - noise_std_est：运行时从explore早期窗口标定，非真值

force_model（材料模型对象）、L0_true 只出现在"真实系统"代码块里
（仿真环境本身需要真值来生成数据），控制器逻辑全程不引用它们。
force_model 可以是线性弹簧（LinearSpring）或非线性材料（见
material_models.py）——控制器不知道、也不需要知道材料是哪一种。
"""
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import t as student_t
import warnings
warnings.filterwarnings('ignore')

# 材料模型：line/硬化/迟滞/各向异性，统一通过 compute(...) 接口接入
# run_sim。线性弹簧(LinearSpring)只是这套接口下最简单的一种特例，
# 不再是 run_sim 内部写死的唯一选项。
from material_models import (
    LinearSpring, HardeningSpring, HysteresisSpring, PiecewiseAnisoSpring,
    estimate_ks_max,
)

# 中文字体适配：按常见候选字体名依次尝试，找不到则静默回退到默认字体
# （回退后中文会变成方块/丢字，但不会报错；建议本地环境装 Noto Sans CJK 或 SimHei）
import matplotlib.font_manager as fm
_CJK_CANDIDATES = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei',
                    'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei',
                    'Arial Unicode MS']
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((name for name in _CJK_CANDIDATES if name in _available), None)
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════
# 算法超参数（不随实验材料/任务变化，留作模块级默认配置）
# ══════════════════════════════════════════════════════
dt = 0.05

# ══════════════════════════════════════════════════════
# MPC 权重
# ══════════════════════════════════════════════════════
W_FORCE  = 30.0
W_TIME   = 0.5
W_INPUT  = 0.0   # 临床场景不需要速度约束，已移除
W_ANGLE  = 10.0
V_MAX    = 0.3
V_MIN    = 0.05
V_RAMP   = 60
MPC_N    = 6

# ── 径向约束参数 ─────────────────────────────────────────
R_TOL        = 0.04
R_REF_UPDATE = 20

# ── Arc 早期安全预热 ─────────────────────────────────────
ARC_WARMUP_STEPS = 20
V_WARMUP         = 0.02

# ── Explore 阶段参数 ─────────────────────────────────────
V_EXPLORE        = 0.06   # 固定探索速度 (m/s)，保守值，与材料参数无关
NOISE_CALIB_STEPS = 20    # explore早期窗口，用于标定传感器噪声水平
                          # （此时还未接触，力读数纯粹是噪声，与材料无关）

# ── Settle 阶段参数 ──────────────────────────────────────
# 折返条件：dist > dist_taut + STRETCH_MAX*0.95（纯几何，无力触发分支）
# 硬/软材料激励幅度相同，不再依赖 SETTLE_AMP
# settle 运动采用梯形速度曲线(详见主循环)，不用插值曲线或恒速。
# 单程步数量级参考：(STRETCH_MAX*0.95)/SETTLE_SPEED/dt，梯形曲线下因
# 加减速段更长，实际步数会更多（见 SETTLE_ACCEL 注释附近的估算）。
# SETTLE_STEPS=450 留出约2次往返的余量，作为超时保底，正常应靠P矩阵
# 收敛判据提前退出，超时只应是极少数情况。
SETTLE_STEPS      = 450   # 最大 settle 步数（超时保护，梯形曲线下约2个完整往返≈193步/趟）
SETTLE_SPEED      = 0.06  # settle 运动峰值速度 (m/s)
SETTLE_ACCEL      = 0.04  # settle 加速度 (m/s²)，梯形速度曲线用
                          # 用恒速bang-bang会让vs/ac在大部分时间里是常数，
                          # RLS无法把ks(对应st)和b(对应vs)的贡献分离
                          # (条件数→inf)。梯形曲线让vs连续变化、ac非零，
                          # 两个参数才有可辨识性。
GPR_MIN_DATA      = 20

# ── RLS 收敛判据 ─────────────────────────────────────────
# 预测误差阈值不直接引用真值 noise_std，改用 explore 早期
# 窗口标定出的 noise_std_est（运行时计算，见 run_sim 内部）。
# 收敛 = 有激励(几何判据) AND 力预测误差小 AND P矩阵不确定度已收缩。
# 激励判据改用几何量 (dist_cur - dist_taut)，不依赖力的绝对大小 ——
# 力阈值判据对软材料(ks小)可能系统性地达不到门槛，几何判据无此问题。
# P矩阵判据替代"看ks_est最近N步变化量"——P是RLS自身维护的不确定度，
# 比间接观察参数轨迹更直接，归一化后不需要为每种材料重新调阈值。
EXCITATION_FRAC   = 0.3    # settle 阶段至少走出 STRETCH_MAX 的这个比例才算有激励
PRED_ERR_MULT     = 2.0    # 预测误差门槛 = PRED_ERR_MULT * noise_std_est
UNCERTAINTY_RATIO_THRESH = 0.15  # P矩阵迹收缩到初始值的这个比例以下才算稳定

# ── 切向探索阶段(径向settle结束后，进 arc 前) ──────────────
# 背景：径向 settle 扫描全程 theta 几乎不变(纯径向往返，不转动)，
# 但 arc 阶段会让 theta 走出 settle 时的角度。GPR 核函数对没见过的
# theta 区域会给出高 σ，导致 local_ratio 在 arc 阶段大 stretch 区域
# (这正是 GPR 该发挥作用的地方)经常因为 σ 超阈值而退回 RLS——实测
# 用渐进硬化材料(arc_deg=90)验证：放宽 theta 的 RBF length_scale
# 上界(5→10)对此没有效果，核函数已经主动把 theta 这一维推到了
# 当时的上界，说明瓶颈不在核函数超参数，而在训练数据本身缺乏角度
# 多样性——这是只能靠"真的让 GPR 见过别的角度"解决的问题。
#
# 设计原则(Hank要求)：摆动幅度尽可能小，但要让 GPR 真正受益；
# 不为节省时间牺牲探索质量，时间不是这个阶段的约束。因此幅度不是
# 拍一个固定角度，而是用 GPR 自己的 σ 反馈做自适应判据：从很小的
# 摆动开始，每完成一次完整摆动就检查 GPR 在边界角度处的 σ 是否已经
# 达标(达标=GPR 真的学会了这个角度范围内的力学行为)，没达标就继续
# 小幅扩大摆动范围，直到达标或触达上限(防止极端情况无限扩大)。
#
# 摆动期间维持的 stretch 水平复用径向扫描已经验证安全的
# max_stretch_seen(或 EXCITATION_FRAC*STRETCH_MAX，取较大者)——这是
# 已经确认过 GPR 真正需要更多角度覆盖的区域，stretch 很小时材料近似
# 线性、GPR 在那没有用武之地，不需要在那花时间做切向探索。
TANGENTIAL_AMP_INIT     = np.radians(3)    # 初始摆动半幅，从很小开始
TANGENTIAL_AMP_STEP     = np.radians(3)    # 每轮未达标后的扩大步长
TANGENTIAL_AMP_MAX      = np.radians(30)   # 摆动半幅上限，防止无限扩大
TANGENTIAL_SIGMA_TARGET = 0.3              # 边界处σ达标阈值，复用
                                            # local_ratio 运行时同样的
                                            # sigma_threshold，确保探索
                                            # 阶段追求的目标和 arc 阶段
                                            # 实际判断"GPR是否可信"用的
                                            # 是同一把尺子
TANGENTIAL_SPEED        = 0.08             # 切向摆动速度，不要求快，
                                            # 留有余量(时间不是约束)

# 切向探索结束后，stretch 仍停留在 tangential_probe_stretch 这个较大
# 水平(实测线性ks=30材料下高达0.234m，接近STRETCH_MAX)，若直接进 arc，
# 远超 arc 阶段"回归子阶段"设计时预想的小缺口——回归子阶段是为处理
# settle 结束位置和 R_ref 之间的小缺口设计的，应付不了这么大的初始
# 缺口。实测直接转换会让起始力远超 f_taut(最高到7N+，f_taut通常<1N)，
# viol 从 0% 升到 6%+。改为先转 phase=1.6 做径向收回，回到 dist_taut
# 附近后再正式进 arc，让 arc 阶段开始时的状态和没有切向探索时一致。
TANGENTIAL_RETRACT_SPEED = 0.08            # 径向收回速度，同样不求快
TANGENTIAL_RETRACT_TOL   = 0.01            # 收回到 dist_taut 这个容差内即视为完成


# ══════════════════════════════════════════════════════
# BOCD
# ══════════════════════════════════════════════════════
class BOCD:
    def __init__(self, hazard=0.05, mu0=0.0, kappa0=1., alpha0=3., beta0=0.5):
        self.h=hazard; self.mu0=mu0; self.kappa0=kappa0
        self.alpha0=alpha0; self.beta0=beta0
        self.mu=np.array([mu0]); self.kappa=np.array([kappa0])
        self.alpha=np.array([alpha0]); self.beta=np.array([beta0])
        self.R=np.array([1.]); self.prev_mode_r=0

    def update(self, x):
        df=2*self.alpha
        scale=np.maximum(np.sqrt(self.beta*(self.kappa+1)/(self.alpha*self.kappa)),1e-10)
        lp=np.clip(student_t.logpdf(x,df=df,loc=self.mu,scale=scale),-50,50)
        pi=np.exp(lp)
        Rn=np.empty(len(self.R)+1)
        Rn[0]=np.sum(self.R*pi)*self.h; Rn[1:]=self.R*pi*(1-self.h)
        s=Rn.sum(); Rn=Rn/s if s>0 else np.array([1.]+[0.]*len(self.R))
        self.R=Rn
        mode_r=int(np.argmax(Rn))
        sig=max(0, self.prev_mode_r-mode_r)/max(1, self.prev_mode_r)
        self.prev_mode_r=mode_r
        kn=self.kappa+1; mn=(self.kappa*self.mu+x)/kn
        an=self.alpha+0.5; bn=self.beta+self.kappa*(x-self.mu)**2/(2*kn)
        self.mu=np.append([self.mu0],mn); self.kappa=np.append([self.kappa0],kn)
        self.alpha=np.append([self.alpha0],an); self.beta=np.append([self.beta0],bn)
        return sig

# ══════════════════════════════════════════════════════
# RLS（感知层：估计刚度，动态维护力边界）
# ══════════════════════════════════════════════════════
class RLS:
    def __init__(self, lam=0.97, lam_settle=0.88):
        self.lam=lam
        self.lam_settle=lam_settle   # settle 阶段用更低的遗忘因子，收敛更快
        self.lam_cur=lam_settle      # 初始用 settle 模式
        # 初始 theta 用中性默认值，不贴近任何材料的真实参数 —— 若初始值
        # 恰好接近 ks_true，图上会出现"explore 阶段看起来已经很准"的假象，
        # 实际上 RLS 此时根本没有更新，是误导性的。中性初值能让图准确
        # 反映真实的学习过程。
        self.theta=np.array([1.0, 0.1, 0.01])
        self.P=np.diag([100., 3., 1.])
        self.P0=self.P.copy()              # 初始矩阵，用于按维度归一化
        self.P0_trace = np.trace(self.P)   # 仍保留，用于整体爆炸截断保护

    def set_lam(self, lam):
        self.lam_cur = lam

    def update(self, phi, y, skip_if_no_stretch=False):
        """
        skip_if_no_stretch: 若为 True 且本次 phi[0]（stretch）≈0，跳过更新。
        语义是"stretch 这一步是否携带有效信息"，不是相对上一步的差值
        （早期版本曾用 stretch_delta 这个名字，容易和"变化量"混淆）。
        settle 阶段三角波折返点附近 stretch≈0 但 vel/acc 可能不为零，
        必须单独检查 phi[0]，不能只看 phi 整体范数。
        """
        if skip_if_no_stretch and abs(phi[0]) < 1e-3:
            return self.theta.copy()
        # phi 整体接近零向量时(三者都≈0)，标准 RLS 更新会让 P 矩阵以
        # 1/lam_cur 的速率指数增长而不是收缩 —— 这是无激励步的已知病态。
        # 这里做兜底防护，避免无激励步污染不确定度估计。
        if np.linalg.norm(phi) < 1e-3:
            return self.theta.copy()
        e=y-phi@self.theta; d=self.lam_cur+phi@self.P@phi
        K=self.P@phi/d; self.theta+=K*e
        self.P=(1./self.lam_cur)*(np.eye(3)-np.outer(K,phi))@self.P
        # 数值安全网：只拦截真正的爆炸性增长(远超初始值)，不干扰带遗忘因子
        # RLS 正常的温和回升(lam_cur<1 时即使有效更新也会让 P 略微抬升，
        # 这是为了保留对未来变化的适应能力，不是病态)。
        # 之前把上限设成 P0_trace 本身，结果在有效收敛后又被这个保护摁回
        # 1.000，反而让 uncertainty_ratio 这个指标失去了可读性。
        cur_trace = np.trace(self.P)
        BLOWUP_CAP = 5.0 * self.P0_trace
        if cur_trace > BLOWUP_CAP:
            self.P *= (BLOWUP_CAP / cur_trace)
        self.theta=np.clip(self.theta,[0.05,0.,0.],[200.,5.,2.])
        return self.theta.copy()

    def pred_error(self, phi, fm):
        """力预测误差，用于判断 RLS 是否已收敛到当前工作点"""
        f_pred = phi @ self.theta
        return abs(f_pred - fm)

    def uncertainty_ratio(self):
        """
        只看 ks（P[0,0]）这一维的不确定度，不用整个矩阵的迹。
        理由：导师的指导是"先用简单方法(ks主导的线性近似)，不够再上
        复杂方法(MLP/GPR)"——b、m对应更精细的受力预测，属于下一级
        复杂度，不需要在 settle 阶段就收敛。用全迹会让 b、m 没收敛时
        拖累整体判据，而 ks 才是力预测和距离约束真正依赖的主导参数。
        返回值：ks 不确定度 / 初始 ks 不确定度，越接近 0 说明 ks 估计越稳定。
        """
        return self.P[0, 0] / self.P0[0, 0]

# ══════════════════════════════════════════════════════
# GPR（建模层）
# ══════════════════════════════════════════════════════
class GPRModel:
    def __init__(self, max_data=80):
        # RBF kernel，3维输入 [stretch, vel, theta]
        # theta 特征让 GPR 感知弧线不同角度位置的力变化
        k = (ConstantKernel(1., (0.1, 10.))
             * RBF(length_scale=[0.05, 0.05, 0.5],
                   length_scale_bounds=[(0.005, 1.), (0.005, 1.), (0.05, 5.)])
             + WhiteKernel(0.1, (1e-3, 1.)))
        self.gpr = GaussianProcessRegressor(
            kernel=k, n_restarts_optimizer=0, normalize_y=True)
        self.fitted=False
        self.xm=np.zeros(3); self.xs=np.ones(3); self.max_data=max_data

        # 数据分两个池子管理：
        # - base_X/base_y：settle 阶段的完整扫描数据，在 settle→arc
        #   转换时通过 freeze_base() 一次性锁存，之后不再变动。这是
        #   GPR 训练数据里"覆盖了完整 stretch 扫描范围"的部分。
        # - X/y：滑动窗口，settle 和 arc 阶段都会持续 add_data 进来，
        #   按 max_data 正常淘汰最旧数据。
        # fit() 时两个池子合并训练。
        #
        # 这个分离修复了一个真实 bug：之前只有一个共享滑动窗口，arc
        # 阶段(尤其是"回归子阶段"，机器人被拉回接近 dist_taut 的位置)
        # 持续产生 stretch≈0 附近的新数据，把 settle 阶段好不容易扫出
        # 的大范围 stretch 数据全部挤出窗口——等机器人真正沿弧线推进、
        # stretch 重新变化时，GPR 训练集已经被"局部化"在很窄的范围，
        # 对实际遇到的更大 stretch 是在做严重外推，而不是真正的预测。
        self.base_X=[]; self.base_y=[]
        self.X=[]; self.y=[]

        # local_ratio 输出的平滑状态：相邻两次调用之间结果可能跳动，
        # 这个状态变量让 local_ratio 内部对自己上一次的输出做指数
        # 滑动平均，None 表示还没有过历史值（第一次调用不做平滑）。
        self._ks_smooth_prev = None

    def add_data(self, s, v, theta, f):
        # 输入：[stretch, vel, theta]，theta 让 GPR 感知弧线位置
        self.X.append([s, v, theta]); self.y.append(f)
        if len(self.X) > self.max_data: self.X.pop(0); self.y.pop(0)

    def freeze_base(self):
        """settle→arc 转换时调用一次：把当前滑动窗口里的数据(此时应
        该正好是 settle 阶段积累的完整扫描数据)复制进 base 池永久锁存。
        之后 arc 阶段的新数据只会进入滑动窗口、淘汰滑动窗口里的旧数据，
        不会再触碰 base 池，settle 的激励范围因此不会被冲掉。"""
        self.base_X = list(self.X)
        self.base_y = list(self.y)

    def fit(self):
        all_X = self.base_X + self.X
        all_y = self.base_y + self.y
        if len(all_X) < 15: return   # 数据量 ≥ 15 才 fit，避免早期过拟合
        X = np.array(all_X)
        self.xm = X.mean(0); self.xs = X.std(0) + 1e-6
        self.gpr.fit((X - self.xm) / self.xs, np.array(all_y))
        self.fitted = True

    def predict(self, s, v, theta):
        if not self.fitted: return None, None
        xn = (np.array([[s, v, theta]]) - self.xm) / self.xs
        mu, sig = self.gpr.predict(xn, return_std=True)
        return float(mu[0]), float(sig[0])

    def local_ratio(self, stretch, vel, theta, ks_fallback,
                     sigma_threshold=0.3, smooth_alpha=0.3,
                     min_stretch_for_ratio=0.03):
        """
        用 GPR 在当前点直接预测一次力，除以当前 stretch 得到一个等效
        比例，代替之前用数值微分反推局部刚度的 local_ks。

        为什么放弃数值微分：实测发现 local_ks 在线性材料(真实ks≈11.5)
        下输出系统性偏小(0.5~6之间，远低于真实值)，而且这不是噪声
        问题——固定同一个点反复测试，不同 delta 给出的结果其实相当
        一致(0.57~0.90)，但全都偏离真实值。根因是 GPR 的 RBF 核函数
        本身有平滑特性，拟合出的 mu(stretch) 曲线斜率天然比真实斜率
        更平缓，两点数值微分会把这个核函数自带的平滑偏差直接放大成
        刚度估计的系统性偏差，调 delta 或加平滑都只能压噪声、压不住
        这种系统性偏差。

        改为只查询一次 GPR，直接用预测力本身（GPR 学到的非线性信息
        没有被中间的微分步骤压缩、丢失），除以 stretch 得到等效比例：
            ratio = f_gpr(stretch, vel, theta) / stretch
        候选点预测时用 ratio*stretch_候选点 做线性外推——这仍然是"用
        GPR这一次查询的结果，外推到 MPC 内部的所有候选点"，满足"每
        步只查一次GPR"的性能约束，但不再经过数值微分这一步有损变换。

        min_stretch_for_ratio 是除了 σ 阈值之外的第二道独立防线：
        实测发现 ks=30 线性材料下，GPR 被采用的全部 11 次都发生在
        stretch 极小(0.0004~0.02)的时刻——σ 在这些点上确实够低(GPR
        "认为"自己有把握)，mu 本身也合理(0.7~1.5N，量级正常)，但
        ratio=mu/stretch 这个除法在分母接近零时，mu 的微小误差会被
        放大到离谱(真实ks=30，实测ratio最高到709)。σ衡量的是"对 mu
        这个值有多大把握"，不衡量"用 mu 做除法是否数值安全"——这是
        两件不同的事，σ阈值这道防线管不到这个问题，必须单独加一道
        基于 stretch 量级本身的保护：stretch 小于此值时，不论 σ 多
        低，一律退回 ks_fallback，因为这时候"等效比例"概念本身是
        病态的(材料几乎没有被拉伸，用一个微弱的力推算"刚度"，物理
        上就是噪声主导)。0.03m 是 STRETCH_MAX(0.25m 典型值)的约
        12%，覆盖了 settle 刚结束、arc 刚开始的"回归子阶段"附近
        stretch 还很小的过渡期。

        smooth_alpha 的指数滑动平均同样保留，抑制相邻步之间的残余
        跳变。
        """
        if not self.fitted:
            self._ks_smooth_prev = None
            return ks_fallback
        if stretch < min_stretch_for_ratio:
            self._ks_smooth_prev = None
            return ks_fallback
        mu, sigma_cur = self.predict(stretch, vel, theta)
        if mu is None or sigma_cur is None or sigma_cur > sigma_threshold:
            self._ks_smooth_prev = None
            return ks_fallback
        ratio_raw = max(0.05, mu / stretch)

        if self._ks_smooth_prev is None:
            ratio = ratio_raw
        else:
            ratio = smooth_alpha * ratio_raw + (1 - smooth_alpha) * self._ks_smooth_prev
        self._ks_smooth_prev = ratio
        return ratio

# ══════════════════════════════════════════════════════
# 安全回缩
# ══════════════════════════════════════════════════════
def retract_velocity(p_cur, anchor_est, speed=0.1):
    d = anchor_est - p_cur
    norm = np.linalg.norm(d)
    if norm < 1e-6: return np.zeros(3)
    return d/norm * speed

# ══════════════════════════════════════════════════════
# Tube-MPC（动态速度上限）
# ══════════════════════════════════════════════════════
def run_mpc_3d(p_cur, v_cur, ks_eff, b_rls, m_rls,
               anchor_est, dist_taut, f_taut,
               theta_cur, theta_target,
               STRETCH_MAX, f_max_safe,
               gpr_model=None, gpr_sigma_thresh=0.3,
               sigma_gpr=0., v_max_cur=0.1, R_ref=None, w_time=None):
    """
    主约束是距离约束，力作为"任务引导"代价项（非安全判据）

    主约束（硬）：dist_taut <= dist <= dist_taut + STRETCH_MAX
      - dist_taut：BOCD 触发时的距离，一次性测量，固定不变
      - STRETCH_MAX：临床输入，与 ks 无关
      - 这是唯一的安全边界，"绷紧"是标准，不是力的大小
    任务引导（软）：力代价项把 f_pred 拉向 f_taut（刚好绷紧时的力）
      - 这不是安全判据，是轨迹质量的引导信号 —— 之前只有角度推进
        (W_ANGLE)和距离硬约束，MPC在可行域内没有偏好，导致轨迹质量差
      - f_taut 引导让 MPC 主动维持"刚好绷紧"的状态去传导运动，
        同时距离硬约束依然兜底，引导不会突破安全边界
    径向约束：保持弧线轨迹半径稳定

    力预测(pf)的有效刚度优先用 GPR 给出的局部估计，GPR 没有把握
    (σ过高/数据不足)时退回 RLS 的 ks_eff——这是大纲原始设计"GPR 为
    主、RLS 为降级后备"，之前实现里被颠倒成了"RLS 全程主导，GPR 只管
    σ"，已用多种子对比实验验证修正方向：σ 和 GPR 实际预测误差强相关
    (相关系数0.83，迟滞材料测试)，证明 GPR 在犯大错时确实"知道自己
    不确定"，σ阈值切换并非凭空假设。

    GPR 只在每次 MPC 求解开始前查询一次（不是在 pf 内部对每个候选点
    查询）——这是吃过亏才确认下来的关键点：大纲原文是"每次滚动优化
    前利用新数据更新GPR后验分布"，即粒度是"每一步求解一次"，不是
    "每个候选点一次"。第一版实现把 gpr.predict 直接放进了 pf 内部，
    而 pf 在一次 SLSQP 求解里会被调用上千次(SLSQP迭代数×MPC_N候选
    点)，实测单步求解时间从几秒暴涨到100+秒。

    第二版改为求解前调用一次 gpr_model.local_ks(...)，对 GPR 当前
    点附近做数值微分反推等效局部刚度。但实测发现这个数值微分系统性
    低估刚度（线性ks=10材料下，local_ks 输出在 0.5~6 之间，真实值
    11.5）——根因不是噪声，是 RBF 核函数本身的平滑特性让拟合曲线
    的斜率天然比真实斜率更平缓，两点数值微分把这个平滑偏差直接放大
    成系统性的刚度低估，加大 delta 或加平滑都只能压随机噪声、压不住
    这种系统性偏差。

    现在改用 gpr_model.local_ratio(...)：只查询一次 GPR 得到当前点
    的预测力，直接用 力/stretch 算等效比例，不经过数值微分这一步
    有损变换，GPR 学到的非线性信息没有被压缩丢失。把这个比例当成一
    个和 RLS 的 ks_eff 等价的标量传给 pf，pf 内部恢复成原来的纯线性
    公式做候选点外推，不再含任何 GPR 调用。gpr_sigma_thresh=0.3 复用
    local_ratio 同样在用的σ阈值。gpr_model=None 时完全退化为原来的
    纯 RLS 行为。

    STRETCH_MAX、f_max_safe 显式作为参数传入（而非闭包读取模块级
    全局变量），使函数在不同实验配置（不同材料/临床设定）下可安全
    复用，不依赖调用前对全局变量的修改。w_time 同理，默认 None 时
    回退到模块级 W_TIME。
    """
    if w_time is None:
        w_time = W_TIME
    if R_ref is None:
        R_ref = np.linalg.norm(p_cur - anchor_est)

    # 距离约束边界（核心，基于纯可观测量）
    dist_lower = dist_taut                    # 不松弛
    dist_upper = dist_taut + STRETCH_MAX      # 不过拉
    # GPR sigma 用于收紧上界（Tube-MPC 思想）
    dist_upper_eff = dist_upper - np.clip(sigma_gpr * 0.5, 0., STRETCH_MAX * 0.3)
    dist_upper_eff = max(dist_upper_eff, dist_lower + 0.01)  # 保证可行域非空

    # GPR 局部刚度：只在这里查询一次(对应当前实际位置)，不在 pf 内部
    # 逐候选点查询。返回值和 RLS 的 ks_eff 是同一个量纲、同一种用法，
    # GPR 没把握时函数内部自己退回 ks_eff，调用方不需要再判断。
    if gpr_model is not None:
        cur_stretch = max(0., np.linalg.norm(p_cur - anchor_est) - dist_taut)
        cur_vel = np.linalg.norm(v_cur)
        ks_eff_for_pf = gpr_model.local_ratio(
            cur_stretch, cur_vel, theta_cur, ks_fallback=ks_eff,
            sigma_threshold=gpr_sigma_thresh)
    else:
        ks_eff_for_pf = ks_eff

    # 力预测（用于代价函数，不作硬约束）。轻量纯算术，不含任何 GPR
    # 调用——GPR 的影响已经通过上面算好的 ks_eff_for_pf 这一个标量
    # 体现，pf 在 SLSQP 内部被反复调用时不会再触发任何 GPR 推断。
    def pf(p_, v_, a_):
        stretch_ = max(0., np.linalg.norm(p_ - anchor_est) - dist_taut)
        return max(0., ks_eff_for_pf * stretch_ + b_rls * np.linalg.norm(v_)
                   + m_rls * np.linalg.norm(a_))

    # 距离辅助函数
    def dist_from_anchor(p_):
        return np.linalg.norm(p_ - anchor_est)

    def cost(u_flat):
        u_seq = u_flat.reshape(MPC_N, 3)
        p = p_cur.copy(); v = v_cur.copy(); c = 0.
        W_FORCE_GUIDE = 8.0  # 力引导权重：把预测力拉向f_taut（柔性偏好，非安全判据）
        for k in range(MPC_N):
            vk = u_seq[k]; ak = (vk - v) / dt; pn = p + dt * vk
            f_pred = pf(pn, vk, ak)
            # 力引导代价：维持在刚好绷紧的力附近，引导轨迹质量
            c += W_FORCE_GUIDE * (f_pred - f_taut) ** 2
            # 任务进度：角度推进
            c += w_time
            tn = np.arctan2(pn[1] - anchor_est[1], pn[0] - anchor_est[0])
            pg = tn - theta_cur
            if pg < -np.pi: pg += 2 * np.pi
            c -= W_ANGLE * np.clip(pg, 0., 0.15)
            p = pn; v = vk
        return c

    cons = []
    for k in range(MPC_N):
        # 距离上界约束：dist <= dist_upper_eff（不过拉）
        def make_dist_upper(k_=k):
            def fn(u):
                p = p_cur.copy(); v = v_cur.copy()
                for i in range(k_ + 1): v = u.reshape(MPC_N, 3)[i]; p = p + dt * v
                return dist_upper_eff - dist_from_anchor(p)
            return fn
        # 距离下界约束：dist >= dist_lower（不松弛）
        def make_dist_lower(k_=k):
            def fn(u):
                p = p_cur.copy(); v = v_cur.copy()
                for i in range(k_ + 1): v = u.reshape(MPC_N, 3)[i]; p = p + dt * v
                return dist_from_anchor(p) - dist_lower
            return fn
        # 径向约束：保持弧线半径（不偏离 R_ref 超过 R_TOL）
        def make_radial_upper(k_=k):
            def fn(u):
                p = p_cur.copy(); v = v_cur.copy()
                for i in range(k_ + 1): v = u.reshape(MPC_N, 3)[i]; p = p + dt * v
                return R_ref + R_TOL - dist_from_anchor(p)
            return fn
        def make_radial_lower(k_=k):
            def fn(u):
                p = p_cur.copy(); v = v_cur.copy()
                for i in range(k_ + 1): v = u.reshape(MPC_N, 3)[i]; p = p + dt * v
                return dist_from_anchor(p) - (R_ref - R_TOL)
            return fn
        cons.append({'type': 'ineq', 'fun': make_dist_upper(k)})
        cons.append({'type': 'ineq', 'fun': make_dist_lower(k)})
        cons.append({'type': 'ineq', 'fun': make_radial_upper(k)})
        cons.append({'type': 'ineq', 'fun': make_radial_lower(k)})

    tang = np.array([-np.sin(theta_cur), np.cos(theta_cur), 0.])
    u0 = np.tile(tang * 0.05, MPC_N)
    res = minimize(cost, u0, method='SLSQP',
                   bounds=[(-v_max_cur, v_max_cur)] * (MPC_N * 3),
                   constraints=cons,
                   options={'maxiter': 40, 'ftol': 2e-3})
    if res.success:
        return res.x.reshape(MPC_N, 3)[0]
    else:
        # Fallback：径向回缩朝向 anchor，stretch 减小力必然减小。
        # 之前的判定是 `res.success or res.fun < 1e8`，但 1e8 这个阈值
        # 极其宽松，几乎任何代价函数值都满足，导致这条件形同虚设——
        # 即使 SLSQP 没有真正收敛（约束很可能被违反），不可靠的解也会
        # 被直接采用，fallback 从未被触发。改为只信任 res.success，
        # 未真正收敛时一律走安全回缩。
        #
        # 但 retract_velocity 本身只知道"朝 anchor 方向走"，不知道
        # dist_lower 这条线在哪——如果连续多步都未收敛(实测出现过连续
        # 13步)，累积的回缩量会穿过 dist_lower，造成距离下界违反(机器人
        # 比绷直点更松弛，这是我们唯一不允许妥协的安全底线)。
        # 这里限制回缩速度，确保单步移动不会越过 dist_lower。
        cur_dist = dist_from_anchor(p_cur)
        room_to_lower = max(0., cur_dist - dist_lower)
        max_safe_speed = room_to_lower / dt  # 这一步最多能走多快而不越界
        fallback_speed = min(0.02, max_safe_speed)
        return retract_velocity(p_cur, anchor_est, speed=fallback_speed)

# ══════════════════════════════════════════════════════
# Taubin 圆拟合
# ══════════════════════════════════════════════════════
def taubin_fit(pts):
    if len(pts)<10: return None,None,None
    x=pts[:,0]; y=pts[:,1]
    X=x-x.mean(); Y=y-y.mean(); Z=X**2+Y**2; Zm=Z.mean()
    if Zm<1e-10: return None,None,None
    Z0=(Z-Zm)/(2*np.sqrt(Zm)); A=np.column_stack([Z0,X,Y])
    _,_,V=np.linalg.svd(A); a=V[-1]
    a0=a[0]/(2*np.sqrt(Zm)); a1,a2=a[1],a[2]
    if abs(a0)<1e-10: return None,None,None
    xc=-a1/(2*a0); yc=-a2/(2*a0)
    val=a1**2+a2**2-4*a0*(a0*Zm-Z0.mean())
    return (xc,yc,np.sqrt(val)/abs(2*a0)) if val>0 else (None,None,None)

# ══════════════════════════════════════════════════════
# 主仿真函数
# ══════════════════════════════════════════════════════
def run_sim(force_model, arc_deg=90, STRETCH_MAX=0.25, f_max_safe=3.0,
            w_time=None, L0_true=0.5, noise_std=0.1, seed=42, verbose=True):
    """
    单次仿真入口：给定材料模型/任务/临床参数，跑完整的三阶段流程并返回结果。

    这是 integrated_sim_3d.py 的核心仿真逻辑，被本文件末尾的"单次运行+
    画图"入口调用一次（默认参数），也被 run_experiment_matrix.py 调用
    多次（不同材料/任务组合）。两者共享同一份逻辑，不再各自维护一份
    可能逐渐不同步的副本。

    材料通过 force_model 这个可插拔对象传入（见 material_models.py），
    而不是裸的 ks_true 标量——这样线性弹簧只是众多材料模型里最简单的
    一种特例（LinearSpring），渐进硬化/迟滞/各向异性等非线性材料用
    同一个 run_sim 就能跑，不需要另外维护一份几乎重复的仿真逻辑。
    所有材料模型实现统一接口：
        force_model.compute(stretch, vel, acc, direction_angle=0.) -> 力大小

    控制器逻辑（BOCD/RLS/GPR/MPC/距离约束）完全不知道、也不需要知道
    传入的是哪种材料模型——这是这个设计的核心：如果架构合理，不应该
    关心 true_force_3d 内部用的是哪种力学模型。

    参数
    ----
    force_model : 实现 compute(...) 接口的材料模型对象
                  (LinearSpring / HardeningSpring / HysteresisSpring /
                  PiecewiseAnisoSpring，见 material_models.py)
    arc_deg : float        目标弧线角度 (度)，默认90°
    STRETCH_MAX : float    临床输入：最大允许位移 (m)
    f_max_safe : float     临床输入：安全力阈值 (N)，仅作观测，不参与决策
    w_time : float|None    MPC 时间代价权重，None 时用模块级默认 W_TIME
    L0_true : float        材料自然长度 (m)，几何量，与材料力学行为无关，
                            所有材料模型共用同一个真实自然长度
    noise_std : float      传感器噪声标准差，仅用于生成仿真观测
    seed : int              随机种子
    verbose : bool          是否打印过程日志（实验矩阵批量跑时应设 False）

    返回
    ----
    dict，包含：
      - 'status': 'OK' 或 'FAILED'（FAILED 表示没能在 T_sim 步内绷直）
      - 轨迹/历史数据（用于画图）：hp, hf, hsig, hks, hgmu, hgsig,
        hphase, h_dist_taut, h_vmax
      - 关键标量：dist_taut, f_taut, anchor_est, theta_start, theta_target,
        noise_std_est, R_fit, R_std, xc, yc
      - 实验指标：ks_settle, ks_final, rms, viol, dist_viol_upper,
        dist_viol_lower, total_steps, arc_steps_count
        （ks_settle/ks_final 是 RLS 估计的等效线性刚度——线性材料下
        逼近真实 ks；非线性材料下是个局部近似值，不对应单一真值）
    """
    np.random.seed(seed)
    anchor = np.array([0.0, 0.0, 0.0])
    if w_time is None:
        w_time = W_TIME
    arc_rad = np.radians(arc_deg)

    # true_force_3d 定义在函数内部（而非模块级），依赖本次调用传入的
    # force_model 和 noise_std，避免不同实验组之间因共享模块级变量而
    # 产生状态污染。
    #
    # 力的大小完全由 force_model.compute(...) 决定，不再有写死的线性
    # 公式。direction_angle 从 d=p-anchor 的方向角算出，传给各向异性
    # 模型用（线性/硬化/迟滞模型会忽略这个参数，签名兼容但不使用）。
    def true_force_3d(p, vel=0.0, acc=0.0):
        d = p - anchor; dist = np.linalg.norm(d)
        noise = np.random.normal(0, noise_std, 3)
        if dist < 1e-6: return noise
        stretch = max(0.0, dist - L0_true)
        direction_angle = np.arctan2(d[1], d[0])
        f = max(0.0, force_model.compute(stretch, vel, acc,
                                          direction_angle=direction_angle))
        if f < 1e-6: return noise
        return f * d/dist + noise

    T_sim        = 800
    theta_start  = np.radians(-20)
    theta_target = theta_start + arc_rad

    R_init      = 0.18   # 初始距离，不依赖 L0_true（治疗师布置机器人位置）
    p_cur       = anchor + R_init * np.array([np.cos(theta_start), np.sin(theta_start), 0.])
    v_cur       = np.zeros(3); v_prev = np.zeros(3)
    anchor_est  = anchor.copy()

    bocd = BOCD(); rls = RLS(); gpr = GPRModel()

    # 三阶段状态机
    phase = 0        # 0:探索  1:settle  2:弧线
    taut_step = None; arc_step = None
    arc_steps = 0; settle_steps_done = 0
    f_taut = 0.2
    sigma_s = 0.; R_ref = None
    returning_to_target = False   # arc 开始时是否在"回归 R_ref"子阶段
    RETURN_TOL = 0.015            # 回归完成判据：|dist-R_ref| 小于此值

    # dist_taut 取代了早期版本的 L0_est，BOCD 触发时一次性记录
    dist_taut = None   # 绷直点到锚点的距离，= 近似 L0

    # 噪声标定（explore早期窗口，纯可观测，与材料无关）
    noise_samples = []
    noise_std_est = 0.05   # 兜底初值，标定完成后会被覆盖

    hp = [p_cur.copy()]; hf = []; hsig = []; hks = []
    hgmu = []; hgsig = []; hL0 = []; hphase = []
    h_dist_taut = []   # 记录 dist_taut 的演化（调试用）
    h_vmax = []
    f_cur = true_force_3d(p_cur)

    if verbose:
        print("="*62)
    if verbose:
        print("Integrated 3D Simulation  —  大纲四层架构 v15")
    if verbose:
        print(f"[真实系统(不可见)] force_model={force_model!r}  L0_true={L0_true}m")
    if verbose:
        print(f"[临床输入] f_max_safe={f_max_safe}N  STRETCH_MAX={STRETCH_MAX}m")
    if verbose:
        print(f"起始θ={np.degrees(theta_start):.0f}°  目标θ={np.degrees(theta_target):.0f}°")
    if verbose:
        print(f"速度: {V_MIN}→{V_MAX} m/s (线性爬升 {V_RAMP} 步)")
    if verbose:
        print(f"Explore速度: {V_EXPLORE} m/s  Settle最大步数: {SETTLE_STEPS}")
    if verbose:
        print(f"噪声标定窗口: 前{NOISE_CALIB_STEPS}步  激励判据: stretch>{EXCITATION_FRAC}*STRETCH_MAX")
    if verbose:
        print("="*62)

    for t in range(T_sim):
        fm = np.linalg.norm(f_cur); hf.append(fm)
        sig = bocd.update(fm); hsig.append(sig)

        theta_cur = np.arctan2(p_cur[1] - anchor_est[1], p_cur[0] - anchor_est[0])
        dist_cur  = np.linalg.norm(p_cur - anchor_est)

        # ── 阶段转换 ─────────────────────────────────────────
        # 0→1：BOCD 检测到绷直，进入 settle
        if phase == 0 and sig > 0.35:
            phase = 1; taut_step = t; settle_steps_done = 0
            max_stretch_seen = 0.   # 记录settle过程中走到过的最大stretch
            settle_dir = 1          # settle运动方向，1=外推，-1=回收，从外推开始
            settle_speed_signed = 0.  # 梯形速度曲线的当前带符号速度
            f_taut = max(fm, 0.1)
            # dist_taut 直接测量得到，取代早期版本的 L0_est
            # 物理含义：绷直点距离 ≈ 自然长度（保守高估，安全方向）
            dist_taut = dist_cur
            # 噪声水平用 explore 早期窗口标定，不直接引用真值 noise_std
            if len(noise_samples) >= 5:
                noise_std_est = max(float(np.std(noise_samples)), 0.01)
            if verbose:
                print(f"  [BOCD]  t={t:3d}: 检测到绷直 → 进入 settle 阶段")
            if verbose:
                print(f"          sig={sig:.3f}  f_taut={f_taut:.3f}N  dist_taut={dist_taut:.3f}m")
            if verbose:
                print(f"          噪声标定 noise_std_est={noise_std_est:.3f}N (基于{len(noise_samples)}个早期样本)")
            if verbose:
                print(f"          距离约束: [{dist_taut:.3f}, {dist_taut+STRETCH_MAX:.3f}] m")

        # 1→2：settle 完成（RLS 收敛 或 步数超时），进入弧线
        if phase == 1 and dist_taut is not None:
            # 收敛判据：三个条件都满足才算收敛
            #   (a) 有激励：settle过程中走到过的最大stretch占STRETCH_MAX的比例够大
            #       （用历史最大值而非瞬时值 —— settle是三角波往返扫描，瞬时值
            #       在折返点附近会回到≈0，用瞬时值判断会导致"刚激励完就被判定
            #       为无激励"的误判）
            #   (b) 预测准：力预测误差 < PRED_ERR_MULT * noise_std_est（标定值，非真值）
            #   (c) 估计稳：P矩阵不确定度已收缩到初始值的一定比例以下
            # 单独的预测误差判据在 stretch≈0 时会假阳性（任何 theta 都能拟合 0），
            # 加上激励条件后这个假阳性被排除。
            # 但激励+预测准仍可能在 theta 还在剧烈调整时就"刚好"满足（参数过冲
            # 途中预测误差恰好够小），加入 P 矩阵判据确保估计本身已经稳定下来，
            # 不是冻结一个还在变化过程中的瞬时值。
            st_settle  = max(0., dist_cur - dist_taut)
            max_stretch_seen = max(max_stretch_seen, st_settle)
            vs_settle  = np.linalg.norm(v_cur)
            ac_settle  = np.linalg.norm((v_cur - v_prev) / dt)
            phi_settle = np.array([st_settle, vs_settle, ac_settle])
            pred_err   = rls.pred_error(phi_settle, fm)
            pred_err_thresh = PRED_ERR_MULT * noise_std_est
            unc_ratio  = rls.uncertainty_ratio()

            has_excitation = max_stretch_seen > EXCITATION_FRAC * STRETCH_MAX
            is_stable      = unc_ratio < UNCERTAINTY_RATIO_THRESH
            rls_converged  = has_excitation and (pred_err < pred_err_thresh) and is_stable
            settle_done    = settle_steps_done >= SETTLE_STEPS
            if rls_converged or settle_done:
                phase = 1.5  # 径向扫描完成，先做切向探索，再进 arc
                tangential_steps_done = 0
                tangential_theta_start = theta_cur  # 径向扫描结束时的角度，
                                                      # 即 taut_step 附近角度，
                                                      # 探索摆动以此为中心
                tangential_amp = TANGENTIAL_AMP_INIT  # 当前摆动半幅，从小开始
                tangential_dir_to_target = 1. if theta_target >= theta_cur else -1.
                tangential_swing_dir = 1   # 当前摆动方向：1=朝target方向, -1=往回
                # 维持径向扫描已验证安全、且 GPR 真正需要角度覆盖的 stretch
                # 水平（stretch 很小时材料近似线性，GPR 在那没有用武之地）
                tangential_probe_stretch = max(max_stretch_seen,
                                                EXCITATION_FRAC * STRETCH_MAX)

                reason = (f"RLS收敛(stretch={st_settle:.3f}m, err={pred_err:.3f}N<{pred_err_thresh:.3f}N, "
                          f"P比={unc_ratio:.3f}<{UNCERTAINTY_RATIO_THRESH})"
                          if rls_converged else
                          f"超时({settle_steps_done}步, 激励={has_excitation}, "
                          f"err={pred_err:.3f}N, P比={unc_ratio:.3f})")
                if verbose:
                    print(f"  [Settle→切向探索] t={t:3d}: {reason}")
                if verbose:
                    print(f"               径向ks={rls.theta[0]:.2f}  "
                          f"探索水平stretch={tangential_probe_stretch:.4f}m")

        # ── 阶段 1.5：切向探索（让 GPR 见过 settle 角度之外的力学行为）
        # 维持 stretch≈tangential_probe_stretch，在 tangential_theta_start
        # 朝 theta_target 方向小角度摆动。每摆到一次边界，检查 GPR 在该
        # 边界角度处的 σ 是否已经达标；未达标则扩大摆幅继续摆，达标或
        # 触达上限则结束探索——但不直接进 arc，先转 phase=1.6 做径向
        # 收回（见下方阶段 1.6 的注释，说明为什么不能直接进 arc）。
        if phase == 1.5:
            target_theta = (tangential_theta_start
                             + tangential_dir_to_target * tangential_swing_dir * tangential_amp)
            mu_b, sigma_at_boundary = gpr.predict(tangential_probe_stretch, 0.,
                                                    target_theta)
            sigma_ok = (sigma_at_boundary is not None
                        and sigma_at_boundary < TANGENTIAL_SIGMA_TARGET)
            amp_capped = tangential_amp >= TANGENTIAL_AMP_MAX
            near_start_angle = abs(theta_cur - tangential_theta_start) < 0.02
            # 至少摆动一轮(tangential_steps_done > 5 防止刚进入这个
            # 阶段、还没真正摆动过就因为"已经在起始角度附近"而误判完成)
            if (sigma_ok or amp_capped) and near_start_angle and tangential_steps_done > 5:
                phase = 1.6
                if verbose:
                    sig_str = f"{sigma_at_boundary:.3f}" if sigma_at_boundary is not None else "N/A"
                    print(f"  [切向探索→径向收回] t={t:3d}: 摆幅={np.degrees(tangential_amp):.1f}° "
                          f"σ边界={sig_str} (达标={sigma_ok}, 触顶={amp_capped})")

        # ── 阶段 1.6：径向收回（探索后回到 dist_taut 附近，再进 arc）
        # 切向探索结束时 stretch 仍停留在 tangential_probe_stretch 这个
        # 较大水平(实测线性ks=30材料下高达0.234m，接近STRETCH_MAX)，若
        # 直接进 arc，远超 arc 阶段"回归子阶段"设计时预想的小缺口——
        # 回归子阶段是为处理 settle 结束位置和 R_ref 之间的小缺口设计
        # 的，应付不了这么大的初始缺口。实测直接转换会让起始力远超
        # f_taut(最高到7N+，f_taut通常<1N)，viol 从 0% 升到 6%+。这里
        # 先做径向收回，回到 dist_taut 附近后再正式进 arc，让 arc 阶段
        # 开始时的状态和没有切向探索时一致。
        if phase == 1.6 and (dist_cur - dist_taut) < TANGENTIAL_RETRACT_TOL:
            phase = 2; arc_step = t
            returning_to_target = True
            rls.set_lam(rls.lam)
            # arc 阶段不再更新 RLS，冻结 settle+切向探索阶段学到的 theta
            # (必须先冻结，R_ref 的计算依赖 ks_frozen)
            theta_frozen = rls.theta.copy()
            ks_frozen = max(theta_frozen[0], 0.5)  # 防止除零/极小值

            # 锁存 GPR 训练数据：径向扫描 + 切向探索两部分数据都已经
            # 在滑动窗口里，此刻复制进 base 池永久保护，不会被 arc
            # 阶段(尤其回归子阶段，机器人在 dist_taut 附近停留)产生
            # 的新数据挤出窗口。
            gpr.freeze_base()

            # R_ref 不再用固定的 STRETCH_MAX 比例，改为从 f_taut 反推。
            # 之前 R_ref = dist_taut + 0.2*STRETCH_MAX 是一个和材料
            # 刚度无关的固定拉伸比例，但力引导代价项的目标是把力拉向
            # f_taut —— 同样的固定拉伸量，软材料下力还在 f_taut 附近，
            # 硬材料下力远超 f_taut。径向硬约束(R_ref±R_TOL)会把机器人
            # 摁在固定拉伸量对应的位置，软目标(力引导)够不到它真正
            # 想要的位置，两者打架，结果是硬约束赢、力失控
            # (材料越硬，问题越明显，这正是 viol 随 ks 上升到 100%
            # 的根因)。
            # 改为 R_ref 直接对应"力维持在 f_taut 附近"这个目标本身：
            # stretch_target = f_target / ks_frozen，目标位置和力引导
            # 目标自动对齐，径向约束不会再和它打架。
            # f_taut*1.3（而非1.0）是留出安全裕量，避免目标贴着
            # dist_lower 导致轻微扰动就松弛 —— 具体系数后续可调。
            F_TARGET_MULT = 1.3
            f_target = f_taut * F_TARGET_MULT
            stretch_target = f_target / ks_frozen
            # 安全夹紧：防止 ks_frozen 估计不准时 stretch_target 算出
            # 离谱的值，R_ref 必须落在距离约束的安全范围内
            stretch_target = np.clip(stretch_target, 0., STRETCH_MAX * 0.9)
            R_ref = dist_taut + stretch_target

            if verbose:
                print(f"  [径向收回→Arc] t={t:3d}: dist={dist_cur:.3f}m (dist_taut={dist_taut:.3f}m)")
            if verbose:
                print(f"               冻结 ks={theta_frozen[0]:.2f}  b={theta_frozen[1]:.3f}  m={theta_frozen[2]:.4f}")
            if verbose:
                print(f"               dist_taut={dist_taut:.3f}m  dist_upper={dist_taut+STRETCH_MAX:.3f}m  "
                      f"R_ref={R_ref:.3f}m (f_target={f_target:.3f}N, stretch_target={stretch_target:.4f}m)")

        if phase == 2 and theta_cur >= theta_target - 0.03:
            if verbose:
                print(f"  [Done]  t={t:3d}: θ={np.degrees(theta_cur):.1f}° 到达目标!")
            break

        # 不设基于 fm>=f_max_safe 的强制回缩分支。临床立场：标准是"绷紧
        # 到位"(距离)，不是"受了多少力"——同样的力对软组织可能早已过度
        # 拉伸、对硬组织可能还没绷紧，力不参与安全决策。安全完全由 MPC
        # 内部的距离硬约束(dist_upper_eff)保证。f_max_safe 仅保留用于
        # 绘图记录，不参与控制流程。

        # ── 阶段 0：径向外探索 ───────────────────────────────
        if phase == 0:
            dn = (p_cur - anchor_est) / (dist_cur + 1e-6)
            # 固定探索速度，不依赖任何力的反馈。之前这里有一条
            # "力接近安全阈值时自动减速"的逻辑，用了 f_max_safe 来调节
            # 速度——这是 f_max_safe 在全流程里最后一处功能性使用，和
            # "标准是绷紧、不是力"的立场不一致，已删除。explore 阶段
            # 本身用固定保守速度(V_EXPLORE)探测，不需要额外的力反馈
            # 保护；真正的安全边界由 BOCD 检测到绷直后的距离约束保证。
            u = dn * V_EXPLORE

            # explore 阶段不更新 RLS：stretch 恒为 0（还没绷直），若强行
            # 把 dist 当 stretch 喂进 phi，会让 RLS 学到错误的 theta（被迫
            # 拟合"力不变但 dist 增大"，逼出 ks≈0），污染 settle 阶段的
            # 初始状态。这段时间唯一有效的信息是噪声水平，单独标定。
            if t < NOISE_CALIB_STEPS:
                noise_samples.append(fm)

            hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(0.)
            h_dist_taut.append(0.); h_vmax.append(0.)


        # ── 阶段 1：Settle（径向拉伸扫描，激励 RLS） ────────
        elif phase == 1:
            dn = (p_cur - anchor_est) / (dist_cur + 1e-6)

            # 折返条件是纯几何距离，不含任何力触发分支。临床立场：标准是
            # "绷紧到位"(距离)，不是受力大小——同样的力对不同刚度组织意义
            # 完全不同，力不参与决策。距离约束本身就是唯一的安全边界，
            # settle 主动逼近 STRETCH_MAX 获取最大激励范围(对 ks/b 的可
            # 辨识性也有好处)，但严格小于它，留余量防止梯形曲线 brake_dist
            # 估算误差导致瞬间越界。
            settle_upper = dist_taut + STRETCH_MAX * 0.95
            settle_lower = dist_taut  # 不低于绷直点

            # 梯形速度曲线：匀加速→匀速→接近折返点前匀减速→反向。选择
            # 梯形而非恒速 bang-bang，是因为恒速运动下 vs 全程是常数(只在
            # 折返瞬间变化)、ac 全程为 0，会导致 phi 设计矩阵秩亏(条件数
            # →inf)，RLS 无法分离 ks 和 b 的贡献。梯形曲线让 vs 在大半个
            # 行程里连续变化、ac 在加减速段非零，ks/b/m 三个参数才有
            # 可辨识性。折返判断基于"当前速度刹停还需要的距离"，提前
            # 开始减速，折返只由距离这一个连续、可预测的量决定，不含任何
            # 力触发的瞬时跳变。
            brake_dist = settle_speed_signed**2 / (2 * SETTLE_ACCEL + 1e-9)
            dist_to_upper = settle_upper - dist_cur
            dist_to_lower = dist_cur - settle_lower

            if settle_dir > 0 and dist_to_upper < brake_dist + SETTLE_SPEED * dt:
                settle_dir = -1  # 接近上界，提前刹车后反向
            elif settle_dir < 0 and dist_to_lower < brake_dist + SETTLE_SPEED * dt:
                settle_dir = 1   # 接近下界，提前刹车后反向

            target_speed = settle_dir * SETTLE_SPEED
            speed_err = target_speed - settle_speed_signed
            d_speed = np.clip(speed_err, -SETTLE_ACCEL * dt, SETTLE_ACCEL * dt)
            settle_speed_signed += d_speed
            # 硬边界保护：即使梯形曲线没刹住，也不允许真的越过约束边界
            if dist_cur >= settle_upper:
                settle_speed_signed = min(settle_speed_signed, 0.)
            if dist_cur <= settle_lower:
                settle_speed_signed = max(settle_speed_signed, 0.)

            u = dn * settle_speed_signed
            settle_steps_done += 1

            # RLS 更新（用 dist - dist_taut 作为 stretch）
            st  = max(0., dist_cur - dist_taut)
            vs  = np.linalg.norm(v_cur)
            ac  = np.linalg.norm((v_cur - v_prev) / dt)
            phi = np.array([st, vs, ac])
            # 折返点附近 st≈0 但 vs/ac 可能不为零（机器人仍在运动），此时
            # stretch 这个维度没有激励信息，单独用 st 做门控，不能依赖 phi
            # 整体范数（那只在三者同时≈0时才触发）
            trls = rls.update(phi, fm, skip_if_no_stretch=True)

            # GPR 积累数据
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0: gpr.fit()

            hks.append(trls[0]); hgmu.append(0.); hgsig.append(0.)
            h_dist_taut.append(dist_taut); h_vmax.append(0.)

        # ── 阶段 1.5：切向探索（维持 stretch，摆动 theta） ──────
        elif phase == 1.5:
            target_theta = (tangential_theta_start
                             + tangential_dir_to_target * tangential_swing_dir * tangential_amp)
            target_dist = dist_taut + tangential_probe_stretch
            target_pos = anchor_est + target_dist * np.array(
                [np.cos(target_theta), np.sin(target_theta), 0.])
            err = target_pos - p_cur
            err_norm = np.linalg.norm(err)
            u = (err / (err_norm + 1e-6)) * min(TANGENTIAL_SPEED, err_norm / dt)
            tangential_steps_done += 1

            # 摆到边界(误差已经很小)就翻转方向；是否达标退出在阶段转换
            # 那块统一判断，这里只负责运动和方向切换。
            if err_norm < 0.01:
                if tangential_swing_dir > 0:
                    tangential_swing_dir = -1  # 摆到朝target方向的边界，翻回起始角度
                else:
                    # 摆回起始角度附近，这一轮还没达标(否则已经在阶段转换
                    # 那块退出了)，扩大摆幅再摆一次
                    tangential_swing_dir = 1
                    tangential_amp = min(tangential_amp + TANGENTIAL_AMP_STEP,
                                          TANGENTIAL_AMP_MAX)

            # 切向探索期间也持续积累 GPR 数据，这正是这个阶段存在的目的
            st = max(0., dist_cur - dist_taut)
            vs = np.linalg.norm(v_cur)
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0: gpr.fit()

            hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(0.)
            h_dist_taut.append(dist_taut); h_vmax.append(0.)

        # ── 阶段 1.6：径向收回（探索结束，收回到 dist_taut 附近再进 arc）
        elif phase == 1.6:
            dn = (p_cur - anchor_est) / (dist_cur + 1e-6)
            err_dist = dist_cur - dist_taut
            u = -dn * min(TANGENTIAL_RETRACT_SPEED, max(err_dist, 0.) / dt)

            # 收回过程中 GPR 数据继续积累，不浪费这段时间的数据
            st = max(0., dist_cur - dist_taut)
            vs = np.linalg.norm(v_cur)
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0: gpr.fit()

            hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(0.)
            h_dist_taut.append(dist_taut); h_vmax.append(0.)

        # ── 阶段 2：弧线（MPC，距离约束） ───────────────────
        elif phase == 2:
            st = max(0., dist_cur - dist_taut)
            vs = np.linalg.norm(v_cur)
            ac = np.linalg.norm((v_cur - v_prev) / dt)

            # arc 阶段不再更新 RLS theta，沿用 settle 阶段冻结的估计。
            # 理由：(1) 距离约束做得好时径向移动很小，新增激励有限，继续
            # 更新收益低；(2) 避免小幅抖动数据引入噪声漂移，污染已收敛的
            # 估计。theta_frozen 仍用于力引导代价项和 GPR fallback。
            trls = theta_frozen

            # 回归子阶段：settle 结束时机器人的实际位置和 R_ref（力引导
            # 目标对应的位置）之间通常有缺口——settle 是梯形往返扫描，
            # 收敛退出的瞬间可能停在扫描行程中的任意位置，不一定靠近
            # R_ref。之前直接让 MPC 从这个缺口位置开始画弧线，指望
            # W_FORCE_GUIDE 这个软代价项把它拉回 R_ref，但径向硬约束
            # (R_ref±R_TOL) 同时把机器人摁在缺口位置不让退回去，软目标
            # 够不到，力因此远超 f_taut（且材料越硬越明显）。
            # 这里改为显式地先用简单径向运动把缺口走完，回到 R_ref
            # 附近后再启动 MPC 角度推进，不依赖 MPC 自己解决这个缺口。
            if returning_to_target:
                gap = R_ref - dist_cur
                if abs(gap) < RETURN_TOL:
                    returning_to_target = False
                else:
                    dn_ret = (p_cur - anchor_est) / (dist_cur + 1e-6)
                    u = dn_ret * np.clip(gap / dt, -SETTLE_SPEED, SETTLE_SPEED)
                    hks.append(trls[0]); hgmu.append(0.); hgsig.append(sigma_s)
                    h_dist_taut.append(dist_taut); h_vmax.append(0.)
                    hphase.append(phase)
                    v_prev = v_cur.copy(); v_cur = u.copy()
                    pn = p_cur + dt * u
                    vs2 = float(np.linalg.norm(u)); as2 = float(np.linalg.norm((u - v_prev) / dt))
                    f_cur = true_force_3d(pn, vs2, as2)
                    p_cur = pn; hp.append(p_cur.copy())
                    continue

            # R_ref 缓慢追踪弧线半径，但限幅在安全拉伸带内，
            # 防止漂移回 dist_taut 附近、抵消 settle→arc 时的目标位置设定
            if arc_steps % R_REF_UPDATE == 0:
                R_ref_new = 0.8 * R_ref + 0.2 * dist_cur
                R_ref = np.clip(R_ref_new,
                                dist_taut + 0.05 * STRETCH_MAX,
                                dist_taut + 0.6 * STRETCH_MAX)

            # GPR 更新：现在不只提供 σ 收紧距离约束上界，均值预测 mu
            # 也会在 run_mpc_3d 的 pf 函数里被实际用于力预测(σ 足够小
            # 时优先于 RLS)。这里同步记录真正参与决策的预测值，不再
            # 像之前那样把 RLS 的值伪装成"GPR显示值"。
            gpr.add_data(st, vs, theta_cur, fm)
            if t % 15 == 0: gpr.fit()
            mu_gpr, sr = gpr.predict(st, vs, theta_cur)
            if sr is None: sr = 0.
            GPR_SIGMA_THRESH = 0.3
            if mu_gpr is not None and sr < GPR_SIGMA_THRESH:
                mu_val = mu_gpr
            else:
                mu_val = trls[0] * st
            sigma_s = 0.2 * sr + 0.8 * sigma_s

            b_rls, m_rls = trls[1], trls[2]

            # 动态速度上限
            if arc_steps < ARC_WARMUP_STEPS:
                v_max_cur = V_WARMUP
            else:
                v_max_cur = min(V_MIN + (V_MAX - V_MIN) * ((arc_steps - ARC_WARMUP_STEPS) / V_RAMP), V_MAX)
            arc_steps += 1

            hks.append(trls[0]); hgmu.append(float(mu_val)); hgsig.append(sigma_s)
            h_dist_taut.append(dist_taut); h_vmax.append(v_max_cur)

            u = run_mpc_3d(p_cur, v_cur, trls[0], b_rls, m_rls,
                           anchor_est, dist_taut, f_taut,
                           theta_cur, theta_target,
                           STRETCH_MAX, f_max_safe,
                           gpr_model=gpr, gpr_sigma_thresh=0.3,
                           sigma_gpr=sigma_s, v_max_cur=v_max_cur,
                           R_ref=R_ref, w_time=w_time)

        hphase.append(phase)
        v_prev = v_cur.copy(); v_cur = u.copy()
        pn     = p_cur + dt * u
        vs2    = float(np.linalg.norm(u))
        as2    = float(np.linalg.norm((u - v_prev) / dt))
        f_cur  = true_force_3d(pn, vs2, as2)
        p_cur  = pn; hp.append(p_cur.copy())

    # ══════════════════════════════════════════════════════
    # 拟合层：anchor-based 直接测距
    # ══════════════════════════════════════════════════════
    hp = np.array(hp); hf_a = np.array(hf)
    R_fit = xc = yc = None

    if arc_step is not None:
        fc   = hf_a[arc_step:]
        rms  = float(np.sqrt(np.mean((fc - f_taut) ** 2)))
        # viol：归一化基准不能假设材料是线性弹簧——非线性材料(硬化/
        # 迟滞/各向异性)没有单一的"真实刚度"常数，等效刚度随 stretch/
        # vel/方向变化。改用 estimate_ks_max 采样估计这个材料模型在
        # [0, STRETCH_MAX] 范围内实际可能产生的保守最大等效刚度，对
        # 任何实现统一 compute 接口的材料模型都适用(线性模型也只是
        # 这套接口下的一个特例，采样结果会精确收敛到其 ks 本身)，不
        # 需要为每种模型单独推导解析公式。f_static_max 代表"材料被
        # 拉伸到 90% 允许范围、按最不利方向/速度状态采样得到的参照
        # 力"。实际力若超过这个值，说明出现了显著的动态力分量或
        # stretch 意外超标——这才是真正反映"力控制得好不好"的信号。
        # 注：force_model 在这里是仿真生成数据用的真值对象，只用于
        # 事后诊断指标计算，不进入任何控制器决策路径，不算真值泄露。
        ks_max_equiv = estimate_ks_max(force_model, STRETCH_MAX)
        f_static_max = 0.9 * STRETCH_MAX * ks_max_equiv
        viol = float(np.mean(fc > f_static_max)) * 100

        # 距离约束违反统计
        if dist_taut is not None:
            hp_arc = hp[arc_step:]
            dists_arc = np.linalg.norm(hp_arc - anchor_est, axis=1)
            dist_upper_val = dist_taut + STRETCH_MAX
            # 加入浮点容差(1mm 量级)：fallback 限速后机器人可能精确停在
            # dist_lower 这条线上，此时浮点运算的机器精度误差(~1e-16，
            # 远低于任何物理意义)会被 < / > 判成"违反"，造成假阳性。
            # 1e-3m(1mm)远小于任何真实的物理违反幅度，不会掩盖真实问题。
            DIST_TOL = 1e-3
            dist_viol_upper = float(np.mean(dists_arc > dist_upper_val + DIST_TOL)) * 100
            dist_viol_lower = float(np.mean(dists_arc < dist_taut - DIST_TOL)) * 100
        else:
            dist_viol_upper = dist_viol_lower = 0.

        if verbose:
            print(f"\n[指标] 弧线阶段 RMS力误差={rms:.3f}N  力超限(>{f_static_max:.2f}N)={viol:.1f}%")
        if verbose:
            print(f"[指标] 距离上界违反={dist_viol_upper:.1f}%  距离下界违反={dist_viol_lower:.1f}%")
        if verbose:
            print(f"[指标] 总步={len(hf_a)}  弧线步={len(fc)}")
        if verbose:
            print(f"[指标] dist_taut={dist_taut:.3f}m  True L0={L0_true:.3f}m  "
                  f"误差={abs(dist_taut-L0_true):.3f}m")

        pts2d  = hp[arc_step + ARC_WARMUP_STEPS:, :2]
        dists  = np.linalg.norm(pts2d - anchor_est[:2], axis=1)
        R_fit  = float(np.mean(dists))
        R_std  = float(np.std(dists))
        xc, yc = anchor_est[0], anchor_est[1]
        if verbose:
            print(f"[半径估计] R_mean={R_fit:.3f}m  R_std={R_std:.4f}m")

    # ══════════════════════════════════════════════════════
    # 返回结果：画图需要的轨迹/历史数据 + 实验脚本需要的标量指标
    # ══════════════════════════════════════════════════════
    if arc_step is None:
        return {
            'status': 'FAILED', 'arc_step': None, 'taut_step': taut_step,
            'hp': hp, 'hf': hf_a, 'hsig': hsig, 'hks': hks,
            'hgmu': hgmu, 'hgsig': hgsig, 'hphase': hphase,
            'h_dist_taut': h_dist_taut, 'h_vmax': h_vmax,
            'dist_taut': dist_taut, 'f_taut': f_taut,
            'anchor_est': anchor_est, 'theta_start': theta_start,
            'theta_target': theta_target, 'noise_std_est': noise_std_est,
            'R_fit': None, 'R_std': None, 'xc': None, 'yc': None,
            'ks_settle': None, 'ks_final': None,
            'rms': None, 'viol': None,
            'dist_viol_upper': None, 'dist_viol_lower': None,
            'total_steps': len(hf_a), 'arc_steps_count': None,
            'gpr_model': gpr, 'theta_frozen': None,
        }

    ks_final = float(theta_frozen[0])
    return {
        'status': 'OK', 'arc_step': arc_step, 'taut_step': taut_step,
        'hp': hp, 'hf': hf_a, 'hsig': hsig, 'hks': hks,
        'hgmu': hgmu, 'hgsig': hgsig, 'hphase': hphase,
        'h_dist_taut': h_dist_taut, 'h_vmax': h_vmax,
        'dist_taut': dist_taut, 'f_taut': f_taut,
        'anchor_est': anchor_est, 'theta_start': theta_start,
        'theta_target': theta_target, 'noise_std_est': noise_std_est,
        'R_fit': R_fit, 'R_std': R_std, 'xc': xc, 'yc': yc,
        'ks_settle': ks_final, 'ks_final': ks_final,
        'rms': rms, 'viol': viol,
        'dist_viol_upper': dist_viol_upper, 'dist_viol_lower': dist_viol_lower,
        'total_steps': len(hf_a), 'arc_steps_count': len(fc),
        'gpr_model': gpr, 'theta_frozen': theta_frozen.copy(),
    }



if __name__ == "__main__":
    # 直接运行本文件（python integrated_sim_3d.py）时才会触发这部分：
    # 跑一次默认参数的仿真、打印日志、生成图片。
    # 被 import（例如 run_experiment_matrix.py 的
    # `from integrated_sim_3d import run_sim`）时不会执行，
    # 因此 import 这个模块是无副作用的。
    # ══════════════════════════════════════════════════════
    # 材料模型选择
    # ══════════════════════════════════════════════════════
    # 编号 -> 模型实例的字典，而不是 if-else 链：加新材料只需要往字典
    # 里加一行，不用碰选择逻辑本身；想看"2号是什么"也只需要看这张表，
    # 不用去翻分支判断。改 MATERIAL_CHOICE 这一个数字就能切换材料。
    MATERIAL_PRESETS = {
        1: LinearSpring(ks=30, b=0.5, m=0.05),
        2: HardeningSpring(ks_base=10, alpha=2.0, b=0.5, m=0.05),
        3: HysteresisSpring(ks=20, hyst_gain=0.3, b=0.5, m=0.05),
        4: PiecewiseAnisoSpring(ks_soft=8, ks_hard=40, STRETCH_MAX=0.25,
                                 aniso_amp=0.2, b=0.5, m=0.05),
        5: LinearSpring(ks=10, b=0.5, m=0.05),
    }
    MATERIAL_CHOICE = 5   # 改这个数字切换材料，对照 MATERIAL_PRESETS 表
    # 注：4号(PiecewiseAnisoSpring)是已知未完全收敛的材料模型——分段
    # 转折点附近刚度突变较大，RLS 单一局部线性假设难以跨越，settle
    # 阶段常常超时退出而非真正收敛。这是当前架构的一个真实边界，不是
    # bug，跑这个模型时建议同时看 verbose 日志里的 P比/err 是否真的
    # 达标，不要只看 status='OK' 就认为收敛正常。

    # ══════════════════════════════════════════════════════
    # 单次运行入口（默认参数）
    # ══════════════════════════════════════════════════════
    # run_experiment_matrix.py 会 import run_sim 并用不同参数多次调用，
    # 不会执行下面这部分；这部分只在直接运行本文件时触发一次。
    _force_model    = MATERIAL_PRESETS[MATERIAL_CHOICE]
    _L0_true_run    = 0.5
    _STRETCH_MAX    = 0.25
    _f_max_safe     = 3.0

    result = run_sim(force_model=_force_model, arc_deg=90,
                      STRETCH_MAX=_STRETCH_MAX, f_max_safe=_f_max_safe,
                      L0_true=_L0_true_run, seed=42, verbose=True)

    # 把返回字典展开成画图代码使用的局部变量名
    anchor        = np.array([0.0, 0.0, 0.0])
    force_model   = _force_model
    L0_true       = _L0_true_run
    STRETCH_MAX   = _STRETCH_MAX
    f_max_safe    = _f_max_safe
    hf_a          = result['hf']
    hphase        = result['hphase']
    hp            = result['hp']
    taut_step     = result['taut_step']
    arc_step      = result['arc_step']
    R_fit         = result['R_fit']
    R_std         = result['R_std']
    xc            = result['xc']
    yc            = result['yc']
    theta_start   = result['theta_start']
    theta_target  = result['theta_target']
    dist_taut     = result['dist_taut']
    anchor_est    = result['anchor_est']
    f_taut        = result['f_taut']
    hgsig         = result['hgsig']
    h_vmax        = result['h_vmax']
    hks           = result['hks']
    noise_std_est = result['noise_std_est']
    hgmu          = result['hgmu']

    if result['status'] == 'FAILED':
        print("\n[警告] 仿真未能在 T_sim 步内完成绷直，跳过画图。")
    else:
        # ══════════════════════════════════════════════════════
        # 绘图
        # ══════════════════════════════════════════════════════
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Integrated 3D Simulation  —  大纲四层架构 v15\n"
            f"距离约束: dist_taut ≤ dist ≤ dist_taut+{STRETCH_MAX}m  "
            f"f_max_safe={f_max_safe}N(安全网)  无真值泄露",
            fontsize=11, y=0.99)

        n = len(hf_a)
        phase_arr = np.array(hphase + [hphase[-1] if hphase else 0])

        # 1. 轨迹
        ax1 = fig.add_subplot(2, 3, 1)
        p0 = np.where(phase_arr == 0)[0]
        p1 = np.where(phase_arr == 1)[0]
        p2 = np.where(phase_arr == 2)[0]
        if len(p0) > 0: ax1.plot(hp[p0, 0], hp[p0, 1], 'gray', lw=1.2, label='Phase0: explore', alpha=0.7)
        if len(p1) > 0: ax1.plot(hp[p1, 0], hp[p1, 1], 'orange', lw=1.5, label='Phase1: settle', alpha=0.8)
        if len(p2) > 0: ax1.plot(hp[p2, 0], hp[p2, 1], 'b-', lw=2, label='Phase2: arc (MPC)')
        ax1.plot(*anchor[:2], 'k+', ms=12, mew=2, label='Anchor')
        if taut_step is not None:
            ax1.plot(*hp[taut_step, :2], 'go', ms=8, zorder=5, label=f'BOCD t={taut_step}')
        if arc_step is not None:
            ax1.plot(*hp[arc_step, :2], 'b^', ms=8, zorder=5, label=f'Arc start t={arc_step}')
        if R_fit is not None:
            tha = np.linspace(theta_start, theta_target, 300)
            ax1.plot(xc + R_fit * np.cos(tha), yc + R_fit * np.sin(tha), 'm:', lw=1.5,
                     label=f'R_mean={R_fit:.2f}m (±{R_std:.3f})')
            # 距离约束带可视化
            if dist_taut is not None:
                for r_, c_, lab_ in [(dist_taut, 'green', f'dist_taut={dist_taut:.3f}m'),
                                     (dist_taut + STRETCH_MAX, 'red', f'dist_upper={dist_taut+STRETCH_MAX:.3f}m')]:
                    circle = plt.Circle(anchor_est[:2], r_, fill=False, color=c_, ls='--', lw=1.2, label=lab_)
                    ax1.add_patch(circle)
        ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_aspect('equal')
        ax1.legend(fontsize=6, loc='upper left'); ax1.set_title('Trajectory (top view)'); ax1.grid(True, alpha=0.3)

        # 2. 力 + 距离约束带
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(np.arange(n), hf_a, 'b-', lw=1.2, label='Force magnitude', alpha=0.85)
        ax2.axhline(f_max_safe, color='red', ls='-', lw=1.5, label=f'f_max_safe={f_max_safe}N')
        if taut_step is not None:
            ax2.axvline(taut_step, color='green', ls='--', alpha=0.7, label=f'BOCD t={taut_step}')
            ax2.axhline(f_taut, color='purple', ls='--', lw=1.2, label=f'f_taut={f_taut:.2f}N')
        if arc_step is not None:
            ax2.axvline(arc_step, color='blue', ls=':', alpha=0.7, label=f'Arc start t={arc_step}')
        ax2.set_ylabel('Force (N)'); ax2.set_xlabel('Time step')
        ax2.legend(fontsize=7); ax2.set_title('Force (距离约束→间接控制力)'); ax2.grid(True, alpha=0.3)

        # 3. 距离演化 + 约束带（核心安全图）
        ax3 = fig.add_subplot(2, 3, 3)
        dist_arr = np.linalg.norm(hp[:n] - anchor_est, axis=1)
        ax3.plot(np.arange(n), dist_arr, 'b-', lw=1.5, label='dist(t) from anchor')
        if dist_taut is not None:
            ax3.axhline(dist_taut, color='green', ls='--', lw=1.5, label=f'dist_taut={dist_taut:.3f}m')
            ax3.axhline(dist_taut + STRETCH_MAX, color='red', ls='--', lw=1.5,
                        label=f'dist_upper={dist_taut+STRETCH_MAX:.3f}m')
            ax3.fill_between(np.arange(n), dist_taut, dist_taut + STRETCH_MAX,
                             alpha=0.08, color='green', label='安全距离带')
        ax3.axhline(L0_true, color='gray', ls=':', lw=1, label=f'True L0={L0_true}m (不可见)')
        if taut_step is not None:
            ax3.axvline(taut_step, color='green', ls='--', alpha=0.6)
        if arc_step is not None:
            ax3.axvline(arc_step, color='blue', ls=':', alpha=0.6)
        ax3.set_ylabel('Distance (m)'); ax3.set_xlabel('Time step')
        ax3.legend(fontsize=7); ax3.set_title('Distance + 约束带 (核心安全图)'); ax3.grid(True, alpha=0.3)

        # 4. GPR σ + 动态速度上限
        ax4 = fig.add_subplot(2, 3, 4)
        if arc_step is not None:
            arc_gsig = np.array(hgsig[arc_step:])
            tc3 = np.arange(arc_step, arc_step + len(arc_gsig))
            if len(arc_gsig) > 2:
                ax4.plot(tc3, arc_gsig, color='purple', lw=1.5, label='GPR σ')
                ax4.fill_between(tc3, 0, arc_gsig, alpha=0.2, color='purple')
                ax4.axhline(0.3, color='red', ls='--', lw=1, label='σ threshold=0.3')
        ax4_r = ax4.twinx()
        if len(h_vmax) > 0:
            vm = np.array(h_vmax)
            ax4_r.plot(np.arange(len(vm)), vm, 'orange', lw=1.5, label='v_max_cur')
            ax4_r.set_ylabel('v_max (m/s)', color='orange')
        ax4.set_ylabel('σ (N)'); ax4.set_xlabel('Time step')
        ax4.legend(fontsize=7, loc='upper left')
        ax4_r.legend(fontsize=7, loc='upper right')
        ax4.set_title('GPR σ + dynamic v_max'); ax4.grid(True, alpha=0.3)

        # 5. RLS 刚度收敛（force_model 可能是线性也可能是非线性材料，
        # 没有统一保证存在的单一"真值"标量；用 estimate_ks_max 采样
        # 估计的等效最大刚度做参考线——线性材料下精确等于 ks 本身，
        # 非线性材料下是个量级参考，不是真值，标签里写清楚避免误读）
        ax5 = fig.add_subplot(2, 3, 5)
        ks_all = np.array(hks)
        ks_max_equiv = estimate_ks_max(force_model, STRETCH_MAX)
        ax5.plot(ks_all, 'b-', lw=1.5, label='RLS ks estimate')
        ax5.axhline(ks_max_equiv, color='r', ls='--',
                    label=f'采样估计ks_max≈{ks_max_equiv:.1f} (量级参考，非真值)')
        ax5.axhline(PRED_ERR_MULT * noise_std_est, color='orange', ls=':', lw=1.2,
                    label=f'pred_err_thresh={PRED_ERR_MULT*noise_std_est:.2f}N (标定)')
        if taut_step is not None:
            ax5.axvline(taut_step, color='green', ls='--', alpha=0.6, label='BOCD')
        if arc_step is not None:
            ax5.axvline(arc_step, color='blue', ls=':', alpha=0.6, label='Arc start')
        ax5.set_ylabel('ks (N/m)'); ax5.set_xlabel('Time step')
        ax5.legend(fontsize=7); ax5.set_title('Stiffness estimation (RLS)'); ax5.grid(True, alpha=0.3)

        # 6. GPR 预测 vs 真实力
        ax6 = fig.add_subplot(2, 3, 6)
        if arc_step is not None:
            arc_gmu  = np.array(hgmu[arc_step:])
            arc_gsig2 = np.array(hgsig[arc_step:])
            tc6 = np.arange(arc_step, arc_step + len(arc_gmu))
            if len(arc_gmu) > 2:
                ftc = hf_a[arc_step:arc_step + len(arc_gmu)]
                ax6.plot(tc6, ftc, 'k-', alpha=0.6, lw=1, label='True force')
                ax6.plot(tc6, arc_gmu, 'b-', lw=1.5, label='RLS est force (μ)')
                ax6.fill_between(tc6, arc_gmu - 2 * arc_gsig2, arc_gmu + 2 * arc_gsig2,
                                 alpha=0.25, color='blue', label='±2σ_GPR')
                ax6.axhline(f_taut, color='purple', ls='--', label=f'f_taut={f_taut:.2f}N')
                ax6.axhline(f_max_safe, color='red', ls='-', lw=1, label=f'f_max={f_max_safe}N')
        ax6.set_ylabel('Force (N)'); ax6.set_xlabel('Time step')
        ax6.legend(fontsize=7); ax6.set_title('Force estimate vs true'); ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'integrated_sim_3d_v15_{MATERIAL_CHOICE}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n[Plot] saved → {out}")