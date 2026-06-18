"""
Integrated 3D Simulation — 大纲四层架构 v11

修改（相对 v9）：
1. RLS 直接驱动 MPC：pf 和 pf_fast 合并为同一个弹簧模型，消除约束不一致
2. GPR 退回 Tube-MPC 角色：只提供 σ 收紧 f_max_eff，不参与力点预测
3. settle 阶段改为径向拉伸扫描：从 L0_est 开始推到 stretch≈0.1m 再收回
   临床意义：治疗师接触后做完整阻力评估，给 RLS 宽范围激励
4. f_lower 提高到 f_taut：绷直作为硬约束
5. 去掉速度惩罚项 W_INPUT
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import t as student_t
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 真实系统（控制器不可见）
# ══════════════════════════════════════════════════════
np.random.seed(42)
dt         = 0.05
ks_true    = 50.0
b_true     = 0.5
m_true     = 0.05
L0_true    = 0.5
anchor     = np.array([0.0, 0.0, 0.0])
f_max_safe = 3.0
noise_std  = 0.1

def true_force_mag(stretch, vel, acc):
    if stretch <= 0:
        return 0.0
    return ks_true*stretch + b_true*vel + m_true*acc

def true_force_3d(p, vel=0.0, acc=0.0):
    d = p - anchor; dist = np.linalg.norm(d)
    noise = np.random.normal(0, noise_std, 3)
    if dist < 1e-6: return noise
    stretch = max(0.0, dist - L0_true)
    f = max(0.0, true_force_mag(stretch, vel, acc))
    if f < 1e-6: return noise
    return f * d/dist + noise

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

# ── Settle 阶段参数 ──────────────────────────────────────
# 临床含义：治疗师接触后做完整阻力评估，从自然长度推到最大安全 stretch 再收回
# 给 RLS 提供宽范围激励（stretch 0→0.10m），比正弦往返收敛快
SETTLE_STEPS      = 160   # 给 RLS 更充分的时间收敛到真值附近
SETTLE_AMP        = 0.05  # 安全拉伸幅度：f_max_safe/ks ≈ 3/50 = 0.06m，留裕量取 0.05
SETTLE_SPEED      = 0.06  # settle 运动速度 (m/s)
KS_MIN_FOR_GPR    = 40.0  # RLS 收敛门槛（真值 50 的 80%）
GPR_MIN_DATA      = 20

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
        self.theta=np.array([10.0, 0.3, 0.05])
        self.P=np.diag([100., 3., 1.])

    def set_lam(self, lam):
        self.lam_cur = lam

    def update(self, phi, y, stretch_delta=None):
        # stretch 变化量极小时跳过更新，避免无激励状态下 ks 漂移
        if stretch_delta is not None and abs(stretch_delta) < 1e-3:
            return self.theta.copy()
        e=y-phi@self.theta; d=self.lam_cur+phi@self.P@phi
        K=self.P@phi/d; self.theta+=K*e
        self.P=(1./self.lam_cur)*(np.eye(3)-np.outer(K,phi))@self.P
        self.theta=np.clip(self.theta,[0.05,0.,0.],[200.,5.,2.])
        return self.theta.copy()

    def force_bounds(self, f_taut):
        f_lower = 0.95 * f_taut                 # 接近绷直，留小余量防止约束不可行
        f_upper = min(2.5 * f_taut, f_max_safe)
        return f_lower, f_upper

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
        self.X=[]; self.y=[]; self.fitted=False
        self.xm=np.zeros(3); self.xs=np.ones(3); self.max_data=max_data

    def add_data(self, s, v, theta, f):
        # 输入：[stretch, vel, theta]，theta 让 GPR 感知弧线位置
        self.X.append([s, v, theta]); self.y.append(f)
        if len(self.X) > self.max_data: self.X.pop(0); self.y.pop(0)

    def fit(self):
        if len(self.X) < 15: return   # 数据量 ≥ 15 才 fit，避免早期过拟合
        X = np.array(self.X)
        self.xm = X.mean(0); self.xs = X.std(0) + 1e-6
        self.gpr.fit((X - self.xm) / self.xs, np.array(self.y))
        self.fitted = True

    def predict(self, s, v, theta):
        if not self.fitted: return None, None
        xn = (np.array([[s, v, theta]]) - self.xm) / self.xs
        mu, sig = self.gpr.predict(xn, return_std=True)
        return float(mu[0]), float(sig[0])

    def local_ks(self, stretch, vel, theta, ks_fallback, delta=0.001, sigma_threshold=0.3):
        if not self.fitted: return ks_fallback
        _, sigma_cur = self.predict(stretch, vel, theta)
        if sigma_cur is None or sigma_cur > sigma_threshold:
            return ks_fallback
        s_plus  = max(0., stretch + delta)
        s_minus = max(0., stretch - delta)
        mu_plus,  _ = self.predict(s_plus,  vel, theta)
        mu_minus, _ = self.predict(s_minus, vel, theta)
        if mu_plus is None or mu_minus is None:
            return ks_fallback
        ks_eff = (mu_plus - mu_minus) / (2 * delta)
        return max(0.05, ks_eff)

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
               anchor_est, L0_est,
               theta_cur, theta_target,
               f_taut, f_lower, f_upper,
               sigma_gpr=0., v_max_cur=0.1, R_ref=None):

    f_max_eff = f_upper - 2.0*np.clip(sigma_gpr, 0., 0.5)
    f_max_eff = max(f_max_eff, f_taut*1.1)

    if R_ref is None:
        R_ref = np.linalg.norm(p_cur - anchor_est)

    def pf(p_, v_, a_):
        """统一力预测：RLS 弹簧模型
        GPR 退回 Tube-MPC 角色，只提供 σ 收紧 f_max_eff，不参与力点预测
        代价函数和约束函数使用同一模型，消除不一致"""
        stretch_ = max(0., np.linalg.norm(p_-anchor_est) - L0_est)
        return max(0., ks_eff*stretch_ + b_rls*np.linalg.norm(v_) + m_rls*np.linalg.norm(a_))

    pf_fast = pf  # 两者完全一致

    def cost(u_flat):
        u_seq=u_flat.reshape(MPC_N,3); p=p_cur.copy(); v=v_cur.copy(); c=0.
        f_mid = 0.7*f_lower + 0.3*f_max_eff  # 偏向绷直下界，倾向在刚绷直状态工作
        W_MID = 3.0
        for k in range(MPC_N):
            vk=u_seq[k]; ak=(vk-v)/dt; pn=p+dt*vk
            f_pred = pf(pn, vk, ak)
            c += W_FORCE * max(0, f_lower - f_pred) ** 2
            c += W_FORCE * max(0, f_pred - f_max_eff) ** 2
            c += W_MID * (f_pred - f_mid) ** 2
            c += W_TIME
            tn=np.arctan2(pn[1]-anchor_est[1], pn[0]-anchor_est[0])
            pg=tn-theta_cur
            if pg<-np.pi: pg+=2*np.pi
            c -= W_ANGLE*np.clip(pg, 0., 0.15)
            p=pn; v=vk
        f_terminal = pf(p, v, np.zeros(3))
        c += W_FORCE * max(0, f_lower - f_terminal) ** 2
        c += W_FORCE * max(0, f_terminal - f_max_eff) ** 2
        c += W_MID * (f_terminal - f_mid) ** 2
        return c

    cons=[]
    for k in range(MPC_N):
        def make_upper(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return f_max_eff - pf_fast(p,v,np.zeros(3))
            return fn
        def make_lower(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return pf_fast(p,v,np.zeros(3)) - f_lower
            return fn
        # 径向约束：预测位置的 dist 不偏离 R_ref 超过 R_TOL
        def make_radial_upper(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return R_ref + R_TOL - np.linalg.norm(p - anchor_est)
            return fn
        def make_radial_lower(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return np.linalg.norm(p - anchor_est) - (R_ref - R_TOL)
            return fn
        cons.append({'type':'ineq','fun':make_upper(k)})
        cons.append({'type':'ineq','fun':make_lower(k)})
        cons.append({'type':'ineq','fun':make_radial_upper(k)})
        cons.append({'type':'ineq','fun':make_radial_lower(k)})

    tang=np.array([-np.sin(theta_cur), np.cos(theta_cur), 0.])
    u0=np.tile(tang*0.05, MPC_N)
    res=minimize(cost, u0, method='SLSQP',
                 bounds=[(-v_max_cur, v_max_cur)]*(MPC_N*3),
                 constraints=cons,
                 options={'maxiter':40,'ftol':2e-3})
    u_out=res.x.reshape(MPC_N,3)[0]
    return u_out if (res.success or res.fun<1e8) else tang*0.02

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
# 主仿真
# ══════════════════════════════════════════════════════
T_sim        = 800
theta_start  = np.radians(-20)
theta_target = theta_start + np.pi/2

R_init   = L0_true*0.35
p_cur    = anchor + R_init*np.array([np.cos(theta_start),np.sin(theta_start),0.])
v_cur    = np.zeros(3); v_prev=np.zeros(3)
L0_est   = R_init*0.8; anchor_est=anchor.copy()

bocd=BOCD(); rls=RLS(); gpr=GPRModel()

# 三阶段状态机
phase=0          # 0:探索  1:settle  2:弧线
taut_step=None; arc_step=None
arc_steps=0; settle_steps_done=0; settle_base_dist=0.
f_taut=0.2; f_lower=0.1; f_upper=1.0
sigma_s=0.; R_ref=None; st_prev=0.

hp=[p_cur.copy()]; hf=[]; hsig=[]; hks=[]
hgmu=[]; hgsig=[]; hL0=[L0_est]; hphase=[]
h_flower=[]; h_fupper=[]; h_vmax=[]
f_cur=true_force_3d(p_cur)

print("="*62)
print("Integrated 3D Simulation  —  大纲四层架构 v6")
print(f"ks_true={ks_true}  L0_true={L0_true}m  f_max_safe={f_max_safe}N")
print(f"起始θ={np.degrees(theta_start):.0f}°  目标θ={np.degrees(theta_target):.0f}°")
print(f"速度: {V_MIN}→{V_MAX} m/s (线性爬升 {V_RAMP} 步)")
print(f"Settle幅度: ±{SETTLE_AMP}m  最大步数: {SETTLE_STEPS}")
print("="*62)

for t in range(T_sim):
    fm=np.linalg.norm(f_cur); hf.append(fm)
    sig=bocd.update(fm); hsig.append(sig)

    theta_cur=np.arctan2(p_cur[1]-anchor_est[1], p_cur[0]-anchor_est[0])

    # ── 阶段转换 ─────────────────────────────────────────
    # 0→1：BOCD 检测到绷直，进入 settle
    if phase==0 and sig>0.35:
        phase=1; taut_step=t; settle_steps_done=0
        f_taut = max(fm, 0.1)
        f_lower, f_upper = rls.force_bounds(f_taut)
        # 修复：L0_est 用物理公式，除以 RLS 当前 ks 估计
        ks_guess = max(rls.theta[0], 1.0)
        L0_est = max(0.05, np.linalg.norm(p_cur-anchor_est) - fm/ks_guess)
        settle_base_dist = np.linalg.norm(p_cur-anchor_est)  # settle 的基准距离
        print(f"  [BOCD]  t={t:3d}: 检测到绷直 → 进入 settle 阶段")
        print(f"          sig={sig:.3f}  f_taut={f_taut:.3f}N  "
              f"L0_est={L0_est:.3f}m  ks_guess={ks_guess:.1f}")

    # 1→2：settle 完成（步数到了或 RLS 已收敛），进入弧线
    if phase==1:
        ks_converged = rls.theta[0] > KS_MIN_FOR_GPR
        settle_done  = settle_steps_done >= SETTLE_STEPS
        if ks_converged or settle_done:
            phase=2; arc_step=t
            # 只有 RLS 真正收敛（过门槛）时才用更新后的 ks 重估 L0
            if ks_converged:
                ks_now = rls.theta[0]
                dist_now = np.linalg.norm(p_cur - anchor_est)
                L0_est = max(0.05, dist_now - fm / ks_now)
                print(f"  [Settle→Arc] t={t:3d}: ks_est={ks_now:.1f} (收敛)  "
                      f"L0_est更新→{L0_est:.3f}m")
            else:
                print(f"  [Settle→Arc] t={t:3d}: ks_est={rls.theta[0]:.1f} (超时，保留L0_est={L0_est:.3f}m)")
            rls.set_lam(rls.lam)
            R_ref = np.linalg.norm(p_cur - anchor_est)
            st_prev = max(0., np.linalg.norm(p_cur - anchor_est) - L0_est)

    if phase==2 and theta_cur>=theta_target-0.03:
        print(f"  [Done]  t={t:3d}: θ={np.degrees(theta_cur):.1f}° 到达目标!")
        break

    # ── 安全回缩（仅弧线阶段） ───────────────────────────
    if fm>=f_max_safe and phase==2:
        u=retract_velocity(p_cur, anchor_est, speed=0.1)
        hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(sigma_s)
        h_flower.append(f_lower); h_fupper.append(f_upper); h_vmax.append(0.)
        hL0.append(L0_est); hphase.append(phase)
        v_prev=v_cur.copy()   # 回缩前正确保存 v_prev，消除下一步 ac 尖峰
        v_cur=u.copy()
        pn=p_cur+dt*u
        f_cur=true_force_3d(pn, np.linalg.norm(u), np.linalg.norm((u-v_prev)/dt))
        p_cur=pn; hp.append(p_cur.copy())
        continue

    # ── 阶段 0：径向外探索 ───────────────────────────────
    if phase==0:
        dist=np.linalg.norm(p_cur-anchor_est); dn=(p_cur-anchor_est)/(dist+1e-6)
        sp=0.15*max(0.3, 1.4-dist/(L0_true*1.3)); u=dn*sp
        hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(0.)
        h_flower.append(f_lower); h_fupper.append(f_upper); h_vmax.append(0.)

    # ── 阶段 1：Settle（径向拉伸扫描，激励 RLS） ────────
    elif phase==1:
        dist = np.linalg.norm(p_cur-anchor_est)
        dn   = (p_cur-anchor_est)/(dist+1e-6)

        # 力安全检查：settle 阶段力接近上限时立即收回
        # 临床意义：评估阻力时不能对患者施加危险力度
        if fm > f_max_safe * 0.8:
            target_dist = settle_base_dist  # 强制收回到基准距离
        else:
            half = SETTLE_STEPS / 2
            if settle_steps_done < half:
                target_dist = settle_base_dist + SETTLE_AMP * (settle_steps_done / half)
            else:
                target_dist = settle_base_dist + SETTLE_AMP * (2 - settle_steps_done / half)

        err = target_dist - dist
        u = dn * np.clip(err / dt, -SETTLE_SPEED, SETTLE_SPEED)
        settle_steps_done += 1

        # RLS 更新（全程激励窗口）
        st  = max(0., dist - L0_est)
        vs  = np.linalg.norm(v_cur)
        ac  = np.linalg.norm((v_cur-v_prev)/dt)
        phi = np.array([st, vs, ac])
        trls = rls.update(phi, fm)
        f_lower, f_upper = rls.force_bounds(f_taut)
        f_upper = min(f_upper, f_max_safe)

        # GPR 积累数据
        gpr.add_data(st, vs, theta_cur, fm)
        if len(gpr.X) >= 15 and t%5==0: gpr.fit()

        hks.append(trls[0]); hgmu.append(0.); hgsig.append(0.)
        h_flower.append(f_lower); h_fupper.append(f_upper); h_vmax.append(0.)

    # ── 阶段 2：弧线（MPC，动态速度上限） ───────────────
    else:
        d=p_cur-anchor_est; dist=np.linalg.norm(d)
        st=max(0., dist-L0_est); vs=np.linalg.norm(v_cur)
        ac=np.linalg.norm((v_cur-v_prev)/dt)

        # RLS 更新：stretch 无激励时跳过，防止无信号漂移
        phi=np.array([st,vs,ac])
        stretch_delta = st - st_prev
        trls=rls.update(phi, fm, stretch_delta=stretch_delta)
        st_prev = st
        f_lower, f_upper = rls.force_bounds(f_taut)
        f_upper = min(f_upper, f_max_safe)

        # L0_est 滚动更新（低通，防跳变）
        if trls[0] > KS_MIN_FOR_GPR and fm > f_taut * 0.5:
            L0_est_new = max(0.05, dist - fm / trls[0])
            L0_est = 0.9 * L0_est + 0.1 * L0_est_new

        # R_ref 缓慢追踪
        if arc_steps % R_REF_UPDATE == 0:
            R_ref = 0.8 * R_ref + 0.2 * dist

        # GPR 更新（只用于提供 σ 收紧约束）
        gpr.add_data(st, vs, theta_cur, fm)
        if t%15==0: gpr.fit()
        _, sr = gpr.predict(st, vs, theta_cur)
        if sr is None: sr=0.
        mu_val = trls[0]*st  # 显示用，用 RLS 线性估计
        sigma_s = 0.2*sr + 0.8*sigma_s

        b_rls, m_rls = trls[1], trls[2]

        # 动态速度上限
        if arc_steps < ARC_WARMUP_STEPS:
            v_max_cur = V_WARMUP
        else:
            v_max_cur = min(V_MIN + (V_MAX-V_MIN)*((arc_steps-ARC_WARMUP_STEPS)/V_RAMP), V_MAX)
        arc_steps += 1

        hks.append(trls[0]); hgmu.append(float(mu_val)); hgsig.append(sigma_s)
        h_flower.append(f_lower); h_fupper.append(f_upper); h_vmax.append(v_max_cur)

        u=run_mpc_3d(p_cur, v_cur, trls[0], b_rls, m_rls,
                     anchor_est, L0_est,
                     theta_cur, theta_target,
                     f_taut, f_lower, f_upper,
                     sigma_gpr=sigma_s, v_max_cur=v_max_cur,
                     R_ref=R_ref)

    hL0.append(L0_est); hphase.append(phase)
    v_prev=v_cur.copy(); v_cur=u.copy()
    pn=p_cur+dt*u
    vs2=float(np.linalg.norm(u)); as2=float(np.linalg.norm((u-v_prev)/dt))
    f_cur=true_force_3d(pn, vs2, as2)
    p_cur=pn; hp.append(p_cur.copy())

# ══════════════════════════════════════════════════════
# 拟合层：Taubin
# ══════════════════════════════════════════════════════
hp=np.array(hp); hf_a=np.array(hf)
R_fit=xc=yc=None

if arc_step is not None:
    fc=hf_a[arc_step:]
    rms=float(np.sqrt(np.mean((fc-f_taut)**2)))
    viol=float(np.mean(fc>f_max_safe))*100
    print(f"\n[指标] 弧线阶段 RMS力误差={rms:.3f}N  超限={viol:.1f}%")
    print(f"[指标] 总步={len(hf_a)}  弧线步={len(fc)}")
    # 新：已知圆心在锚点，直接算各点到锚点的距离
    pts2d = hp[arc_step + ARC_WARMUP_STEPS:, :2]
    dists = np.linalg.norm(pts2d - anchor_est[:2], axis=1)
    R_fit = float(np.mean(dists))
    R_std = float(np.std(dists))
    xc, yc = anchor_est[0], anchor_est[1]
    print(f"[半径估计] R_mean={R_fit:.3f}m  R_std={R_std:.4f}m  (锚点已知，直接测距)")

# ══════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════
fig=plt.figure(figsize=(16,10))
fig.suptitle("Integrated 3D Simulation  —  大纲四层架构 v11\n"
             "三阶段: 探索→settle(拉伸扫描)→弧线  RLS驱动MPC  GPR仅提供σ  ks=50",
             fontsize=11, y=0.99)

n=len(hf_a)
phase_arr=np.array(hphase+[hphase[-1] if hphase else 0])

# 1. 轨迹
ax1=fig.add_subplot(2,3,1)
p0=np.where(phase_arr==0)[0]
p1=np.where(phase_arr==1)[0]
p2=np.where(phase_arr==2)[0]
if len(p0)>0: ax1.plot(hp[p0,0],hp[p0,1],'gray',lw=1.2,label='Phase0: explore',alpha=0.7)
if len(p1)>0: ax1.plot(hp[p1,0],hp[p1,1],'orange',lw=1.5,label='Phase1: settle',alpha=0.8)
if len(p2)>0: ax1.plot(hp[p2,0],hp[p2,1],'b-',lw=2,label='Phase2: arc (MPC)')
ax1.plot(*anchor[:2],'k+',ms=12,mew=2,label='Anchor')
if taut_step is not None:
    ax1.plot(*hp[taut_step,:2],'go',ms=8,zorder=5,label=f'BOCD t={taut_step}')
if arc_step is not None:
    ax1.plot(*hp[arc_step,:2],'b^',ms=8,zorder=5,label=f'Arc start t={arc_step}')
if R_fit is not None:
    tha=np.linspace(theta_start,theta_target,300)
    ax1.plot(xc+R_fit*np.cos(tha), yc+R_fit*np.sin(tha), 'm:', lw=1.5,
         label=f'R_mean={R_fit:.2f}m (±{R_std:.3f})')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_aspect('equal')
ax1.legend(fontsize=7,loc='upper left'); ax1.set_title('Trajectory (top view)'); ax1.grid(True,alpha=0.3)

# 2. 力 + 动态边界
ax2=fig.add_subplot(2,3,2)
ax2.plot(np.arange(n),hf_a,'b-',lw=1.2,label='Force magnitude',alpha=0.85)
ax2.axhline(f_max_safe,color='red',ls='-',lw=1.5,label=f'f_max_safe={f_max_safe}N')
if taut_step is not None:
    ax2.axvline(taut_step,color='green',ls='--',alpha=0.7,label=f'BOCD t={taut_step}')
    ax2.axhline(f_taut,color='purple',ls='--',lw=1.2,label=f'f_taut={f_taut:.2f}N')
if arc_step is not None:
    ax2.axvline(arc_step,color='blue',ls=':',alpha=0.7,label=f'Arc start t={arc_step}')
if len(h_flower)>0:
    fl=np.array(h_flower); fu=np.array(h_fupper)
    ax2.fill_between(np.arange(len(fl)),fl,fu,alpha=0.1,color='green',label='RLS force bounds')
ax2.set_ylabel('Force (N)'); ax2.set_xlabel('Time step')
ax2.legend(fontsize=7); ax2.set_title('Force magnitude + dynamic bounds'); ax2.grid(True,alpha=0.3)

# 3. GPR 预测 vs 真实
ax3=fig.add_subplot(2,3,3)
if arc_step is not None:
    arc_gmu=np.array(hgmu[arc_step:]); arc_gsig=np.array(hgsig[arc_step:])
    tc3=np.arange(arc_step,arc_step+len(arc_gmu))
    if len(arc_gmu)>2:
        ftc=hf_a[arc_step:arc_step+len(arc_gmu)]
        ax3.plot(tc3,ftc,'k-',alpha=0.6,lw=1,label='True force')
        ax3.plot(tc3,arc_gmu,'b-',lw=1.5,label='GPR mean μ')
        ax3.fill_between(tc3,arc_gmu-2*arc_gsig,arc_gmu+2*arc_gsig,
                         alpha=0.25,color='blue',label='±2σ')
        ax3.axhline(f_taut,color='purple',ls='--',label=f'f_taut={f_taut:.2f}N')
        ax3.axhline(f_max_safe,color='red',ls='-',lw=1,label=f'f_max={f_max_safe}N')
ax3.set_ylabel('Force (N)'); ax3.set_xlabel('Time step')
ax3.legend(fontsize=7); ax3.set_title('GPR prediction vs true force'); ax3.grid(True,alpha=0.3)

# 4. GPR 不确定度 + 动态速度上限
ax4=fig.add_subplot(2,3,4)
if arc_step is not None and len(arc_gsig)>2:
    ax4.plot(tc3,arc_gsig,color='purple',lw=1.5,label='GPR σ')
    ax4.fill_between(tc3,0,arc_gsig,alpha=0.2,color='purple')
    ax4.axhline(0.3,color='red',ls='--',lw=1,label='σ threshold=0.3')
ax4_r=ax4.twinx()
if len(h_vmax)>0:
    vm=np.array(h_vmax)
    ax4_r.plot(np.arange(len(vm)),vm,'orange',lw=1.5,label='v_max_cur')
    ax4_r.set_ylabel('v_max (m/s)',color='orange')
ax4.set_ylabel('σ (N)'); ax4.set_xlabel('Time step')
ax4.legend(fontsize=7,loc='upper left')
ax4_r.legend(fontsize=7,loc='upper right')
ax4.set_title('GPR σ + dynamic v_max'); ax4.grid(True,alpha=0.3)

# 5. RLS 刚度收敛
ax5=fig.add_subplot(2,3,5)
ks_all=np.array(hks)
ax5.plot(ks_all,'b-',lw=1.5,label='RLS ks estimate')
ax5.axhline(ks_true,color='r',ls='--',label=f'True ks={ks_true}')
ax5.axhline(KS_MIN_FOR_GPR,color='orange',ls=':',lw=1.2,label=f'GPR handoff={KS_MIN_FOR_GPR}')
if taut_step is not None:
    ax5.axvline(taut_step,color='green',ls='--',alpha=0.6,label='BOCD')
if arc_step is not None:
    ax5.axvline(arc_step,color='blue',ls=':',alpha=0.6,label='Arc start')
ax5.set_ylabel('ks (N/m)'); ax5.set_xlabel('Time step')
ax5.legend(fontsize=7); ax5.set_title('Stiffness estimation (RLS)'); ax5.grid(True,alpha=0.3)

# 6. 半径演化
ax6=fig.add_subplot(2,3,6)
L0_arr=np.array(hL0)
ax6.plot(L0_arr,'g-',lw=1.5,label='L0 estimate')
ax6.axhline(L0_true,color='r',ls='--',label=f'True L0={L0_true}m')
dist_arr=np.linalg.norm(hp[:-1]-anchor,axis=1)[:n]
ax6.plot(dist_arr,'b-',lw=1,alpha=0.6,label='dist(t) from anchor')
if taut_step is not None:
    ax6.axvline(taut_step,color='green',ls='--',alpha=0.6,label='BOCD')
if arc_step is not None:
    ax6.axvline(arc_step,color='blue',ls=':',alpha=0.6,label='Arc start')
ax6.set_ylabel('Distance (m)'); ax6.set_xlabel('Time step')
ax6.legend(fontsize=7); ax6.set_title('L0 estimate & radius evolution'); ax6.grid(True,alpha=0.3)

plt.tight_layout()
out='integrated_sim_3d.png'
plt.savefig(out,dpi=150,bbox_inches='tight')
print(f"\n[Plot] saved → {out}")