"""
Integrated 3D Simulation — 面向下肢康复的充分拉伸牵引架构 v15
核心升级：
1. 引入 Taubin 动态圆拟合，在线修正锚点与 L0，彻底解决形变(stretch)越界
2. 真实物理环境引入 Bouc-Wen 迟滞模型与温和的非线性刚度，贴合软材料/生物组织
3. GPR 输入特征增加速度方向 sign(v)，提升对迟滞非线性的感知能力
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
warnings.filterwarnings('ignore')
# ══════════════════════════════════════════════════════
# 1. 真实系统环境（引入迟滞与合理刚度）
# ══════════════════════════════════════════════════════
np.random.seed(42)
dt         = 0.05
L0_true    = 0.5
anchor     = np.array([0.0, 0.0, 0.0])
f_max_safe = 3.0
noise_std  = 0.05
STRETCH_MAX = 0.06
# 迟滞内部状态
z_hyst = 0.0 
def get_true_stiffness(theta_rad):
    """模拟下肢关节活动度极限：角度越大，刚度温和上升"""
    angle_deg = np.degrees(theta_rad)
    if angle_deg < 60:
        return 50.0
    else:
        # 60度后刚度缓慢上升，最高约 150 N/m
        return 50.0 + 2.0 * (angle_deg - 60)**1.5
def true_force_3d(p, vel):
    global z_hyst
    d = p - anchor; dist = np.linalg.norm(d)
    noise = np.random.normal(0, noise_std, 3)
    if dist < 1e-6: return noise
    theta = np.arctan2(d[1], d[0])
    ks_true = get_true_stiffness(theta)
    stretch = max(0.0, dist - L0_true)
    if stretch > 0:
        # 简化的 Bouc-Wen 迟滞模型：拉伸和回弹产生不同的阻力
        v = np.linalg.norm(vel)
        sign_v = np.sign(v) if v > 1e-4 else 0
        z_hyst = (0.8 * stretch - 2.0 * abs(z_hyst) * z_hyst) * sign_v + 0.1 * z_hyst
        z_hyst = np.clip(z_hyst, -0.05, 0.05)
        # 总力 = 线性刚度力 + 迟滞残余力
        f = ks_true * stretch + 30.0 * z_hyst
    else:
        f = 0.0
    if f < 1e-6: return noise
    return f * d/dist + noise
# ══════════════════════════════════════════════════════
# 2. 轨迹与参数定义
# ══════════════════════════════════════════════════════
theta_start  = np.radians(0)
theta_target = np.radians(90)
R_path       = L0_true + 0.04
N_SAMPLES = 300
HORIZON   = 8
V_MAX     = 0.3
V_MIN     = 0.0
W_TIME    = 1.0
W_FORCE   = 80.0    # 增大力惩罚权重，避免极限试探
W_TRACK   = 150.0
# ══════════════════════════════════════════════════════
# 3. 感知与建模层
# ══════════════════════════════════════════════════════
class LocalGPR:
    def __init__(self, max_data=50):
        k = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=0.1, length_scale_bounds=(0.01, 1.0)) + WhiteKernel(noise_level=0.1)
        self.gpr = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0, normalize_y=True)
        self.X = []; self.y = []; self.max_data = max_data
        self.fitted = False
    def add_data(self, s, v, sign_v, f):
        # 特征增加 sign_v 以捕捉迟滞方向
        self.X.append([s, v, sign_v]); self.y.append(f)
        if len(self.X) > self.max_data:
            self.X.pop(0); self.y.pop(0)
    def fit(self):
        if len(self.X) < 15: return
        self.gpr.fit(np.array(self.X), np.array(self.y))
        self.fitted = True
    def predict(self, s, v, sign_v):
        if not self.fitted: return 0.0, 0.5
        mu, sig = self.gpr.predict(np.array([[s, v, sign_v]]), return_std=True)
        return float(mu[0]), float(sig[0])
    def get_local_stiffness(self, s, v, sign_v):
        if not self.fitted: return 0.0
        delta = 0.001
        mu_plus, _ = self.predict(s + delta, v, sign_v)
        mu_minus, _ = self.predict(max(0, s - delta), v, sign_v)
        return (mu_plus - mu_minus) / (2 * delta)
class RLS:
    def __init__(self, lam=0.95):
        self.lam = lam
        self.theta = np.array([10.0, 0.5, 0.05])
        self.P = np.diag([100., 10., 1.])
    def update(self, phi, y):
        e = y - phi @ self.theta
        d = self.lam + phi @ self.P @ phi
        K = self.P @ phi / d
        self.theta += K * e
        self.P = (1./self.lam) * (np.eye(3) - np.outer(K, phi)) @ self.P
        self.theta = np.clip(self.theta, [0.1, 0., 0.], [500., 10., 2.])
        return self.theta.copy()
# ══════════════════════════════════════════════════════
# 4. Taubin 动态圆拟合（在线修正锚点与 L0）
# ══════════════════════════════════════════════════════
def taubin_fit(pts):
    if len(pts) < 15: return None, None, None
    x = pts[:,0]; y = pts[:,1]
    X = x - x.mean(); Y = y - y.mean(); Z = X**2 + Y**2; Zm = Z.mean()
    if Zm < 1e-10: return None, None, None
    Z0 = (Z - Zm) / (2 * np.sqrt(Zm)); A = np.column_stack([Z0, X, Y])
    _, _, V = np.linalg.svd(A); a = V[-1]
    a0 = a[0] / (2 * np.sqrt(Zm)); a1, a2 = a[1], a[2]
    if abs(a0) < 1e-10: return None, None, None
    xc = -a1 / (2 * a0); yc = -a2 / (2 * a0)
    val = a1**2 + a2**2 - 4 * a0 * (a0 * Zm - Z0.mean())
    return (xc, yc, np.sqrt(val) / abs(2 * a0)) if val > 0 else (None, None, None)
# ══════════════════════════════════════════════════════
# 5. 控制层：MPPI
# ══════════════════════════════════════════════════════
def run_mppi(p_cur, v_cur_vec, theta_cur, anchor_est, L0_est, ks, b, m, f_max_eff):
    tang = np.array([-np.sin(theta_cur), np.cos(theta_cur), 0.])
    v_cur_scalar = np.dot(v_cur_vec, tang)
    v_nominal = np.full(HORIZON, V_MAX * 0.8)
    noise = np.random.normal(0, 0.1, (N_SAMPLES, HORIZON))
    v_samples = np.clip(v_nominal + noise, V_MIN, V_MAX)
    costs = np.zeros(N_SAMPLES)
    for k in range(HORIZON):
        v_k = v_samples[:, k]
        p_next = p_cur + dt * v_k[:, None] * tang
        dist = np.linalg.norm(p_next - anchor_est, axis=1)
        s = np.maximum(0.0, dist - L0_est)
        a_k = (v_k - v_cur_scalar) / dt
        f_pred = ks * s + b * v_k + m * a_k
        cost_time = -W_TIME * v_k
        cost_force = W_FORCE * np.maximum(0, f_pred - f_max_eff)**2
        cost_track = W_TRACK * (dist - R_path)**2
        costs += cost_time + cost_force + cost_track
    costs -= np.min(costs)
    weights = np.exp(-costs / 10.0)
    weights /= np.sum(weights)
    v_opt = np.sum(weights * v_samples[:, 0])
    return float(np.clip(v_opt, V_MIN, V_MAX))
# ══════════════════════════════════════════════════════
# 6. 安全层：Bio-CBF
# ══════════════════════════════════════════════════════
def apply_bio_cbf(p_cur, u_cmd, anchor_est, L0_est):
    d = p_cur - anchor_est
    dist = np.linalg.norm(d)
    if dist < 1e-6: return u_cmd
    radial_dir = d / dist
    tang_dir = np.array([-radial_dir[1], radial_dir[0], 0.])
    v_radial = np.dot(u_cmd, radial_dir)
    v_tang = np.dot(u_cmd, tang_dir)
    stretch_cur = max(0.0, dist - L0_est)
    # CBF 硬约束：确保 stretch 绝不超过 STRETCH_MAX
    if stretch_cur > STRETCH_MAX * 0.85:
        v_radial_safe = min(0.0, v_radial)
        if stretch_cur > STRETCH_MAX * 0.95:
            v_radial_safe = -0.02 
        u_safe = v_radial_safe * radial_dir + v_tang * tang_dir
        return u_safe
    return u_cmd
# ══════════════════════════════════════════════════════
# 7. 主仿真循环
# ══════════════════════════════════════════════════════
T_sim = 400
p_cur = anchor + R_path * np.array([np.cos(theta_start), np.sin(theta_start), 0.])
v_cur = np.zeros(3); v_prev = np.zeros(3)
L0_est = L0_true * 0.9
anchor_est = anchor.copy()
gpr = LocalGPR(); rls = RLS()
hp = [p_cur.copy()]; hf = []; hv = []; htheta = []; hs = []; hkeff = []
print("="*60)
print("开始仿真: MPPI + Bio-CBF 架构 v15 (带动态拟合与迟滞模型)")
print("="*60)
for t in range(T_sim):
    theta_cur = np.arctan2(p_cur[1]-anchor_est[1], p_cur[0]-anchor_est[0])
    if theta_cur >= theta_target:
        print(f"  [完成] t={t}: 到达目标角度!")
        break
    f_cur_vec = true_force_3d(p_cur, v_cur)
    fm = np.linalg.norm(f_cur_vec)
    # --- 动态锚点拟合修正 ---
    if t > 20 and t % 10 == 0:
        pts = np.array(hp[-50:])
        xc, yc, R_fit = taubin_fit(pts)
        if xc is not None:
            anchor_est = np.array([xc, yc, 0.0])
            if fm > 0.1 and rls.theta[0] > 5.0:
                L0_est = max(0.1, R_fit - fm / rls.theta[0])
    dist = np.linalg.norm(p_cur - anchor_est)
    st = max(0.0, dist - L0_est)
    vs = np.linalg.norm(v_cur)
    sign_v = np.sign(vs) if vs > 1e-4 else 0
    ac = np.linalg.norm((v_cur - v_prev) / dt)
    phi = np.array([st, vs, ac])
    rls.update(phi, fm)
    gpr.add_data(st, vs, sign_v, fm)
    if t % 10 == 0 and len(gpr.X) >= 15:
        gpr.fit()
    mu_gpr, sig_gpr = gpr.predict(st, vs, sign_v)
    k_eff = gpr.get_local_stiffness(st, vs, sign_v)
    f_max_eff = f_max_safe - 2.0 * sig_gpr
    if k_eff > 50.0: 
        f_max_eff = min(f_max_eff, 2.0)
    f_max_eff = max(f_max_eff, 0.5) 
    v_opt = run_mppi(p_cur, v_cur, theta_cur, anchor_est, L0_est, rls.theta[0], rls.theta[1], rls.theta[2], f_max_eff)
    tang = np.array([-np.sin(theta_cur), np.cos(theta_cur), 0.])
    u_cmd = v_opt * tang
    u_safe = apply_bio_cbf(p_cur, u_cmd, anchor_est, L0_est)
    v_prev = v_cur.copy()
    v_cur = u_safe.copy()
    p_cur = p_cur + dt * v_cur
    hp.append(p_cur.copy()); hf.append(fm); hv.append(np.linalg.norm(u_safe))
    htheta.append(np.degrees(theta_cur)); hs.append(st); hkeff.append(k_eff)
# ══════════════════════════════════════════════════════
# 8. 结果可视化
# ══════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lower Limb Rehab: MPPI + Bio-CBF v15\n(Dynamic Taubin Fitting & Hysteresis Robustness)", fontsize=14, fontweight='bold')
ax = axs[0, 0]
hp_arr = np.array(hp)
ax.plot(hp_arr[:, 0], hp_arr[:, 1], 'b-', lw=2, label='Actual Trajectory')
theta_ideal = np.linspace(theta_start, theta_target, 100)
ax.plot(R_path*np.cos(theta_ideal), R_path*np.sin(theta_ideal), 'g--', label='Reference Arc')
safe_R = L0_true + STRETCH_MAX
ax.plot(safe_R*np.cos(theta_ideal), safe_R*np.sin(theta_ideal), 'r:', label=f'Safety Boundary')
ax.plot(*anchor[:2], 'k+', ms=15, mew=2, label='True Anchor')
ax.plot(*anchor_est[:2], 'rx', ms=10, mew=2, label='Est Anchor (Taubin)')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_title('1. Trajectory & Dynamic Anchor Fitting')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.legend(fontsize=8)
ax = axs[0, 1]
t_axis = np.arange(len(hf))
ax.plot(t_axis, hf, 'b-', lw=1.5, label='Force Magnitude (N)')
ax.axhline(f_max_safe, color='r', ls='-', lw=1.5, label=f'F_max_safe={f_max_safe}N')
ax.set_ylabel('Force (N)', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax2 = ax.twinx()
ax2.plot(t_axis, hv, 'g-', lw=1.5, label='Tangential Velocity')
ax2.set_ylabel('Velocity (m/s)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax.set_title('2. MPPI Time-Force Tradeoff')
ax.set_xlabel('Time Step')
ax.grid(True, alpha=0.3)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)
ax = axs[1, 0]
ax.plot(t_axis, htheta, 'm-', lw=2, label='Joint Angle (deg)')
ax.set_ylabel('Angle (deg)', color='m')
ax.tick_params(axis='y', labelcolor='m')
ax2 = ax.twinx()
ax2.plot(t_axis, hs, 'y-', lw=1.5, label='Stretch (m)')
ax2.axhline(STRETCH_MAX, color='r', ls=':', label='Stretch Max')
ax2.set_ylabel('Stretch (m)', color='y')
ax2.tick_params(axis='y', labelcolor='y')
ax.set_title('3. Motion Progression & Deformation Safety')
ax.set_xlabel('Time Step')
ax.grid(True, alpha=0.3)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='center right', fontsize=8)
ax = axs[1, 1]
ax.plot(t_axis, hkeff, 'c-', lw=2, label='Local GPR k_eff')
true_ks_curve = [get_true_stiffness(np.radians(a)) for a in htheta]
ax.plot(t_axis, true_ks_curve, 'k--', label='True Base Stiffness')
ax.set_title('4. Stiffness Estimation under Hysteresis')
ax.set_xlabel('Time Step')
ax.set_ylabel('Stiffness (N/m)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
out_name = 'mppi_cbf_sim_v15.png'
plt.savefig(out_name, dpi=150, bbox_inches='tight')
print(f"\n[结果] 图像已保存: {out_name}")
viol = np.mean(np.array(hf) > f_max_safe) * 100
print(f"[指标] 力超限比例: {viol:.2f}%")
print(f"[指标] 到达目标时间: {len(hf)*dt:.2f}s")