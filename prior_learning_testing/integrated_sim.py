import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import t as student_t

# ══════════════════════════════════════════════════════
# 真实系统
# ══════════════════════════════════════════════════════
np.random.seed(42)
dt      = 0.1
ks_true = 3.0

def true_force(p):
    return ks_true * p + 0.3 * np.sin(3 * p)

# ══════════════════════════════════════════════════════
# BOCD 模块（和之前一样）
# ══════════════════════════════════════════════════════
class BOCD:
    def __init__(self, hazard=0.1, mu0=0.2, kappa0=1.0, alpha0=2.0, beta0=1.0):
        self.h      = hazard
        self.mu0    = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0  = beta0
        self.mu     = np.array([mu0])
        self.kappa  = np.array([kappa0])
        self.alpha  = np.array([alpha0])
        self.beta   = np.array([beta0])
        self.R      = np.array([1.0])
        self.prev_mode_r = 0

    def update(self, x):
        df    = 2 * self.alpha
        scale = np.maximum(np.sqrt(self.beta * (self.kappa + 1) /
                                   (self.alpha * self.kappa)), 1e-10)
        log_pi = np.clip(student_t.logpdf(x, df=df, loc=self.mu, scale=scale), -50, 50)
        pi     = np.exp(log_pi)

        R_new     = np.empty(len(self.R) + 1)
        R_new[0]  = np.sum(self.R * pi) * self.h
        R_new[1:] = self.R * pi * (1 - self.h)
        total     = R_new.sum()
        R_new     = R_new / total if total > 0 else np.array([1.0] + [0.0]*len(self.R))
        self.R    = R_new

        mode_r   = np.argmax(R_new)
        drop     = max(0, self.prev_mode_r - mode_r)
        signal   = drop / max(1, self.prev_mode_r)
        self.prev_mode_r = mode_r

        kappa_new = self.kappa + 1
        mu_new    = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new  = self.beta + self.kappa * (x - self.mu)**2 / (2 * kappa_new)
        self.mu    = np.append([self.mu0],    mu_new)
        self.kappa = np.append([self.kappa0], kappa_new)
        self.alpha = np.append([self.alpha0], alpha_new)
        self.beta  = np.append([self.beta0],  beta_new)

        return signal

# ══════════════════════════════════════════════════════
# RLS 模块（和之前一样）
# ══════════════════════════════════════════════════════
class RLS:
    def __init__(self, lambda_f=0.95, theta0=1.0, P0=10.0):
        self.lam   = lambda_f
        self.theta = theta0
        self.P     = P0

    def update(self, phi, y):
        e          = y - phi * self.theta
        K          = self.P * phi / (self.lam + phi**2 * self.P)
        self.theta = self.theta + K * e
        self.P     = (1 / self.lam) * (self.P - K * phi * self.P)
        return self.theta

# ══════════════════════════════════════════════════════
# GPR 模块（和之前一样）
# ══════════════════════════════════════════════════════
class GPRModel:
    def __init__(self):
        kernel   = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + \
                   WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                             normalize_y=True, alpha=1e-6)
        self.X      = []
        self.y      = []
        self.fitted = False

    def add_data(self, p, f):
        self.X.append([p])
        self.y.append(f)

    def fit(self):
        if len(self.X) >= 5:
            self.gpr.fit(np.array(self.X), np.array(self.y))
            self.fitted = True

    def predict(self, p):
        if not self.fitted:
            return None, None
        mu, sigma = self.gpr.predict([[p]], return_std=True)
        return float(mu[0]), float(sigma[0])

# ══════════════════════════════════════════════════════
# 非线性 MPC 模块（新）
# ══════════════════════════════════════════════════════
def run_mpc_nonlinear(x_cur, gpr_model, ks_fallback, sigma=0.0, N=10, dt=0.1):
    """
    x_cur       : 当前状态 [position, force]
    gpr_model   : GPR 模型，用于预测力
    ks_fallback : GPR 不可用时的降级刚度估计（RLS 提供）
    sigma       : GPR 预测标准差，用于收紧约束
    """
    Q     = np.diag([1.0, 10.0])
    R_mat = 0.1
    x_ref = np.array([1.0, 3.0])
    f_max = 8.0 - 2.0 * sigma
    f_min = 0.0
    v_max = 1.0
    v_min = -1.0

    # ── 预测模型 ──────────────────────────────────────
    def predict_next(p_cur, f_cur, v):
        """用 GPR 预测下一步状态，GPR 不可用时降级为线性模型"""
        p_next = p_cur + dt * v
        mu, _ = gpr_model.predict(p_next)
        if mu is None:
            # 降级：用 RLS 刚度线性预测
            f_next = f_cur + ks_fallback * dt * v
        else:
            f_next = mu
        return np.array([p_next, f_next])

    # ── 代价函数 ──────────────────────────────────────
    def cost_fn(u_flat):
        u_seq = u_flat.reshape(N)
        x     = x_cur.copy()
        total_cost = 0.0
        for k in range(N):
            err = x - x_ref
            total_cost += err @ Q @ err + R_mat * u_seq[k]**2
            x = predict_next(x[0], x[1], u_seq[k])
        # 终端代价
        total_cost += (x - x_ref) @ Q @ (x - x_ref)
        return total_cost

    # ── 约束 ─────────────────────────────────────────
    constraints = []
    def force_constraint_k(k):
        def fn(u_flat):
            u_seq = u_flat.reshape(N)
            x = x_cur.copy()
            for i in range(k + 1):
                x = predict_next(x[0], x[1], u_seq[i])
            return f_max - x[1]   # f_max - f >= 0
        return fn

    def force_min_constraint_k(k):
        def fn(u_flat):
            u_seq = u_flat.reshape(N)
            x = x_cur.copy()
            for i in range(k + 1):
                x = predict_next(x[0], x[1], u_seq[i])
            return x[1] - f_min   # f - f_min >= 0
        return fn

    for k in range(N):
        constraints.append({'type': 'ineq', 'fun': force_constraint_k(k)})
        constraints.append({'type': 'ineq', 'fun': force_min_constraint_k(k)})

    # ── 求解 ─────────────────────────────────────────
    u0     = np.zeros(N)
    bounds = [(v_min, v_max)] * N

    result = minimize(cost_fn, u0, method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 50, 'ftol': 1e-4})

    if result.success:
        return float(result.x[0])
    else:
        return 0.0

# ══════════════════════════════════════════════════════
# 主仿真
# ══════════════════════════════════════════════════════
T_sim = 100
x_cur = np.array([0.0, 0.0])
taut  = False

bocd = BOCD(hazard=0.1, mu0=0.1)
rls  = RLS(lambda_f=0.95, theta0=1.0, P0=10.0)
gpr  = GPRModel()

history_x      = [x_cur.copy()]
history_u      = []
history_ks     = []
history_signal = []

for t in range(T_sim):
    f_cur  = x_cur[1]
    signal = bocd.update(f_cur)
    history_signal.append(signal)

    if not taut and signal > 0.5:
        taut = True
        print(f"Step {t}: 检测到绷直！")

    if not taut:
        u = 0.1
    else:
        p_cur  = x_cur[0]
        ks_est = rls.update(p_cur, f_cur)
        ks_est = np.clip(ks_est, 0.1, 10.0)

        gpr.add_data(p_cur, f_cur)
        gpr.fit()

        _, sigma = gpr.predict(p_cur)
        sigma    = sigma if sigma is not None else 0.0

        u = run_mpc_nonlinear(x_cur, gpr, ks_est, sigma=sigma)

    history_ks.append(rls.theta)
    history_u.append(u)

    p_next = x_cur[0] + dt * u
    f_next = true_force(p_next) + np.random.normal(0, 0.05)
    x_cur  = np.array([p_next, f_next])
    history_x.append(x_cur.copy())

# ══════════════════════════════════════════════════════
# 画图
# ══════════════════════════════════════════════════════
history_x  = np.array(history_x)
history_ks = np.array(history_ks)

fig, axes = plt.subplots(4, 1, figsize=(10, 12))

axes[0].plot(history_x[:, 1], label='Force')
axes[0].axhline(3.0, color='r', linestyle='--', label='target 3N')
axes[0].axhline(8.0, color='orange', linestyle=':', label='f_max')
axes[0].set_ylabel('Force (N)')
axes[0].legend()

axes[1].plot(history_x[:, 0], label='Position')
axes[1].set_ylabel('Position (m)')
axes[1].legend()

axes[2].plot(history_ks, label='RLS ks estimate')
axes[2].axhline(ks_true, color='r', linestyle='--', label=f'True ks={ks_true}')
axes[2].set_ylabel('Stiffness ks')
axes[2].legend()

axes[3].plot(history_signal, label='BOCD signal')
axes[3].axhline(0.5, color='gray', linestyle=':', label='threshold')
axes[3].set_ylabel('Changepoint signal')
axes[3].set_xlabel('Time step')
axes[3].legend()

plt.tight_layout()
plt.savefig('integrated_sim.png', dpi=150)
print("done")