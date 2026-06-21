import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ── 真实系统参数（仿真用，控制器不知道）─────────────
dt = 0.1
ks_true = 3.0        # 真实刚度，控制器不知道
ks_init = 1.0        # 控制器初始猜测，故意猜错

def true_force(p):
    return ks_true * p + 0.3 * np.sin(3 * p)   # 非线性弹簧

# ── MPC 参数 ──────────────────────────────────────────
N     = 10
Q     = np.diag([1.0, 10.0])
R     = np.array([[0.1]])
f_max = 8.0
f_min = 0.0
v_max = 1.0
v_min = -1.0
x_ref = np.array([1.0, 3.0])

# ── GPR 初始化 ────────────────────────────────────────
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + \
         WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                normalize_y=True, alpha=1e-6)

X_data = []   # 位移数据
y_data = []   # 力数据

# ── 用 GPR 均值更新 A、B 矩阵 ─────────────────────────
def get_AB(ks):
    A = np.array([[1,   0],
                  [ks,  1]])
    B = np.array([[dt],
                  [ks * dt]])
    return A, B

# ── 仿真 ──────────────────────────────────────────────
T_sim = 60
x_cur = np.array([0.0, 0.0])
ks_est = ks_init     # 当前刚度估计

history_x  = [x_cur.copy()]
history_u  = []
history_ks = [ks_init]

for t in range(T_sim):

    # 用当前刚度估计构建 A、B
    A, B = get_AB(ks_est)

    # ── MPC 求解 ──────────────────────────────────────
    n, m = 2, 1
    x_var = cp.Variable((n, N + 1))
    u_var = cp.Variable((m, N))
    cost, constraints = 0, []
    constraints += [x_var[:, 0] == x_cur]

    for k in range(N):
        cost += cp.quad_form(x_var[:, k] - x_ref, Q)
        cost += cp.quad_form(u_var[:, k], R)
        constraints += [x_var[:, k+1] == A @ x_var[:, k] + B @ u_var[:, k]]
        constraints += [u_var[:, k] >= v_min, u_var[:, k] <= v_max]
        constraints += [x_var[1, k] >= f_min, x_var[1, k] <= f_max]

    cost += cp.quad_form(x_var[:, N] - x_ref, Q)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    u0 = u_var.value[:, 0]

    # ── 真实系统前进一步 ──────────────────────────────
    noise = np.random.normal(0, 0.05, 2)
    x_next = np.array([
        x_cur[0] + dt * u0[0],
        true_force(x_cur[0] + dt * u0[0])
    ]) + noise

    # ── 把新数据加进 GPR ──────────────────────────────
    X_data.append([x_next[0]])   # 位移
    y_data.append(x_next[1])     # 力

    # GPR 至少需要 3 个点才能拟合
    if len(X_data) >= 3:
        gpr.fit(np.array(X_data), np.array(y_data))

        # 用 GPR 估计当前位置附近的局部刚度
        # 数值微分：ks ≈ df/dp
        p_cur = x_next[0]
        dp = 0.01
        f_plus,  _ = gpr.predict([[p_cur + dp]], return_std=True)
        f_minus, _ = gpr.predict([[p_cur - dp]], return_std=True)
        ks_est = float((f_plus.item() - f_minus.item()) / (2 * dp))
        ks_est = np.clip(ks_est, 0.1, 10.0)   # 防止估计值跑飞

    x_cur = x_next
    history_x.append(x_cur.copy())
    history_u.append(u0[0])
    history_ks.append(ks_est)

# ── 画图 ──────────────────────────────────────────────
history_x  = np.array(history_x)
history_ks = np.array(history_ks)

fig, axes = plt.subplots(3, 1, figsize=(9, 9))

axes[0].plot(history_x[:, 1], label='Force')
axes[0].axhline(x_ref[1], color='r', linestyle='--', label='target (3N)')
axes[0].axhline(f_max,    color='orange', linestyle=':', label='f_max')
axes[0].set_ylabel('Force (N)')
axes[0].legend()

axes[1].plot(history_x[:, 0], label='Position')
axes[1].axhline(x_ref[0], color='r', linestyle='--', label='target (1m)')
axes[1].set_ylabel('Position (m)')
axes[1].legend()

axes[2].plot(history_ks, label='Estimated ks')
axes[2].axhline(ks_true, color='r', linestyle='--', label=f'True ks={ks_true}')
axes[2].axhline(ks_init, color='gray', linestyle=':', label=f'Initial guess={ks_init}')
axes[2].set_ylabel('Stiffness estimate')
axes[2].set_xlabel('Time step')
axes[2].legend()

plt.tight_layout()
plt.savefig('mpc_gpr_demo.png', dpi=150)
plt.show()