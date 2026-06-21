import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# ── 系统参数 ──────────────────────────────────────────
dt = 0.1        # 时间步长
ks = 2.0        # 弹簧刚度（真实值，仿真用）
N  = 3         # 预测时域步数

# 状态：x = [position, force]，输入：u = velocity
A = np.array([[1,    0],
              [ks,   1]])
B = np.array([[dt],
              [ks * dt]])

n = 2   # 状态维度
m = 1   # 输入维度

# ── 代价函数权重 ──────────────────────────────────────
Q = np.diag([1.0, 10.0])   # 位置误差权重大一些
R = np.array([[0.1]])

# ── 约束 ─────────────────────────────────────────────
f_max  =  5.0   # 力上限 (N)
f_min  =  0.0   # 力下限
v_max  =  1.0   # 速度上限
v_min  = -1.0

# ── 目标状态 ──────────────────────────────────────────
x_ref = np.array([1.0, 3.0])   # 目标：位置1m，力3N

# ── 仿真设置 ──────────────────────────────────────────
T_sim = 50                          # 仿真步数
x0    = np.array([0.0, 0.0])        # 初始状态
x_cur = x0.copy()

history_x = [x0.copy()]
history_u = []

# ── 主循环 ────────────────────────────────────────────
for t in range(T_sim):

    # 定义优化变量
    x_var = cp.Variable((n, N + 1))
    u_var = cp.Variable((m, N))

    cost        = 0
    constraints = []

    # 初始状态约束
    constraints += [x_var[:, 0] == x_cur]

    for k in range(N):
        # 代价累加
        cost += cp.quad_form(x_var[:, k] - x_ref, Q)
        cost += cp.quad_form(u_var[:, k], R)

        # 系统模型约束
        constraints += [x_var[:, k + 1] == A @ x_var[:, k] + B @ u_var[:, k]]

        # 输入约束
        constraints += [u_var[:, k] >= v_min,
                        u_var[:, k] <= v_max]

        # 状态约束（力）
        constraints += [x_var[1, k] >= f_min,
                        x_var[1, k] <= f_max]

    # 终端代价
    cost += cp.quad_form(x_var[:, N] - x_ref, Q)

    # 求解
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    # 取第一步控制输入
    u0 = u_var.value[:, 0]

    # 真实系统前进一步（加一点噪声模拟模型误差）
    noise = np.random.normal(0, 0.05, n)
    x_cur = A @ x_cur + B.flatten() * u0[0] + noise

    history_x.append(x_cur.copy())
    history_u.append(u0[0])

# ── 画图 ──────────────────────────────────────────────
history_x = np.array(history_x)
history_u = np.array(history_u)

fig, axes = plt.subplots(3, 1, figsize=(8, 8))

axes[0].plot(history_x[:, 0])
axes[0].axhline(x_ref[0], color='r', linestyle='--', label='target')
axes[0].set_ylabel('Position (m)')
axes[0].legend()

axes[1].plot(history_x[:, 1])
axes[1].axhline(x_ref[1], color='r', linestyle='--', label='target')
axes[1].axhline(f_max,    color='orange', linestyle=':', label='f_max')
axes[1].set_ylabel('Force (N)')
axes[1].legend()

axes[2].plot(history_u)
axes[2].axhline(v_max, color='orange', linestyle=':', label='v_max')
axes[2].axhline(v_min, color='orange', linestyle=':')
axes[2].set_ylabel('Velocity input (m/s)')
axes[2].set_xlabel('Time step')
axes[2].legend()

plt.tight_layout()
plt.savefig('mpc_demo.png', dpi=150)
plt.show()