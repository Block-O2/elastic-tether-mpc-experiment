import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 真实系统（仿真用）────────────────────────────────
np.random.seed(42)
ks_true = 3.0      # 真实刚度，RLS 不知道
T = 100

# 模拟位移序列：从 0 慢慢增大
p = np.linspace(0.01, 2.0, T)

# 真实力 = 刚度 × 位移 + 噪声
f = ks_true * p + np.random.normal(0, 0.1, T)

# ── RLS 初始化 ────────────────────────────────────────
lambda_f = 0.95      # 遗忘因子
theta = 1.0          # 初始刚度猜测（故意猜错）
P = 10.0             # 初始不确定性（大 = 不确定）

history_theta = [theta]
history_P     = [P]

# ── 主循环 ────────────────────────────────────────────
for t in range(T):
    phi = p[t]       # 回归向量（这里就是位移）
    y   = f[t]       # 观测值（力）

    # 第一步：预测误差
    e = y - phi * theta

    # 第二步：增益
    K = P * phi / (lambda_f + phi**2 * P)

    # 第三步：更新参数
    theta = theta + K * e

    # 第四步：更新协方差
    P = (1 / lambda_f) * (P - K * phi * P)

    history_theta.append(theta)
    history_P.append(P)

# ── 画图 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 6))

axes[0].plot(history_theta, label='RLS estimate of ks')
axes[0].axhline(ks_true, color='r', linestyle='--', label=f'True ks={ks_true}')
axes[0].axhline(1.0, color='gray', linestyle=':', label='Initial guess=1.0')
axes[0].set_ylabel('Stiffness ks')
axes[0].legend()

axes[1].plot(history_P, label='Uncertainty P')
axes[1].set_ylabel('P (uncertainty)')
axes[1].set_xlabel('Time step')
axes[1].legend()

plt.tight_layout()
plt.savefig('rls_demo.png', dpi=150)
print("done")