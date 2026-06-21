import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ── 真实函数（假装不知道）───────────────────────────
def true_force(x):
    return 2.0 * x + 0.5 * np.sin(3 * x)  # 模拟橡皮筋的非线性力-位移关系

# ── 采样几个观测点（模拟机械臂采集的数据）──────────
np.random.seed(42)
X_train = np.linspace(0.2, 4.8, 10).reshape(-1, 1)
y_train = true_force(X_train.flatten()) + np.random.normal(0, 0.2, len(X_train))

# ── 定义 GPR 模型 ────────────────────────────────────
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

# ── 用观测数据训练 ───────────────────────────────────
gpr.fit(X_train, y_train)

# ── 在新点上预测 ─────────────────────────────────────
X_test = np.linspace(0, 5, 200).reshape(-1, 1)
mu, sigma = gpr.predict(X_test, return_std=True)

# ── 画图 ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))

# 真实函数
plt.plot(X_test, true_force(X_test), 'k--', label='True function')

# GPR 预测均值
plt.plot(X_test, mu, 'b-', label='GPR mean')

# 置信区间：均值 ± 2倍标准差（约95%置信区间）
plt.fill_between(X_test.flatten(),
                 mu - 2 * sigma,
                 mu + 2 * sigma,
                 alpha=0.3, color='blue', label='95% confidence')

# 观测数据点
plt.scatter(X_train, y_train, c='red', zorder=5, label='Observations')

plt.xlabel('Displacement (m)')
plt.ylabel('Force (N)')
plt.legend()
plt.title('GPR: force-displacement fitting')
plt.savefig('gpr_demo.png', dpi=150)
plt.show()