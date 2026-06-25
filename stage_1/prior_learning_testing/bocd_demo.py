import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

# ── 生成模拟力传感器序列 ──────────────────────────────
np.random.seed(42)
segment1 = np.random.normal(0.2, 0.1, 70)
segment2 = np.random.normal(3.0, 0.1, 130)
y = np.concatenate([segment1, segment2])
true_changepoint = 70

# ── BOCD ─────────────────────────────────────────────
mu0, kappa0, alpha0, beta0 = 0.2, 1.0, 2.0, 1.0
mu    = np.array([mu0])
kappa = np.array([kappa0])
alpha = np.array([alpha0])
beta  = np.array([beta0])
R     = np.array([1.0])
h     = 0.1

prev_mode_r = 0
changepoint_signal = []

for t, x in enumerate(y):
    df    = 2 * alpha
    scale = np.maximum(np.sqrt(beta * (kappa + 1) / (alpha * kappa)), 1e-10)
    log_pi = np.clip(student_t.logpdf(x, df=df, loc=mu, scale=scale), -50, 50)
    pi     = np.exp(log_pi)

    R_new      = np.empty(len(R) + 1)
    R_new[0]   = np.sum(R * pi) * h
    R_new[1:]  = R * pi * (1 - h)
    total      = R_new.sum()
    R_new      = R_new / total if total > 0 else np.array([1.0] + [0.0]*len(R))
    R          = R_new

    mode_r = np.argmax(R)
    drop   = max(0, prev_mode_r - mode_r)
    changepoint_signal.append(drop / max(1, prev_mode_r))
    prev_mode_r = mode_r

    kappa_new = kappa + 1
    mu_new    = (kappa * mu + x) / kappa_new
    alpha_new = alpha + 0.5
    beta_new  = beta + kappa * (x - mu)**2 / (2 * kappa_new)
    mu    = np.append([mu0],    mu_new)
    kappa = np.append([kappa0], kappa_new)
    alpha = np.append([alpha0], alpha_new)
    beta  = np.append([beta0],  beta_new)

changepoint_signal = np.array(changepoint_signal)

# ── 画图 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(y, color='steelblue', label='Force sensor')
axes[0].axvline(true_changepoint, color='red', linestyle='--', label='True changepoint')
axes[0].set_ylabel('Force (N)')
axes[0].legend()

axes[1].plot(changepoint_signal, color='darkorange', label='Changepoint signal')
axes[1].axvline(true_changepoint, color='red', linestyle='--', label='True changepoint')
axes[1].axhline(0.5, color='gray', linestyle=':', label='threshold=0.5')
axes[1].set_ylabel('Signal strength')
axes[1].set_xlabel('Time step')
axes[1].legend()

plt.tight_layout()
plt.savefig('bocd_demo.png', dpi=150)
print("done")