"""
Integrated 3D Simulation — 大纲四层架构 v4

场景：机械臂引导患者手臂沿弧线运动，手臂刚度极大，过临界点后力急剧上升。

改动（相对 v3）：
1. ks_true=50，f_max_safe=3N，模拟大刚度患者手臂
2. 力模型：L0 之前纯噪声，L0 之后线性大刚度，去掉非线性项
3. noise_std=0.1，始终存在
4. 整定阶段退出条件：步数超过20步，不看力
5. 整定和弧线阶段：力超 f_max_safe 立刻回缩
6. MPC 目标：维持力在安全范围内，尽快完成角度任务
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
ks_true    = 50.0       # 大刚度：模拟患者手臂
b_true     = 0.5
m_true     = 0.05
L0_true    = 0.5        # 自然长度（控制器未知）
anchor     = np.array([0.0, 0.0, 0.0])
f_max_safe = 3.0        # 患者安全上限（超过会痛）
noise_std  = 0.1        # 传感器噪声，始终存在

def true_force_mag(stretch, vel, acc):
    if stretch <= 0:
        return 0.0      # L0 之前无弹力
    return ks_true * stretch + b_true * vel + m_true * acc

def true_force_3d(p, vel=0.0, acc=0.0):
    d = p - anchor; dist = np.linalg.norm(d)
    if dist < 1e-6:
        return np.random.normal(0, noise_std, 3)
    stretch = max(0.0, dist - L0_true)
    f = max(0.0, true_force_mag(stretch, vel, acc))
    # 噪声始终存在
    noise = np.random.normal(0, noise_std, 3)
    if f < 1e-6:
        return noise    # L0 之前：纯噪声
    return f * d/dist + noise

# ══════════════════════════════════════════════════════
# MPC 权重超参数
# ══════════════════════════════════════════════════════
W_FORCE  = 20.0
W_TIME   = 0.5
W_INPUT  = 0.05
W_ANGLE  = 10.0
V_MAX    = 0.3
MPC_N    = 8

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
    def __init__(self, lam=0.97):
        self.lam=lam
        self.theta=np.array([10.0, 0.3, 0.05])  # 初始猜测，ks偏小
        self.P=np.diag([50., 3., 1.])

    def update(self, phi, y):
        e=y-phi@self.theta; d=self.lam+phi@self.P@phi
        K=self.P@phi/d; self.theta+=K*e
        self.P=(1./self.lam)*(np.eye(3)-np.outer(K,phi))@self.P
        self.theta=np.clip(self.theta,[0.05,0.,0.],[200.,5.,2.])
        return self.theta.copy()

    def force_bounds(self, f_taut):
        f_lower = 0.8 * f_taut
        f_upper = min(2.5 * f_taut, f_max_safe)
        return f_lower, f_upper

# ══════════════════════════════════════════════════════
# GPR（建模层）
# ══════════════════════════════════════════════════════
class GPRModel:
    def __init__(self, max_data=60):
        k=(ConstantKernel(1.,(0.1,10.))*RBF(0.05,(0.005,1.))+WhiteKernel(0.1,(1e-3,1.)))
        self.gpr=GaussianProcessRegressor(kernel=k,n_restarts_optimizer=1,normalize_y=True)
        self.X=[]; self.y=[]; self.fitted=False
        self.xm=np.zeros(2); self.xs=np.ones(2); self.max_data=max_data

    def add_data(self, s, v, f):
        self.X.append([s,v]); self.y.append(f)
        if len(self.X)>self.max_data: self.X.pop(0); self.y.pop(0)

    def fit(self):
        if len(self.X)<8: return
        X=np.array(self.X); self.xm=X.mean(0); self.xs=X.std(0)+1e-6
        self.gpr.fit((X-self.xm)/self.xs, np.array(self.y)); self.fitted=True

    def predict(self, s, v):
        if not self.fitted: return None, None
        xn=(np.array([[s,v]])-self.xm)/self.xs
        mu,sig=self.gpr.predict(xn,return_std=True)
        return float(mu[0]), float(sig[0])

    def local_ks(self, stretch, vel, ks_fallback, delta=0.001, sigma_threshold=0.3):
        if not self.fitted:
            return ks_fallback
        _, sigma_cur = self.predict(stretch, vel)
        if sigma_cur is None or sigma_cur > sigma_threshold:
            return ks_fallback
        s_plus  = max(0., stretch + delta)
        s_minus = max(0., stretch - delta)
        mu_plus,  _ = self.predict(s_plus,  vel)
        mu_minus, _ = self.predict(s_minus, vel)
        if mu_plus is None or mu_minus is None:
            return ks_fallback
        ks_eff = (mu_plus - mu_minus) / (2 * delta)
        return max(0.05, ks_eff)

# ══════════════════════════════════════════════════════
# 安全回缩：力超限时往锚点方向退
# ══════════════════════════════════════════════════════
def retract_velocity(p_cur, anchor_est, speed=0.1):
    d = anchor_est - p_cur
    norm = np.linalg.norm(d)
    if norm < 1e-6: return np.zeros(3)
    return d / norm * speed

# ══════════════════════════════════════════════════════
# Tube-MPC
# ══════════════════════════════════════════════════════
def run_mpc_3d(p_cur, v_cur, ks_eff, b_rls, m_rls,
               anchor_est, L0_est,
               theta_cur, theta_target,
               f_taut, f_lower, f_upper,
               sigma_gpr=0.):
    f_max_eff = f_upper - 2.0*np.clip(sigma_gpr, 0., 0.5)
    f_max_eff = max(f_max_eff, f_taut*1.1)

    def pf(p_, v_, a_):
        stretch = max(0., np.linalg.norm(p_-anchor_est) - L0_est)
        return max(0., ks_eff*stretch + b_rls*np.linalg.norm(v_) + m_rls*np.linalg.norm(a_))

    def cost(u_flat):
        u_seq=u_flat.reshape(MPC_N,3); p=p_cur.copy(); v=v_cur.copy(); c=0.
        for k in range(MPC_N):
            vk=u_seq[k]; ak=(vk-v)/dt; pn=p+dt*vk
            c += W_FORCE*(pf(pn,vk,ak)-f_taut)**2
            c += W_TIME
            c += W_INPUT*float(np.dot(vk,vk))
            tn=np.arctan2(pn[1]-anchor_est[1], pn[0]-anchor_est[0])
            pg=tn-theta_cur
            if pg<-np.pi: pg+=2*np.pi
            c -= W_ANGLE*np.clip(pg, 0., 0.15)
            p=pn; v=vk
        c += W_FORCE*(pf(p,v,np.zeros(3))-f_taut)**2
        return c

    cons=[]
    for k in range(MPC_N):
        def make_upper(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return f_max_eff - pf(p,v,np.zeros(3))
            return fn
        def make_lower(k_=k):
            def fn(u):
                p=p_cur.copy(); v=v_cur.copy()
                for i in range(k_+1): v=u.reshape(MPC_N,3)[i]; p=p+dt*v
                return pf(p,v,np.zeros(3)) - f_lower
            return fn
        cons.append({'type':'ineq','fun':make_upper(k)})
        cons.append({'type':'ineq','fun':make_lower(k)})

    tang=np.array([-np.sin(theta_cur), np.cos(theta_cur), 0.])
    u0=np.tile(tang*0.1, MPC_N)
    res=minimize(cost, u0, method='SLSQP',
                 bounds=[(-V_MAX,V_MAX)]*(MPC_N*3),
                 constraints=cons,
                 options={'maxiter':60,'ftol':1e-3})
    u_out=res.x.reshape(MPC_N,3)[0]
    return u_out if (res.success or res.fun<1e8) else tang*0.05

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
T_sim        = 700
theta_start  = np.radians(-20)
theta_target = theta_start + np.pi/2

R_init     = L0_true * 0.35
p_cur      = anchor + R_init*np.array([np.cos(theta_start),np.sin(theta_start),0.])
v_cur      = np.zeros(3); v_prev=np.zeros(3)
L0_est     = R_init*0.8; anchor_est=anchor.copy()

bocd=BOCD(); rls=RLS(); gpr=GPRModel()

phase=0
taut_step=None; arc_step=None; settle_count=0
f_taut=0.2; f_lower=0.1; f_upper=1.0
sigma_s=0.

hp=[p_cur.copy()]; hf=[]; hsig=[]; hks=[]
hgmu=[]; hgsig=[]; hL0=[L0_est]; hphase=[]
h_flower=[]; h_fupper=[]
f_cur=true_force_3d(p_cur)

print("="*62)
print("Integrated 3D Simulation  —  大纲四层架构 v4")
print(f"ks_true={ks_true}  L0_true={L0_true}m  f_max_safe={f_max_safe}N")
print(f"起始θ={np.degrees(theta_start):.0f}°  目标θ={np.degrees(theta_target):.0f}°")
print("="*62)

for t in range(T_sim):
    fm=np.linalg.norm(f_cur); hf.append(fm)
    sig=bocd.update(fm); hsig.append(sig)

    theta_cur=np.arctan2(p_cur[1]-anchor_est[1], p_cur[0]-anchor_est[0])

    # ── 阶段转换 ─────────────────────────────────────
    if phase==0 and sig>0.35:
        phase=1; taut_step=t
        f_taut  = max(fm, 0.1)
        f_lower, f_upper = rls.force_bounds(f_taut)
        L0_est  = max(0.05, np.linalg.norm(p_cur-anchor_est) - fm/10.0)
        print(f"  [BOCD]  t={t:3d}: 检测到绷直 → 进入整定阶段")
        print(f"          sig={sig:.3f}  f_taut={f_taut:.3f}N  L0_est={L0_est:.3f}m")

    if phase==1 and settle_count>=20:
        phase=2; arc_step=t
        print(f"  [整定]  t={t:3d}: 整定完成 → 进入弧线阶段")
        print(f"          ks_rls={rls.theta[0]:.2f} (真值{ks_true})  dist={np.linalg.norm(p_cur-anchor_est):.3f}m")

    if phase==2 and theta_cur>=theta_target-0.03:
        print(f"  [Done]  t={t:3d}: θ={np.degrees(theta_cur):.1f}° 到达目标!")
        break

    # ── 安全检查：任何阶段力超限立刻回缩 ────────────
    if fm >= f_max_safe and phase>=1:
        u = retract_velocity(p_cur, anchor_est, speed=0.15)
        hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(sigma_s)
        h_flower.append(f_lower); h_fupper.append(f_upper)
        hL0.append(L0_est); hphase.append(phase)
        v_prev=v_cur.copy(); v_cur=u.copy()
        pn=p_cur+dt*u
        f_cur=true_force_3d(pn, np.linalg.norm(u), np.linalg.norm((u-v_prev)/dt))
        p_cur=pn; hp.append(p_cur.copy())
        continue

    # ── 阶段0：探索 ──────────────────────────────────
    if phase==0:
        dist=np.linalg.norm(p_cur-anchor_est); dn=(p_cur-anchor_est)/(dist+1e-6)
        sp=0.15*max(0.3, 1.4-dist/(L0_true*1.3)); u=dn*sp
        if fm<noise_std*2: L0_est=dist*0.9
        hks.append(rls.theta[0]); hgmu.append(0.); hgsig.append(0.)
        h_flower.append(f_lower); h_fupper.append(f_upper)

    # ── 阶段1：整定（20步后转弧线）──────────────────
    elif phase==1:
        d=p_cur-anchor_est; dist=np.linalg.norm(d)
        st=max(0., dist-L0_est); vs=np.linalg.norm(v_cur)
        ac=np.linalg.norm((v_cur-v_prev)/dt)
        phi=np.array([st,vs,ac]); trls=rls.update(phi,fm)
        gpr.add_data(st,vs,fm)
        if t%8==0: gpr.fit()

        f_lower, f_upper = rls.force_bounds(f_taut)
        f_upper = min(f_upper, f_max_safe)

        # 缓速外伸，RLS 预热
        R_target = L0_est + f_taut*2.0/max(trls[0], 0.1)
        d_norm=d/(dist+1e-6)
        u=d_norm*0.08 if dist<R_target else np.zeros(3)
        settle_count+=1

        mu,sr=gpr.predict(st,vs)
        if sr is None: sr=0.
        mu_val=trls[0]*st if mu is None else mu
        sigma_s=0.2*sr+0.8*sigma_s
        hks.append(trls[0]); hgmu.append(float(mu_val)); hgsig.append(sigma_s)
        h_flower.append(f_lower); h_fupper.append(f_upper)

    # ── 阶段2：弧线（Tube-MPC）──────────────────────
    else:
        d=p_cur-anchor_est; dist=np.linalg.norm(d)
        st=max(0., dist-L0_est); vs=np.linalg.norm(v_cur)
        ac=np.linalg.norm((v_cur-v_prev)/dt)
        phi=np.array([st,vs,ac]); trls=rls.update(phi,fm)
        gpr.add_data(st,vs,fm)
        if t%10==0: gpr.fit()

        f_lower, f_upper = rls.force_bounds(f_taut)
        f_upper = min(f_upper, f_max_safe)

        mu,sr=gpr.predict(st,vs)
        if sr is None: sr=0.
        mu_val=trls[0]*st if mu is None else mu
        sigma_s=0.2*sr+0.8*sigma_s

        ks_eff=gpr.local_ks(st, vs, ks_fallback=trls[0])
        b_rls, m_rls = trls[1], trls[2]

        hks.append(trls[0]); hgmu.append(float(mu_val)); hgsig.append(sigma_s)
        h_flower.append(f_lower); h_fupper.append(f_upper)

        u=run_mpc_3d(p_cur, v_cur, ks_eff, b_rls, m_rls,
                     anchor_est, L0_est,
                     theta_cur, theta_target,
                     f_taut, f_lower, f_upper,
                     sigma_gpr=sigma_s)

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
    print(f"[指标] 总步={len(hf_a)}  整定步={arc_step-taut_step}  弧线步={len(fc)}")
    pts2d=hp[arc_step:,:2]
    xc,yc,R_fit=taubin_fit(pts2d)
    if R_fit is not None:
        print(f"[Taubin] 圆心({xc:.3f},{yc:.3f}) R={R_fit:.3f}m")
    else:
        print("[Taubin] 拟合失败")

# ══════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════
fig=plt.figure(figsize=(16,10))
fig.suptitle("Integrated 3D Simulation  —  大纲四层架构 v4\n"
             "大刚度手臂模型  ks=50  f_max_safe=3N",
             fontsize=11, y=0.99)

n=len(hf_a)
phase_arr=np.array(hphase+[hphase[-1] if hphase else 0])

# 1. 轨迹
ax1=fig.add_subplot(2,3,1)
p0=np.where(phase_arr==0)[0]; p1=np.where(phase_arr==1)[0]; p2=np.where(phase_arr==2)[0]
if len(p0)>0: ax1.plot(hp[p0,0],hp[p0,1],'gray',lw=1.2,label='Phase0: explore',alpha=0.7)
if len(p1)>0: ax1.plot(hp[p1,0],hp[p1,1],'orange',lw=1.5,label='Phase1: settle')
if len(p2)>0: ax1.plot(hp[p2,0],hp[p2,1],'b-',lw=2,label='Phase2: arc (MPC)')
ax1.plot(*anchor[:2],'k+',ms=12,mew=2,label='Anchor')
if arc_step is not None: ax1.plot(*hp[arc_step,:2],'go',ms=8,zorder=5,label=f'Arc t={arc_step}')
if R_fit is not None:
    tha=np.linspace(theta_start,theta_target,300)
    ax1.plot(xc+R_fit*np.cos(tha),yc+R_fit*np.sin(tha),'m:',lw=1.5,label=f'Taubin R={R_fit:.2f}m')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_aspect('equal')
ax1.legend(fontsize=7,loc='upper left'); ax1.set_title('Trajectory (top view)'); ax1.grid(True,alpha=0.3)

# 2. 力 + 动态边界
ax2=fig.add_subplot(2,3,2)
ax2.plot(np.arange(n),hf_a,'b-',lw=1.2,label='Force magnitude',alpha=0.85)
ax2.axhline(f_max_safe,color='red',ls='-',lw=1.5,label=f'f_max_safe={f_max_safe}N')
if taut_step is not None:
    ax2.axvline(taut_step,color='green',ls='--',alpha=0.7,label=f'Taut t={taut_step}')
    ax2.axhline(f_taut,color='purple',ls='--',lw=1.2,label=f'f_taut={f_taut:.2f}N')
if arc_step is not None:
    ax2.axvline(arc_step,color='blue',ls='--',alpha=0.7,label=f'Arc t={arc_step}')
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

# 4. GPR 不确定度
ax4=fig.add_subplot(2,3,4)
if arc_step is not None and len(arc_gsig)>2:
    ax4.plot(tc3,arc_gsig,color='purple',lw=1.5,label='GPR σ (smoothed)')
    ax4.fill_between(tc3,0,arc_gsig,alpha=0.2,color='purple')
    ax4.axhline(0.3,color='red',ls='--',lw=1,label='σ threshold=0.3')
    ax4.set_xlabel('Time step')
ax4.set_ylabel('σ (N)'); ax4.legend(fontsize=7)
ax4.set_title('GPR uncertainty → Tube margin'); ax4.grid(True,alpha=0.3)

# 5. RLS 刚度收敛
ax5=fig.add_subplot(2,3,5)
ks_all=np.array(hks)
ax5.plot(ks_all,'b-',lw=1.5,label='RLS ks estimate')
ax5.axhline(ks_true,color='r',ls='--',label=f'True ks={ks_true}')
if taut_step is not None: ax5.axvline(taut_step,color='green',ls='--',alpha=0.6,label='Taut')
if arc_step  is not None: ax5.axvline(arc_step, color='blue', ls='--',alpha=0.6,label='Arc start')
ax5.set_ylabel('ks (N/m)'); ax5.set_xlabel('Time step')
ax5.legend(fontsize=7); ax5.set_title('Stiffness estimation (RLS)'); ax5.grid(True,alpha=0.3)

# 6. 半径演化
ax6=fig.add_subplot(2,3,6)
L0_arr=np.array(hL0)
ax6.plot(L0_arr,'g-',lw=1.5,label='L0 estimate')
ax6.axhline(L0_true,color='r',ls='--',label=f'True L0={L0_true}m')
dist_arr=np.linalg.norm(hp[:-1]-anchor,axis=1)[:n]
ax6.plot(dist_arr,'b-',lw=1,alpha=0.6,label='dist(t) from anchor')
if taut_step is not None: ax6.axvline(taut_step,color='green',ls='--',alpha=0.6)
if arc_step  is not None: ax6.axvline(arc_step, color='blue', ls='--',alpha=0.6)
ax6.set_ylabel('Distance (m)'); ax6.set_xlabel('Time step')
ax6.legend(fontsize=7); ax6.set_title('L0 estimate & radius evolution'); ax6.grid(True,alpha=0.3)

plt.tight_layout()
out='integrated_sim_3d.png'
plt.savefig(out,dpi=150,bbox_inches='tight')
print(f"\n[Plot] saved → {out}")