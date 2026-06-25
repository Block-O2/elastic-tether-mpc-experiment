"""
run_mujoco_sim.py  ——  MuJoCo physics backend for the v15 algorithm stack.

Design principle
----------------
  ALL algorithm code (BOCD, RLS, GPRModel, run_mpc_3d_qp, run_mpc_3d, every
  constant) is imported directly from ongoing_parts/integrated_sim_3d.py.
  This file adds zero algorithm logic of its own.

  The only thing that changes vs. run_sim():
    true_force_3d(pn, vs2, as2)  →  env.step(desired_pos) + env.get_obs()

  dt = 0.05 s (from integrated_sim_3d) is matched by setting n_substeps=25
  in the env constructor (25 × 0.002 s/step = 0.05 s per env.step() call).

Usage
-----
    from elastic_env import ElasticTetherMJEnv
    from run_mujoco_sim import run_mujoco_sim

    env = ElasticTetherMJEnv('models/scene.xml', n_substeps=25)
    env.set_material_preset('rubber_band')   # ground truth — algorithm never sees this
    result = run_mujoco_sim(env, arc_deg=90, STRETCH_MAX=0.25, verbose=True)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# ── Algorithm modules: import everything from ongoing_parts, touch nothing ──
_ALGO = Path(__file__).resolve().parent.parent / 'ongoing_parts'
if str(_ALGO) not in sys.path:
    sys.path.insert(0, str(_ALGO))

from integrated_sim_3d import (
    # Classes
    BOCD, RLS, GPRModel,
    # MPC solvers
    run_mpc_3d_qp, run_mpc_3d, retract_velocity,
    # Algorithm dt (must match env.ctrl_dt = timestep × n_substeps)
    dt,
    # MPC weights / sizes
    W_TIME, W_ANGLE, MPC_N,
    # Speed / arc params
    V_MAX, V_MIN, V_RAMP,
    ARC_WARMUP_STEPS, V_WARMUP,
    R_TOL, R_REF_UPDATE,
    # Explore params
    V_EXPLORE, NOISE_CALIB_STEPS,
    # Settle params
    SETTLE_STEPS, SETTLE_SPEED, SETTLE_ACCEL,
    EXCITATION_FRAC, PRED_ERR_MULT, UNCERTAINTY_RATIO_THRESH,
    # Tangential exploration params
    TANGENTIAL_AMP_INIT, TANGENTIAL_AMP_STEP, TANGENTIAL_AMP_MAX,
    TANGENTIAL_SIGMA_TARGET, TANGENTIAL_SPEED,
    TANGENTIAL_RETRACT_SPEED, TANGENTIAL_RETRACT_TOL,
    # Arc sigma floor
    SIGMA_S_FLOOR,
)

from stage_1.mujoco_sim.elastic_env import ElasticTetherMJEnv   # same directory


# ────────────────────────────────────────────────────────────────────────────
def run_mujoco_sim(
    env: ElasticTetherMJEnv,
    arc_deg: float = 90,
    STRETCH_MAX: float = 0.25,
    f_max_safe: float = 3.0,
    w_time: float | None = None,
    verbose: bool = True,
    solver: str = 'qp',
    seed: int = 42,
) -> dict:
    """
    Run the v15 three-phase state machine with MuJoCo as physics backend.

    Parameters
    ----------
    env : ElasticTetherMJEnv
        Pre-configured (set_material called). Will be reset inside.
    arc_deg : float
        Target arc angle [degrees].
    STRETCH_MAX : float
        Clinical input: max allowed stretch [m]. Algorithm input.
    f_max_safe : float
        Clinical input: safety force [N]. Logging only — not in control.
    w_time : float | None
        MPC time cost weight. None → uses W_TIME from integrated_sim_3d.
    verbose : bool
        Print state transitions.
    solver : str
        'qp' (OSQP, default) or 'slsqp'.
    seed : int
        Random seed for injected sensor noise.

    Returns
    -------
    dict  — same schema as run_sim() in integrated_sim_3d.py, so
            run_experiment_matrix-style post-processing works unchanged.
    """
    np.random.seed(seed)
    if w_time is None:
        w_time = W_TIME
    arc_rad = np.radians(arc_deg)

    # ── Geometry (same as run_sim; not derived from L0_true) ─────────────
    anchor_est = env.anchor_pos.copy()            # measured before experiment
    R_init       = 0.18
    theta_start  = np.radians(-20)
    theta_target = theta_start + arc_rad
    ee_start = anchor_est + R_init * np.array(
        [np.cos(theta_start), np.sin(theta_start), 0.])

    # ── Initialise MuJoCo env ────────────────────────────────────────────
    ee_pos, force_norm_raw, force_vec, dist_obs = env.reset(ee_start=ee_start)

    p_cur  = ee_pos.copy()
    v_cur  = np.zeros(3)
    v_prev = np.zeros(3)

    # ── Algorithm modules (no parameters from env.L0_true / env.ks_true) ─
    bocd = BOCD()
    rls  = RLS()
    gpr  = GPRModel()

    # ── State machine variables (exact mirror of run_sim) ─────────────────
    phase = 0
    taut_step = None;  arc_step = None
    arc_steps = 0;     settle_steps_done = 0
    f_taut = 0.2;      sigma_s = 0.;  R_ref = None
    returning_to_target = False;  RETURN_TOL = 0.015
    dist_taut = None
    noise_samples: list[float] = []
    noise_std_est = 0.05

    # settle-phase vars (initialised on 0→1 transition)
    max_stretch_seen = 0.
    settle_dir = 1;  settle_speed_signed = 0.

    # tangential-phase vars (initialised on 1→1.5 transition)
    tangential_steps_done = 0
    tangential_theta_start = 0.
    tangential_amp = TANGENTIAL_AMP_INIT
    tangential_dir_to_target = 1.
    tangential_swing_dir = 1
    tangential_probe_stretch = 0.

    # arc-phase vars (initialised on 1.6→2 transition)
    theta_frozen = np.zeros(3)
    ks_frozen    = 1.0

    # ── History buffers ───────────────────────────────────────────────────
    T_sim = 800
    hp: list = [p_cur.copy()]
    hf: list = [];  hsig: list = [];  hks: list = []
    hgmu: list = [];  hgsig: list = []
    hphase: list = [];  h_dist_taut: list = [];  h_vmax: list = []
    h_rls_pred: list = [];  h_gpr_anchor: list = []

    # First observation with injected noise
    fm = max(0., force_norm_raw)

    if verbose:
        print("=" * 62)
        print("MuJoCo Simulation  —  Algorithm v15")
        print(f"  [MuJoCo env]  L0={env.L0_true:.3f}m  k={env.ks_true:.1f}N/m"
              "  ← UNKNOWN to algorithm")
        print(f"  [Clinical]    f_max_safe={f_max_safe}N  STRETCH_MAX={STRETCH_MAX}m")
        print(f"  θ_start={np.degrees(theta_start):.0f}°  "
              f"θ_target={np.degrees(theta_target):.0f}°")
        print("=" * 62)

    # ════════════════════════════════════════════════════════════════════
    # Main loop
    # ════════════════════════════════════════════════════════════════════
    for t in range(T_sim):
        hf.append(fm)
        sig = bocd.update(fm)
        hsig.append(sig)

        theta_cur = np.arctan2(p_cur[1] - anchor_est[1], p_cur[0] - anchor_est[0])
        dist_cur  = float(np.linalg.norm(p_cur - anchor_est))

        # ── Phase transitions (verbatim from run_sim) ─────────────────

        # 0 → 1 : BOCD detects taut
        if phase == 0 and sig > 0.35:
            phase = 1;  taut_step = t;  settle_steps_done = 0
            max_stretch_seen = 0.
            settle_dir = 1;  settle_speed_signed = 0.
            f_taut    = max(fm, 0.1)
            dist_taut = dist_cur
            if len(noise_samples) >= 5:
                noise_std_est = max(float(np.std(noise_samples)), 0.01)
            if verbose:
                print(f"  [BOCD]  t={t:3d}: taut → settle  "
                      f"sig={sig:.3f}  f_taut={f_taut:.3f}N  "
                      f"dist_taut={dist_taut:.3f}m  "
                      f"noise_std_est={noise_std_est:.3f}N")
                print(f"          dist constraint: "
                      f"[{dist_taut:.3f}, {dist_taut+STRETCH_MAX:.3f}] m")

        # 1 → 1.5 : settle convergence or timeout
        if phase == 1 and dist_taut is not None:
            st_s  = max(0., dist_cur - dist_taut)
            max_stretch_seen = max(max_stretch_seen, st_s)
            vs_s  = float(np.linalg.norm(v_cur))
            ac_s  = float(np.linalg.norm((v_cur - v_prev) / dt))
            phi_s = np.array([st_s, vs_s, ac_s])
            pred_err       = rls.pred_error(phi_s, fm)
            pred_err_thresh = PRED_ERR_MULT * noise_std_est
            unc_ratio      = rls.uncertainty_ratio()
            has_excitation = max_stretch_seen > EXCITATION_FRAC * STRETCH_MAX
            is_stable      = unc_ratio < UNCERTAINTY_RATIO_THRESH
            rls_converged  = has_excitation and (pred_err < pred_err_thresh) and is_stable
            settle_done    = settle_steps_done >= SETTLE_STEPS
            if rls_converged or settle_done:
                phase = 1.5
                tangential_steps_done   = 0
                tangential_theta_start  = theta_cur
                tangential_amp          = TANGENTIAL_AMP_INIT
                tangential_dir_to_target = 1. if theta_target >= theta_cur else -1.
                tangential_swing_dir    = 1
                tangential_probe_stretch = max(max_stretch_seen,
                                               EXCITATION_FRAC * STRETCH_MAX)
                reason = (
                    f"RLS conv(st={st_s:.3f}m err={pred_err:.3f}<{pred_err_thresh:.3f} "
                    f"P={unc_ratio:.3f})"
                    if rls_converged else
                    f"timeout({settle_steps_done} steps)"
                )
                if verbose:
                    print(f"  [Settle→Tangential] t={t:3d}: {reason}")
                    print(f"               ks={rls.theta[0]:.2f}  "
                          f"probe_stretch={tangential_probe_stretch:.4f}m")

        # 1.5 → 1.6 : tangential done (GPR sigma below target or amp capped)
        if phase == 1.5:
            tgt_theta = (tangential_theta_start
                         + tangential_dir_to_target * tangential_swing_dir * tangential_amp)
            _, sigma_bnd = gpr.predict(tangential_probe_stretch, 0., tgt_theta)
            sigma_ok  = sigma_bnd is not None and sigma_bnd < TANGENTIAL_SIGMA_TARGET
            amp_capped = tangential_amp >= TANGENTIAL_AMP_MAX
            near_start  = abs(theta_cur - tangential_theta_start) < 0.02
            if (sigma_ok or amp_capped) and near_start and tangential_steps_done > 5:
                phase = 1.6
                if verbose:
                    s_str = f"{sigma_bnd:.3f}" if sigma_bnd is not None else "N/A"
                    print(f"  [Tangential→Retract] t={t:3d}: "
                          f"amp={np.degrees(tangential_amp):.1f}°  σ={s_str}  "
                          f"(ok={sigma_ok} capped={amp_capped})")

        # 1.6 → 2 : retracted back to dist_taut
        if phase == 1.6 and (dist_cur - dist_taut) < TANGENTIAL_RETRACT_TOL:
            phase = 2;  arc_step = t
            returning_to_target = True
            rls.set_lam(rls.lam)
            theta_frozen = rls.theta.copy()
            ks_frozen    = max(theta_frozen[0], 0.5)
            gpr.freeze_base()
            f_target       = f_taut * 1.3
            stretch_target = np.clip(f_target / ks_frozen, 0., STRETCH_MAX * 0.9)
            R_ref = dist_taut + stretch_target

            # sigma_s 重置（同 Python sim）
            _st_init = max(0., dist_cur - dist_taut)
            _, _sr_init = gpr.predict(_st_init, 0., theta_cur)
            if _sr_init is None: _sr_init = 0.
            sigma_s = max(_sr_init, SIGMA_S_FLOOR)

            if verbose:
                print(f"  [Retract→Arc] t={t:3d}: "
                      f"ks_frozen={ks_frozen:.2f}  R_ref={R_ref:.3f}m  "
                      f"sigma_s_init={sigma_s:.3f}")
                print(f"               dist_taut={dist_taut:.3f}m  "
                      f"dist_upper={dist_taut+STRETCH_MAX:.3f}m  "
                      f"b={theta_frozen[1]:.4f}  m={theta_frozen[2]:.4f}  "
                      f"f_taut={f_taut:.3f}N")

        # Done?
        if phase == 2 and theta_cur >= theta_target - 0.03:
            if verbose:
                print(f"  [Done]  t={t:3d}: θ={np.degrees(theta_cur):.1f}° reached!")
            break

        # ── Action computation (verbatim logic from run_sim) ──────────

        if phase == 0:
            dn = (p_cur - anchor_est) / (dist_cur + 1e-6)
            u  = dn * V_EXPLORE
            if t < NOISE_CALIB_STEPS:
                noise_samples.append(fm)
            hks.append(rls.theta[0]);  hgmu.append(0.);  hgsig.append(0.)
            h_dist_taut.append(0.);    h_vmax.append(0.)
            h_rls_pred.append(float('nan'));  h_gpr_anchor.append(float('nan'))

        elif phase == 1:
            dn = (p_cur - anchor_est) / (dist_cur + 1e-6)
            settle_upper = dist_taut + STRETCH_MAX * 0.95
            settle_lower = dist_taut
            brake_dist   = settle_speed_signed**2 / (2 * SETTLE_ACCEL + 1e-9)
            if settle_dir > 0 and (settle_upper - dist_cur) < brake_dist + SETTLE_SPEED * dt:
                settle_dir = -1
            elif settle_dir < 0 and (dist_cur - settle_lower) < brake_dist + SETTLE_SPEED * dt:
                settle_dir = 1
            tgt_spd = settle_dir * SETTLE_SPEED
            d_spd   = np.clip(tgt_spd - settle_speed_signed,
                               -SETTLE_ACCEL * dt, SETTLE_ACCEL * dt)
            settle_speed_signed += d_spd
            if dist_cur >= settle_upper: settle_speed_signed = min(settle_speed_signed, 0.)
            if dist_cur <= settle_lower: settle_speed_signed = max(settle_speed_signed, 0.)
            u = dn * settle_speed_signed
            settle_steps_done += 1

            st = max(0., dist_cur - dist_taut)
            vs = float(np.linalg.norm(v_cur))
            ac = float(np.linalg.norm((v_cur - v_prev) / dt))
            phi = np.array([st, vs, ac])
            trls = rls.update(phi, fm, skip_if_no_stretch=True)
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0:
                gpr.fit()
            hks.append(trls[0]);  hgmu.append(0.);  hgsig.append(0.)
            h_dist_taut.append(dist_taut);  h_vmax.append(0.)
            h_rls_pred.append(float('nan'));  h_gpr_anchor.append(float('nan'))

        elif phase == 1.5:
            tgt_theta  = (tangential_theta_start
                          + tangential_dir_to_target * tangential_swing_dir * tangential_amp)
            tgt_dist   = dist_taut + tangential_probe_stretch
            tgt_pos    = anchor_est + tgt_dist * np.array(
                [np.cos(tgt_theta), np.sin(tgt_theta), 0.])
            err = tgt_pos - p_cur;  err_n = float(np.linalg.norm(err))
            u   = (err / (err_n + 1e-6)) * min(TANGENTIAL_SPEED, err_n / dt)
            tangential_steps_done += 1
            if err_n < 0.01:
                if tangential_swing_dir > 0:
                    tangential_swing_dir = -1
                else:
                    tangential_swing_dir = 1
                    tangential_amp = min(tangential_amp + TANGENTIAL_AMP_STEP,
                                         TANGENTIAL_AMP_MAX)
            st = max(0., dist_cur - dist_taut)
            vs = float(np.linalg.norm(v_cur))
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0:
                gpr.fit()
            hks.append(rls.theta[0]);  hgmu.append(0.);  hgsig.append(0.)
            h_dist_taut.append(dist_taut);  h_vmax.append(0.)
            h_rls_pred.append(float('nan'));  h_gpr_anchor.append(float('nan'))

        elif phase == 1.6:
            dn      = (p_cur - anchor_est) / (dist_cur + 1e-6)
            err_d   = dist_cur - dist_taut
            u       = -dn * min(TANGENTIAL_RETRACT_SPEED, max(err_d, 0.) / dt)
            st = max(0., dist_cur - dist_taut)
            vs = float(np.linalg.norm(v_cur))
            gpr.add_data(st, vs, theta_cur, fm)
            if len(gpr.X) >= 15 and t % 5 == 0:
                gpr.fit()
            hks.append(rls.theta[0]);  hgmu.append(0.);  hgsig.append(0.)
            h_dist_taut.append(dist_taut);  h_vmax.append(0.)
            h_rls_pred.append(float('nan'));  h_gpr_anchor.append(float('nan'))

        elif phase == 2:
            st = max(0., dist_cur - dist_taut)
            trls = theta_frozen

            # Returning-to-target sub-phase (fill gap to R_ref before MPC)
            if returning_to_target:
                gap = R_ref - dist_cur
                if abs(gap) < RETURN_TOL:
                    returning_to_target = False
                else:
                    dn_ret = (p_cur - anchor_est) / (dist_cur + 1e-6)
                    u = dn_ret * np.clip(gap / dt, -SETTLE_SPEED, SETTLE_SPEED)
                    hks.append(trls[0]);  hgmu.append(0.);  hgsig.append(sigma_s)
                    h_dist_taut.append(dist_taut);  h_vmax.append(0.)
                    h_rls_pred.append(float(trls[0] * st))
                    h_gpr_anchor.append(float('nan'))
                    hphase.append(phase)
                    v_prev = v_cur.copy();  v_cur = u.copy()
                    desired_pos = p_cur + dt * u
                    ee_pos, force_norm_raw, _, _ = env.step(desired_pos)
                    p_cur = ee_pos.copy()
                    fm = max(0., force_norm_raw)
                    hp.append(p_cur.copy())
                    continue

            # R_ref slow tracking
            if arc_steps % R_REF_UPDATE == 0:
                R_ref_new = 0.8 * R_ref + 0.2 * dist_cur
                R_ref = float(np.clip(R_ref_new,
                                       dist_taut + 0.05 * STRETCH_MAX,
                                       dist_taut + 0.6  * STRETCH_MAX))

            # GPR update
            vs = float(np.linalg.norm(v_cur))
            ac = float(np.linalg.norm((v_cur - v_prev) / dt))
            gpr.add_data(st, vs, theta_cur, fm)
            if t % 15 == 0:
                gpr.fit()
            mu_gpr, sr = gpr.predict(st, vs, theta_cur)
            if sr is None: sr = 0.
            GPR_SIGMA_THRESH = 0.3
            mu_val = (mu_gpr if (mu_gpr is not None and sr < GPR_SIGMA_THRESH)
                      else trls[0] * st)
            sigma_s = 0.2 * sr + 0.8 * sigma_s
            b_rls, m_rls = trls[1], trls[2]

            # Dynamic speed cap
            if arc_steps < ARC_WARMUP_STEPS:
                v_max_cur = V_WARMUP
            else:
                v_max_cur = min(V_MIN + (V_MAX - V_MIN)
                                 * ((arc_steps - ARC_WARMUP_STEPS) / V_RAMP), V_MAX)
            arc_steps += 1

            hks.append(trls[0]);  hgmu.append(float(mu_val));  hgsig.append(sigma_s)
            h_dist_taut.append(dist_taut);  h_vmax.append(v_max_cur)
            h_rls_pred.append(float(trls[0] * st))
            mu_anc, _ = gpr.gpr_force_anchor(st, vs, theta_cur)
            h_gpr_anchor.append(float(mu_anc) if mu_anc is not None else float('nan'))

            _mpc_fn = run_mpc_3d_qp if solver == 'qp' else run_mpc_3d
            u = _mpc_fn(
                p_cur, v_cur, trls[0], b_rls, m_rls,
                anchor_est, dist_taut, f_taut,
                theta_cur, theta_target,
                STRETCH_MAX, f_max_safe,
                gpr_model=gpr, gpr_sigma_thresh=GPR_SIGMA_THRESH,
                sigma_gpr=sigma_s, v_max_cur=v_max_cur,
                R_ref=R_ref, w_time=w_time,
            )

        # ── Physics step (replaces true_force_3d + pn = p_cur + dt*u) ──
        hphase.append(phase)
        v_prev = v_cur.copy();  v_cur = u.copy()
        desired_pos = p_cur + dt * u
        ee_pos, force_norm_raw, _, _ = env.step(desired_pos)
        p_cur = ee_pos.copy()
        fm = max(0., force_norm_raw)
        hp.append(p_cur.copy())

    # ════════════════════════════════════════════════════════════════════
    # Post-processing (mirrors run_sim return dict exactly)
    # ════════════════════════════════════════════════════════════════════
    hp   = np.array(hp)
    hf_a = np.array(hf)

    if arc_step is None:
        if verbose:
            print(f"\n[FAILED] Did not reach arc phase in {T_sim} steps.")
        return {
            'status': 'FAILED', 'arc_step': None, 'taut_step': taut_step,
            'hp': hp, 'hf': hf_a,
            'hsig': np.array(hsig), 'hks': np.array(hks),
            'hgmu': np.array(hgmu), 'hgsig': np.array(hgsig),
            'hphase': np.array(hphase),
            'h_dist_taut': np.array(h_dist_taut), 'h_vmax': np.array(h_vmax),
            'h_rls_pred': np.array(h_rls_pred), 'h_gpr_anchor': np.array(h_gpr_anchor),
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

    fc  = hf_a[arc_step:]
    rms = float(np.sqrt(np.mean((fc - f_taut) ** 2)))

    # viol: reference force = 90% STRETCH_MAX × env.ks_true (post-hoc only)
    f_static_max = 0.9 * STRETCH_MAX * env.ks_true
    viol         = float(np.mean(fc > f_static_max)) * 100

    hp_arc = hp[arc_step:]
    dists_arc = np.linalg.norm(hp_arc - anchor_est, axis=1)
    DIST_TOL        = 1e-3
    dist_upper_val  = dist_taut + STRETCH_MAX
    dist_viol_upper = float(np.mean(dists_arc > dist_upper_val + DIST_TOL)) * 100
    dist_viol_lower = float(np.mean(dists_arc < dist_taut    - DIST_TOL)) * 100

    pts2d    = hp[arc_step + ARC_WARMUP_STEPS:, :2]
    dists_f  = np.linalg.norm(pts2d - anchor_est[:2], axis=1)
    R_fit    = float(np.mean(dists_f))
    R_std    = float(np.std(dists_f))
    xc, yc   = float(anchor_est[0]), float(anchor_est[1])
    ks_final = float(theta_frozen[0])

    if verbose:
        print(f"\n[Metrics] RMS={rms:.3f}N  "
              f"viol(>{f_static_max:.2f}N)={viol:.1f}%")
        print(f"[Metrics] dist_upper_viol={dist_viol_upper:.1f}%  "
              f"dist_lower_viol={dist_viol_lower:.1f}%")
        print(f"[Metrics] total_steps={len(hf_a)}  arc_steps={len(fc)}")
        print(f"[Metrics] dist_taut={dist_taut:.3f}m  L0_true={env.L0_true:.3f}m  "
              f"err={abs(dist_taut - env.L0_true):.3f}m")
        print(f"[Radius]  R_mean={R_fit:.3f}m  R_std={R_std:.4f}m")

    return {
        'status': 'OK', 'arc_step': arc_step, 'taut_step': taut_step,
        'hp': hp, 'hf': hf_a,
        'hsig': np.array(hsig), 'hks': np.array(hks),
        'hgmu': np.array(hgmu), 'hgsig': np.array(hgsig),
        'hphase': np.array(hphase),
        'h_dist_taut': np.array(h_dist_taut), 'h_vmax': np.array(h_vmax),
        'h_rls_pred': np.array(h_rls_pred), 'h_gpr_anchor': np.array(h_gpr_anchor),
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


# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Quick single-run test.
    Run from mujoco_sim/:  python run_mujoco_sim.py
    """
    from stage_1.mujoco_sim.elastic_env import ElasticTetherMJEnv, MATERIAL_PRESETS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    XML = Path(__file__).parent / 'models' / 'scene.xml'

    # n_substeps=25: 25 × 0.002s = 0.05s = dt → one env.step() per algo step
    env = ElasticTetherMJEnv(XML, n_substeps=25)
    # 改后，对标 Python 仿真默认参数
    env.set_material(stiffness=30.0, L0=0.5, damping=0.5)

    print(f"Running: rubber_band  L0={env.L0_true:.3f}m  k={env.ks_true:.1f}N/m")
    result = run_mujoco_sim(env, arc_deg=90, STRETCH_MAX=0.25,
                             f_max_safe=3.0, verbose=True, seed=42)

    print(f"\nStatus: {result['status']}")
    if result['status'] == 'OK':
        print(f"  RMS={result['rms']:.3f}N  viol={result['viol']:.1f}%")
        print(f"  dist_upper_viol={result['dist_viol_upper']:.1f}%  "
              f"dist_lower_viol={result['dist_viol_lower']:.1f}%")
        print(f"  ks_estimated={result['ks_final']:.2f}  L0_true={env.L0_true:.3f}m")

        # Simple trajectory plot
        hp = result['hp'];  arc = result['arc_step']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(hp[:, 0], hp[:, 1], 'b-', lw=0.8, label='trajectory')
        if arc is not None:
            ax.plot(hp[arc:, 0], hp[arc:, 1], 'r-', lw=1.5, label='arc phase')
        ax.plot(0, 0, 'k+', ms=10, label='anchor')
        ax.set_aspect('equal');  ax.legend();  ax.set_title('EE trajectory (XY)')
        ax = axes[1]
        ax.plot(result['hf'], lw=0.8)
        if arc is not None:
            ax.axvline(arc, color='r', linestyle='--', label='arc start')
        if result['taut_step'] is not None:
            ax.axvline(result['taut_step'], color='g', linestyle='--', label='BOCD taut')
        ax.axhline(result['f_taut'], color='orange', linestyle=':', label='f_taut')
        ax.set_xlabel('step');  ax.set_ylabel('force (N)')
        ax.set_title('Force history');  ax.legend()
        plt.tight_layout()
        out = Path(__file__).parent / 'run_mujoco_result.png'
        plt.savefig(out, dpi=150)
        print(f"\nSaved: {out}")