"""
elastic_env.py  ——  MuJoCo environment for elastic tether experiments.

Architecture
------------
  EE position is commanded via MuJoCo mocap body (data.mocap_pos[0]).
  The EE body is declared directly as mocap="true" in the MJCF — no weld
  constraint needed. Elastic force is computed analytically from tendon
  length, avoiding MuJoCo API ambiguity around ten_force (actuator force,
  not passive spring).

  Force velocity term: MuJoCo's data.ten_velocity is always 0 for mocap
  bodies (position set instantaneously). We compute ee_speed = ||Δpos||/dt
  and use it in f = k*stretch + d*ee_speed, matching the Python sim's
  force model f = ks*stretch + b*||vel_EE||.

Interface
---------
  obs = env.reset()                     → (ee_pos, force_norm, force_vec, dist)
  obs = env.step(ee_pos_desired)        → same
  env.set_material(stiffness, L0, d)    → override tendon params before reset
  env.set_material_preset('rubber_band')

The algorithm stack in ongoing_parts/ sees only (dist, force_norm) from get_obs().
It must never access env.L0_true or env.ks_true.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# Material presets  (true physical params — UNKNOWN to algorithm layer)
# ---------------------------------------------------------------------------
MATERIAL_PRESETS: dict[str, dict] = {
    'rubber_band':    {'stiffness': 80.0,  'L0': 0.20, 'damping': 0.5},
    'silicone':       {'stiffness': 40.0,  'L0': 0.25, 'damping': 0.3},
    'cotton_rope':    {'stiffness': 120.0, 'L0': 0.28, 'damping': 0.8},
    'elastic_fabric': {'stiffness': 25.0,  'L0': 0.22, 'damping': 0.2},
}


class ElasticTetherMJEnv:
    """MuJoCo environment wrapping the elastic tether scene.

    Parameters
    ----------
    xml_path : str or Path
        Path to scene.xml.
    n_substeps : int
        Physics steps per control step. ctrl_dt = timestep * n_substeps.
        Default 5 → 0.002 × 5 = 0.010 s control period (100 Hz).
    sensor_noise_std : float
        F/T sensor noise standard deviation [N].
        Applied as 3D vector (independent on each axis) to force vector before taking magnitude.
        Default 0.1 N matches industrial F/T sensor specs.
    """

    def __init__(self, xml_path: str | Path, n_substeps: int = 5, sensor_noise_std: float = 0.1) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)
        self.n_substeps = n_substeps
        self.ctrl_dt = self.model.opt.timestep * n_substeps
        self.sensor_noise_std = sensor_noise_std

        # ---- Cache MuJoCo IDs ----
        def bid(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._ee_body_id   = bid('ee')
        self._tendon_id    = 0   # 'elastic_tether' is the only tendon
        self._mocap_id     = 0   # 'ee_target' is the only mocap body

        # Anchor world position (fixed throughout simulation)
        mujoco.mj_forward(self.model, self.data)
        self.anchor_pos: np.ndarray = self.data.body('anchor').xpos.copy()

        # EE position tracking for velocity estimation.
        # MuJoCo's data.ten_velocity is 0 for mocap bodies, and even with
        # analytical tendon velocity the SIGN differs from what the algorithm
        # expects (RLS uses ||vel_EE|| which is always ≥ 0, but tendon
        # velocity is signed: positive when stretching, negative when
        # shortening — these cancel in RLS, giving b≈0).
        # We track EE position and use ||vel_EE|| to match the Python sim's
        # force model: f = k*stretch + d*||vel_EE||.
        self._prev_ee_pos: np.ndarray = self.data.body('ee').xpos.copy()

        self.time: float = 0.0

    # -----------------------------------------------------------------------
    # Material configuration
    # -----------------------------------------------------------------------
    def set_material(
        self,
        stiffness: float,
        L0: float,
        damping: float = 0.5,
    ) -> None:
        """Set true tendon parameters. Call before reset() for each experiment.

        These are the ground-truth values — the algorithm MUST NOT read them.
        """
        tid = self._tendon_id
        self.model.tendon_stiffness[tid]         = stiffness
        self.model.tendon_damping[tid]           = damping
        self.model.tendon_lengthspring[tid, 0]   = 0.0   # min of neutral zone
        self.model.tendon_lengthspring[tid, 1]   = L0    # max of neutral zone = natural length

    def set_material_preset(self, name: str) -> None:
        """Convenience wrapper for MATERIAL_PRESETS dict."""
        if name not in MATERIAL_PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(MATERIAL_PRESETS)}")
        p = MATERIAL_PRESETS[name]
        self.set_material(p['stiffness'], p['L0'], p['damping'])

    # -----------------------------------------------------------------------
    # Core env interface
    # -----------------------------------------------------------------------
    def reset(self, ee_start: np.ndarray | None = None) -> tuple:
        """Reset simulation state.

        Parameters
        ----------
        ee_start : array(3,), optional
            Initial EE world position. Defaults to 0.5 × L0 on the x-axis
            (comfortably inside the slack region).

        Returns
        -------
        obs : tuple  — see get_obs()
        """
        mujoco.mj_resetData(self.model, self.data)
        self.time = 0.0

        if ee_start is None:
            L0 = float(self.model.tendon_lengthspring[self._tendon_id, 1])
            ee_start = np.array([L0 * 0.5, 0.0, 0.0])

        # Align mocap target with EE start
        self.data.mocap_pos[self._mocap_id]  = ee_start.copy()
        self.data.mocap_quat[self._mocap_id] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        self._prev_ee_pos = self.data.body('ee').xpos.copy()
        return self.get_obs()

    def step(self, ee_pos_desired: np.ndarray) -> tuple:
        """Command EE to desired position and advance physics by n_substeps.

        Parameters
        ----------
        ee_pos_desired : array(3,)
            Desired EE world position (output of MPC).

        Returns
        -------
        obs : tuple  — see get_obs()
        """
        # Save EE position BEFORE moving, for velocity estimation
        self._prev_ee_pos = self.data.body('ee').xpos.copy()

        self.data.mocap_pos[self._mocap_id] = ee_pos_desired
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.time += self.ctrl_dt
        return self.get_obs()

    def get_obs(self) -> tuple:
        """Read current state from MuJoCo.

        Returns
        -------
        ee_pos     : np.ndarray(3,)  — EE world position [m]
        force_norm : float            — elastic force magnitude [N]
        force_vec  : np.ndarray(3,)  — elastic force on EE (points toward anchor) [N]
        dist       : float            — EE-to-anchor distance [m]

        Noise model (F/T sensor):
          Force noise is a 3D vector (independent on each axis), added before taking magnitude.
          This matches industrial F/T sensor behavior and Python sim physics model.
        """
        ee_pos     = self.data.body('ee').xpos.copy()
        dist       = float(np.linalg.norm(ee_pos - self.anchor_pos))

        force_norm = self._compute_tendon_force()

        if dist > 1e-6:
            toward_anchor = (self.anchor_pos - ee_pos) / dist
        else:
            toward_anchor = np.zeros(3)

        # Calculate force vector (no noise yet)
        force_vec = force_norm * toward_anchor

        # F/T sensor noise: 3D vector, independent on each axis
        if self.sensor_noise_std > 0:
            noise_3d = np.random.normal(0, self.sensor_noise_std, 3)
            force_vec = force_vec + noise_3d
            # Recalculate magnitude and enforce non-negativity (rubber band cannot pull backward)
            force_norm = max(0.0, np.linalg.norm(force_vec))
        
        return ee_pos, force_norm, force_vec, dist

    # -----------------------------------------------------------------------
    # Ground-truth properties (for logging/plotting — never pass to algorithm)
    # -----------------------------------------------------------------------
    @property
    def L0_true(self) -> float:
        """True natural length [m]. Ground truth — algorithm must NOT use this."""
        return float(self.model.tendon_lengthspring[self._tendon_id, 1])

    @property
    def ks_true(self) -> float:
        """True stiffness [N/m]. Ground truth — algorithm must NOT use this."""
        return float(self.model.tendon_stiffness[self._tendon_id])

    @property
    def sim_time(self) -> float:
        """Simulation wall-clock time [s]."""
        return float(self.data.time)

    # -----------------------------------------------------------------------
    # Optional: passive viewer (call from main thread only)
    # -----------------------------------------------------------------------
    def launch_viewer(self) -> None:
        """Open MuJoCo passive viewer (blocks until closed)."""
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                mujoco.mj_step(self.model, self.data)
                v.sync()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _compute_tendon_force(self) -> float:
        """Analytical spring force [N]. Uses tendon length from MuJoCo state.

        For springlength="0 L0":
          stretch = ten_length - L0
          force   = k * stretch + d * ee_speed   if stretch > 0
                  = 0                              otherwise
        Clamped to ≥ 0 (rubber band cannot push).

        ee_speed = ||vel_EE|| is the EE velocity norm (always ≥ 0), matching
        the Python sim's force model where b * ||vel|| captures velocity-
        dependent resistance. MuJoCo's data.ten_velocity is unusable (always
        0 for mocap bodies), and signed tendon velocity causes RLS to learn
        b≈0 because positive/negative contributions cancel out.
        """
        tid = self._tendon_id
        length = float(self.data.ten_length[tid])
        L0 = float(self.model.tendon_lengthspring[tid, 1])
        k  = float(self.model.tendon_stiffness[tid])
        d  = float(self.model.tendon_damping[tid])

        # EE speed (norm, always ≥ 0) — matches Python sim's b * ||vel||
        ee_pos = self.data.body('ee').xpos
        ee_speed = float(np.linalg.norm(ee_pos - self._prev_ee_pos)) / self.ctrl_dt

        stretch = length - L0

        if stretch <= 0.0:
            return 0.0
        return max(0.0, k * stretch + d * ee_speed)