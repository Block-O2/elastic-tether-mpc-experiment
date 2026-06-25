# Robotic Arc Traction Control on Unknown Elastic Objects

This repository implements a four-layer algorithm stack for robotic arc traction control on unknown elastic objects such as elastic bands, silicone, and fabric. The robot end-effector follows a 3D arc trajectory around an unknown anchor point while maintaining a just-taut critical tension, without using true material parameters such as stiffness, natural length, anchor position, or arc radius. The clinical motivation is upper-limb rehabilitation, where the robot acts as a therapist that applies calibrated elastic resistance during guided limb movement.

## Method

1. **Perception**

   Bayesian Online Changepoint Detection (BOCD) detects the slack-to-taut transition from streaming force and displacement observations. Recursive Least Squares (RLS) then estimates local stiffness and damping coefficients online, providing a lightweight parametric model without access to ground-truth material properties.

2. **Modeling**

   Gaussian Process Regression (GPR) models the force-displacement relation and provides posterior uncertainty over the learned elastic response. This nonparametric layer is used when sufficient online data are available and supports unknown nonlinear, hysteretic, and piecewise material behavior.

3. **Control**

   The controller uses Tube-MPC / QP-based MPC to track the desired traction force while respecting distance safety constraints. The GPR posterior variance dynamically tightens the distance constraint, making the controller more conservative in regions where the learned model is uncertain.

4. **Fitting**

   The fitting layer estimates the anchor-point geometry directly from end-effector motion and force response. This enables arc radius tracking without knowing the true anchor position or true arc radius.

## Simulation Results

![Material 1 linear spring simulation](ongoing_parts/integrated_sim_3d_v15_1.gif)

Material 1: Linear spring (ks=30). Trajectory, force, distance constraint, and GPR/RLS estimates over time.

![Material 2 hardening spring simulation](ongoing_parts/integrated_sim_3d_v15_2.gif)

Material 2: Hardening spring. Stiffness increases with stretch.

![Material 3 hysteresis spring simulation](ongoing_parts/integrated_sim_3d_v15_3.gif)

Material 3: Hysteresis spring. Loading and unloading follow different force curves.

![Material 4 piecewise anisotropic spring simulation](ongoing_parts/integrated_sim_3d_v15_4.gif)

Material 4: Piecewise anisotropic spring. Direction-dependent stiffness; known edge case for the current architecture.

![Material 5 soft linear spring simulation](ongoing_parts/integrated_sim_3d_v15_5.gif)

Material 5: Soft linear spring (ks=10). Low SNR operating condition.

The Python simulation campaign covers 4 material types across 3 arc angles (30, 60, and 90 degrees), for 24 total experiments. The controller achieved zero safety violations (`dist_viol = 0%`) across all 24 experiments, with GPR usage rates from 65% to 100% across materials. Force RMS tracking error ranged from 0.39 N to 0.59 N.

## MuJoCo Migration

The method was migrated from Python simulation to the MuJoCo physics simulator to evaluate sim-to-sim transfer under a more realistic dynamics backend. A key migration issue was that `data.ten_velocity` returned zero for mocap bodies, which corrupted damping and radius estimates. This was fixed by estimating mocap body velocity from end-effector displacement, using `||Delta ee_pos|| / ctrl_dt`. After the fix, arc radius stability improved substantially, with `R_std` decreasing from 0.026 m to 0.009 m.

## Target Venue

IEEE Robotics and Automation Letters (RA-L)
