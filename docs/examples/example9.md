# example9: K1 velocity, a quiet trunk and arm swing (the full reward)

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example9. This example adds **`body_ang_vel`** and **`angular_momentum`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`body_ang_vel`** (weight −0.06): Penalize roll/pitch angular velocity of the trunk (wobble).

```math
f = \lVert \boldsymbol{\omega}^{\text{trunk}}_{xy}\rVert^2
```

**`angular_momentum`** (weight −0.02): Penalize whole-body angular momentum; counter-swinging arms emerge to cancel the legs.

```math
f = \lVert \mathbf{L}\rVert^2
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (14 of 14)

| Term | Weight | Added in |
| --- | --- | --- |
| `track_linear_velocity` | +2.0 | example3 |
| `track_angular_velocity` | +2.0 | example3 |
| `upright` | +1.0 | example4 |
| `pose` | +1.0 | example4 |
| `action_rate_l2` | −0.1 | example5 |
| `dof_pos_limits` | −1.0 | example5 |
| `foot_clearance` | −2.0 | example6 |
| `foot_swing_height` | −0.25 | example6 |
| `foot_slip` | −0.1 | example7 |
| `soft_landing` | −10⁻⁵ | example7 |
| `air_time` | +0.3 | example8 |
| `self_collisions` | −1.0 | example8 |
| `body_ang_vel` | −0.06 | example9 |
| `angular_momentum` | −0.02 | example9 |

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example9 --env.scene.num-envs 4096
uv run play example9 --checkpoint models/example9.pt   # course: pretrained policy, no W&B needed
uv run play example9 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example9 --checkpoint-file logs/rsl_rl/example9/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- This is the complete K1 reward on flat ground, without randomization. `body_ang_vel` damps trunk wobble; `angular_momentum` makes the arms swing against the legs (`Metrics/angular_momentum_mean`).
- Compare with example10, which adds rough terrain, domain randomization, pushes and a terrain curriculum with the same reward: the gap between the two is sim-to-real robustness, not reward design.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example8](example8.md) · Next: [example10](example10.md)
