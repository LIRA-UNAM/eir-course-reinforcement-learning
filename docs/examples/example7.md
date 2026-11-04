# example7: K1 velocity, no slipping, soft landings

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example7. This example adds **`foot_slip`** and **`soft_landing`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`foot_slip`** (weight −0.1): A foot in contact must not slide.

```math
f = \mathbb{1}[s > 0.05]\sum_{f}\mathbb{1}[\text{contact}_f]\,\lVert \mathbf{v}^{\text{foot}}_{f,xy}\rVert^2
```

**`soft_landing`** (weight −10⁻⁵): Penalize the contact force at touchdown (in newtons, hence the tiny weight).

```math
f = \mathbb{1}[s > 0.05]\sum_{f}\mathbb{1}[\text{touchdown}_f]\,\lVert \mathbf{F}_f\rVert
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (10 of 14)

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

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example7 --env.scene.num-envs 4096
uv run play example7                                      # pretrained policy from checkpoints/example7/
uv run play example7 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example7 --checkpoint-file logs/rsl_rl/example7/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- `Metrics/slip_velocity_mean` should drop: stance feet stop skating.
- `Metrics/landing_force_mean` shows the touchdown impact the robot is trading off; with weight −10⁻⁵ the effect is gentle.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example6](example6.md) · Next: [example8](example8.md)
