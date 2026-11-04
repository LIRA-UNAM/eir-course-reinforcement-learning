# example6: K1 velocity, lift the feet

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example6. This example adds **`foot_clearance`** and **`foot_swing_height`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`foot_clearance`** (weight −2.0): A moving foot should travel 10 cm above the ground; a stance foot (v ≈ 0) is not penalized.

```math
f = \mathbb{1}[s > 0.05]\sum_{f}\big|h_f - 0.1\big|\cdot \lVert \mathbf{v}^{\text{foot}}_{f,xy}\rVert
```

**`foot_swing_height`** (weight −0.25): At each touchdown, compare the step's peak foot height to 10 cm.

```math
f = \mathbb{1}[s > 0.05]\sum_{f}\mathbb{1}[\text{touchdown}_f]\left(\frac{h^{\text{peak}}_f}{0.1} - 1\right)^2
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (8 of 14)

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

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example6 --env.scene.num-envs 4096
uv run play example6                                      # pretrained policy from checkpoints/example6/
uv run play example6 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example6 --checkpoint-file logs/rsl_rl/example6/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- Before this step nothing asks the feet to leave the ground. Expect visible foot lift (~10 cm) while walking, and less tripping.
- `Metrics/peak_height_mean` (logged by `foot_swing_height`) should move towards 0.1 m.
- Both terms are gated by the command (s > 0.05): standing still should not start stepping in place.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example5](example5.md) · Next: [example7](example7.md)
