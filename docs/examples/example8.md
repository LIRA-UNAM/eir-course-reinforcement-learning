# example8: K1 velocity, real steps, no self-collisions

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example8. This example adds **`air_time`** and **`self_collisions`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`air_time`** (weight +0.3): Reward every step a foot has been in the air for 0.05–0.5 s (a real swing phase, not shuffling).

```math
f = \mathbb{1}[s > 0.5]\sum_{f}\mathbb{1}\big[0.05 < t^{\text{air}}_f < 0.5\big]
```

**`self_collisions`** (weight −1.0): Count physics sub-steps with a self-contact above 10 N (arms on the trunk, legs rubbing).

```math
f = \sum_{h=1}^{4}\mathbb{1}\Big[\max_{\text{pairs}}\lVert \mathbf{F}^{\text{self}}_{h}\rVert > 10\ \text{N}\Big]
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (12 of 14)

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

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example8 --env.scene.num-envs 4096
uv run play example8 --checkpoint models/example8.pt   # course: pretrained policy, no W&B needed
uv run play example8 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example8 --checkpoint-file logs/rsl_rl/example8/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- `air_time` turns shuffling into steps with a swing phase of 0.05–0.5 s; `Metrics/air_time_mean` should rise. It is gated at s > 0.5, so slow commands are not forced to step.
- `self_collisions` stops the arms from bracing against the trunk; the K1 arms are part of the action space, so they could otherwise be used as a crutch.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example7](example7.md) · Next: [example9](example9.md)
