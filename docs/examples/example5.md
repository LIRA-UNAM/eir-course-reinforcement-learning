# example5: K1 velocity, smooth actions, respect joint limits

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example5. This example adds **`action_rate_l2`** and **`dof_pos_limits`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`action_rate_l2`** (weight −0.1): Penalize jumps between consecutive actions (smoother torques on hardware).

```math
f = \lVert \mathbf{a}_t - \mathbf{a}_{t-1}\rVert^2
```

**`dof_pos_limits`** (weight −1.0): Penalize joints past 90 % of their range (stay off the mechanical stops).

```math
f = \sum_j \big[\max(0, q^{\min}_{j,\text{soft}} - q_j) + \max(0, q_j - q^{\max}_{j,\text{soft}})\big]
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (6 of 14)

| Term | Weight | Added in |
| --- | --- | --- |
| `track_linear_velocity` | +2.0 | example3 |
| `track_angular_velocity` | +2.0 | example3 |
| `upright` | +1.0 | example4 |
| `pose` | +1.0 | example4 |
| `action_rate_l2` | −0.1 | example5 |
| `dof_pos_limits` | −1.0 | example5 |

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example5 --env.scene.num-envs 4096
uv run play example5                                      # pretrained policy from checkpoints/example5/
uv run play example5 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example5 --checkpoint-file logs/rsl_rl/example5/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- `action_rate_l2` is the first penalty: look at `Episode_Metrics/mean_action_acc` and at the video for less high-frequency jitter.
- `dof_pos_limits` should keep joints off their stops; arm and ankle-roll ranges on the K1 are narrow.
- Tracking may get slightly worse: every penalty trades a little task reward for behaviour we prefer.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example4](example4.md) · Next: [example6](example6.md)
