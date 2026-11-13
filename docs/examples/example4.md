# example4: K1 velocity, stay upright, keep a natural pose

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example4. This example adds **`upright`** and **`pose`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`upright`** (weight +1.0): Keep the trunk vertical; the reward drops to 0.37 at a 26.6° tilt.

```math
f = \exp\!\left(-\frac{\lVert \mathbf{g}^{b}_{xy}\rVert^2}{0.2}\right)
```

**`pose`** (weight +1.0): Stay close to the default pose, with per-joint tolerances that loosen with command speed (standing 0.05 rad).

```math
f = \exp\!\left(-\frac{1}{J}\sum_{j} \frac{(q_j - \bar{q}_j)^2}{\sigma_j(s)^2}\right)
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (4 of 14)

| Term | Weight | Added in |
| --- | --- | --- |
| `track_linear_velocity` | +2.0 | example3 |
| `track_angular_velocity` | +2.0 | example3 |
| `upright` | +1.0 | example4 |
| `pose` | +1.0 | example4 |

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example4 --env.scene.num-envs 4096
uv run play example4 --checkpoint models/example4.pt   # course: pretrained policy, no W&B needed
uv run play example4 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example4 --checkpoint-file logs/rsl_rl/example4/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- `upright` and `pose` are positive kernels like the tracking terms (total positive weight 6). Expect a more human-like posture and fewer falls, but feet may still shuffle and actions may still jitter.
- When standing (zero command), `pose` uses σ = 0.05 rad: the robot should return to its home pose instead of drifting.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example3](example3.md) · Next: [example5](example5.md)
