# example3: K1 velocity, velocity tracking only

**Goal:** train the Booster K1 to follow velocity commands on flat ground with the reward terms of example3–example3. This example adds **`track_linear_velocity`** and **`track_angular_velocity`**.

**Lecture:** section 8 (the reward in detail) and section 3 (reward design and the MDP).

## What is new

**`track_linear_velocity`** (weight +2.0): Follow the commanded planar velocity; vertical bouncing (v_z) also lowers it.

```math
f = \exp\!\left(-\frac{\lVert \mathbf{v}^{*}_{xy} - \mathbf{v}_{xy}\rVert^2 + v_z^2}{0.25}\right)
```

**`track_angular_velocity`** (weight +2.0): Follow the commanded yaw rate; roll/pitch rocking (ω_xy) also lowers it.

```math
f = \exp\!\left(-\frac{(\omega^{*}_z - \omega_z)^2 + \lVert \boldsymbol{\omega}_{xy}\rVert^2}{0.5}\right)
```

Notation: s = ‖v*_xy‖ + |ω*_z| is the command magnitude, f runs over the feet, 1[·] is the indicator. Every term is multiplied by its weight and Δt = 0.02 s.

## Active reward terms (2 of 14)

| Term | Weight | Added in |
| --- | --- | --- |
| `track_linear_velocity` | +2.0 | example3 |
| `track_angular_velocity` | +2.0 | example3 |

## The environment (identical in example3–example9)

- Booster K1, 20 actuated joints (legs + arms; the head holds its pose). The actor sees 69 values (gyro, gravity, joint positions/velocities, last action, command), the critic 84 (adds base linear velocity, foot heights, air times, contacts, contact forces).
- Flat ground, **no domain randomization and no pushes**. Commands come from fixed ranges, v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s, with no speed curriculum. Only the reward changes between these examples.
- Terminations: trunk tilt > 63° (stochastic) or any non-foot body touching the ground; time-out at 20 s.
- PPO, 4096 environments, 10 000 iterations (`src/eir_course/tasks/velocity/config/k1/`).

## Run it

Training needs a Linux machine with an NVIDIA GPU. Playing works on any machine.

```bash
uv run train example3 --env.scene.num-envs 4096
uv run play example3 --checkpoint models/example3.pt   # course: pretrained policy, no W&B needed
uv run play example3 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example3 --checkpoint-file logs/rsl_rl/example3/<run>/model_9999.pt
```

`play` opens a Viser viewer in the browser with a joystick to send velocity commands (`--viewer native` for a desktop window).

## What to watch for (hypotheses to check against your run)

- The only signal is "move at the commanded velocity". Nothing says *how*: expect odd gaits that still score, e.g. shuffling, hopping, leaning, flailing arms, or crawling-like motions just above the termination limits.
- Episodes still end on falls (`fell_over`, `illegal_contact`): falling loses all future tracking reward, which is the only reason to stay up.
- Compare `Episode_Reward/track_linear_velocity` with the video: high tracking reward does not mean a usable gait. This is reward hacking in its purest form.

In W&B, the per-term curves `Episode_Reward/<term>` show what the policy is optimizing; compare them with the previous example's run.

Previous: [example2](example2.md) · Next: [example4](example4.md)
