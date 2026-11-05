# example2: Furuta pendulum swing-up and balance with tabular Q-learning

**Goal:** start with the pendulum hanging down at rest, swing it up, and keep it balanced, using the same robot, motor and algorithm as [example1](example1.md).

**Lecture:** section 2: how fast a table grows when the task needs more of the state space.

## What changes from example1

| | example1: balance | example2: swing-up + balance |
| --- | --- | --- |
| Start | within ±15° of upright, at rest | hanging down (α = π), at rest |
| α bins | 9 over ±0.4 rad | 21 over the full circle [−π, π], **denser near upright** |
| α̇, θ, θ̇ bins | 9, 5, 5 | 15, 7, 7 |
| States × actions | 2025 × 7 = 14 175 | 15 435 × 7 = **108 045** (7.6×) |
| Action scale | 0.35 rad (K1 rule) | 1.0 rad: pumping energy into the pendulum needs larger arm swings |
| Reward | `alive` (+1), `arm_angle` (−0.1 θ²) | `pendulum_upright` ((1 + cos α)/2), `arm_angle` (−0.1 θ²), `arm_velocity` (−0.001 θ̇²) |
| Termination | \|α\| > 0.4 rad, \|θ\| > 1.2 rad, 10 s | \|θ\| > 1.5 rad, 15 s (falling is allowed: swinging *is* falling) |
| Iterations | 300 (1.8 M transitions) | 3000 (18.4 M transitions) |

Two design choices are worth discussing in class:

- **Non-uniform bins.** α bins are spaced as x ∝ u² (`sqrt_spacing` in `src/eir_course/tasks/furuta/rl_cfg.py`): wide where the pendulum swings through quickly, narrow near upright where balancing needs resolution. Hand-designing the discretization is exactly the kind of engineering a neural network removes (lecture section 2).
- **A non-negative reward.** The swing-up reward is in [0, 1] when the arm is centred: 0 hanging, 1 upright. If it were cos α (negative while hanging), ending the episode early by hitting the arm limit would *increase* the return, and the agent would learn to crash on purpose (lecture section 3, "no explicit alive bonus").

## Run it

```bash
uv run train example2
uv run train example2 --agent.max-iterations 6000     # more experience for the bigger table
uv run play example2                                   # pretrained Q-table from checkpoints/example2/
uv run play example2 --wandb-run-path <entity>/eir_course/<run_id>
```

Training takes about 6 minutes on a laptop GPU but about 40 minutes on a CPU: without an NVIDIA GPU, start it before a break or play the pretrained policy.

## What to look at in W&B

Compare the example1 and example2 runs side by side in the `eir_course` project:

- `q_table/visited_fraction` against `env_steps`: example2 needs far more steps to cover its table.
- `success_rate`: swing-up is learned much later than balance, and less reliably.
- The run summary: `num_states`, `q_entries`, `q_table_kib`.
- `q_table/value_slice`: the value now has to "see" the whole circle of α, not only the region near upright.

## What to watch for (hypotheses to check with your run)

- Tabular swing-up is hard: the policy may pump energy but fail to catch the pendulum, or balance only from some swing directions. If it does not converge, the knobs in order are: finer α bins near upright, more iterations (`--agent.max-iterations`), then the action scale (`--env.actions.arm.scale`).
- This is the point of the example: adding one capability (swing-up) multiplies the table and the experience needed. A humanoid with 20 joints is out of reach for a table (lecture section 2), which is why example3 onwards use a neural network.
