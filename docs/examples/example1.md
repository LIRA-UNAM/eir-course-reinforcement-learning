# example1: Furuta pendulum balance with tabular Q-learning

**Goal:** keep a rotary inverted (Furuta) pendulum balanced, starting within ±15° of upright, with a Q-table instead of a neural network.

**Lecture:** sections 2 (tabular methods) and 4 (value-based methods).

## The robot

- A horizontal arm (0.3 m) turns about the vertical axis and carries a 0.4 m pendulum on a hinge along the arm (`src/eir_course/asset_zoo/robots/furuta/`).
- The arm is driven by the **Booster K1 knee motor**: the same actuator config (`ACTUATOR_E6416`) that drives the K1's knees in example3–example10. That is a position servo with K_p = 80 N·m/rad, K_d = 4 N·m·s/rad, ±112 N·m and 0.095625 kg·m² of reflected rotor inertia.
- It is an mjlab task like every other example: MuJoCo Warp simulates 256 pendulums in parallel, physics at 200 Hz, the agent acting at 50 Hz (the same rates as the humanoid).

## The learning problem

| Element | Value |
| --- | --- |
| State | α (pendulum angle from upright), α̇, θ (arm angle), θ̇: the observation term `furuta_state` |
| Discretization | 9 × 9 × 5 × 5 = 2025 states over ±0.4 rad, ±4 rad/s, ±0.8 rad, ±8 rad/s |
| Actions | 7 arm position targets q* = s · a, a ∈ {−1, −⅔, …, 1}, s = 0.25 · 112 / 80 = 0.35 rad (the K1 action-scale rule) |
| Q-table | 2025 × 7 = 14 175 entries (55 KiB) |
| Reward | `alive` (+1) and `arm_angle` (−0.1 θ²), each × Δt = 0.02 s |
| Termination | `pendulum_fell` (\|α\| > 0.4 rad), `arm_out_of_range` (\|θ\| > 1.2 rad); time-out at 10 s |
| Learning | learning rate 0.1, γ = 0.99, ε-greedy decaying 1.0 → 0.05 over 80 % of 300 iterations |
| Experience | 256 envs × 24 steps = 6144 transitions per iteration, 1.8 M in total |

The algorithm is in `src/eir_course/tasks/furuta/runner.py`. For each step of all 256 pendulums:

```python
# next_value = max_a Q(s', a)
target = reward + gamma * next_value * (~terminated)
# mean over the envs that hit the same (s, a) in this step
q[s, a] += learning_rate * mean(target - q[s, a])
```

- A fall is terminal: it has no future value.
- A time-out is not terminal. mjlab resets a finished episode before returning its observation, so the runner bootstraps a time-out from the current state, as rsl_rl's PPO does (lecture section 3, point 4).
- Two pendulums landing in the same table cell in the same step share one update: the mean of their errors.

## Run it

```bash
uv run train example1                          # logs to the W&B project eir_course
WANDB_MODE=offline uv run train example1       # no W&B account needed
uv run play example1                           # pretrained Q-table from checkpoints/example1/
uv run play example1 --checkpoint-file logs/rsl_rl/example1/<run>/model_299.pt
uv run play example1 --wandb-run-path <entity>/eir_course/<run_id>
```

Training takes about 1 minute on a laptop GPU and about 5 minutes on a CPU. `play` opens the Viser viewer in the browser (`--viewer native` for a desktop window).

## What to look at in W&B

- `return/mean`, `episode_length_s/mean`, `success_rate`: an episode succeeds if it ends with the pendulum within 0.2 rad of upright for the last 2 s.
- `Episode_Reward/alive`, `Episode_Reward/arm_angle`: each reward term separately.
- `q_table/visited_fraction`: how much of the table the agent has ever touched.
- `q_table/value_slice` and `q_table/greedy_action_slice`: max_a Q and the greedy action over α (rows) × α̇ (columns), with the arm centred and still. Grey cells were never visited.
- `q_table/values`: histogram of the learned Q-values.
- `uv run train example1 --video` also records and uploads videos of training.

## Try this

Every Q-learning setting is a CLI flag (`uv run train example1 --help`). Tuples are written with commas.

- `--agent.bins 15,15,9,9` (18 225 states): the table is 9× larger. Does `visited_fraction` keep up in the same number of iterations?
- `--agent.bins 3,3,3,3`: too coarse to tell "falling left" from "upright". What happens to `success_rate`?
- `--env.actions.arm.scale 0.1`: the arm cannot move far enough to catch the pendulum.
- `--agent.epsilon-end 0.3`: keep exploring. Does the greedy policy still improve?

Then move on to [example2](example2.md), which keeps everything the same but starts with the pendulum hanging down.
