# example1: Furuta pendulum balance with tabular Q-learning

**Goal:** keep a rotary inverted (Furuta) pendulum balanced, starting within ±15° of upright, with a Q-table.

**Lecture:** sections 2 (tabular methods) and 4 (value-based methods).

## The robot

- A horizontal arm (0.3 m) turns about the vertical axis and carries a 0.4 m pendulum on a hinge along the arm (`src/eir_course/asset_zoo/robots/furuta/furuta.xml`).
- The arm is driven by the **Booster K1 knee motor**: a position servo with K_p = 80 N·m/rad, K_d = 4 N·m·s/rad, ±112 N·m and 0.095625 kg·m² of reflected rotor inertia. This is the same model, and the same numbers, that drive the K1's knees in example3–example10.
- Physics runs at 200 Hz and the agent acts at 50 Hz, the same rates as the humanoid.

## The learning problem

| Element | Value |
| --- | --- |
| State | α (pendulum angle from upright), α̇, θ (arm angle), θ̇ |
| Discretization | 9 × 9 × 5 × 5 = 2025 states over ±0.4 rad, ±4 rad/s, ±0.8 rad, ±8 rad/s |
| Actions | 7 arm position targets q* = s · a, a ∈ {−1, −⅔, …, 1}, s = 0.25 · 112 / 80 = 0.35 rad (the K1 action-scale rule) |
| Q-table | 2025 × 7 = 14 175 entries (55 KiB) |
| Reward | 1 − 0.1 θ² per step (stay alive, keep the arm centred) |
| Termination | \|α\| > 0.4 rad or \|θ\| > 1.2 rad; time-out at 10 s |
| Learning | α_lr = 0.1, γ = 0.99, ε-greedy decaying 1.0 → 0.05 over 80 % of 5000 episodes |

The whole algorithm is the update in `src/eir_course/tasks/furuta/q_learning.py`:

```python
target = r if terminated else r + gamma * q[s_next].max()
q[s, a] += learning_rate * (target - q[s, a])
```

A fall is terminal (no future), a time-out is not: the target still bootstraps from `q[s_next]` (lecture section 3, point 4).

## Run it

Training runs on any CPU and takes a few minutes.

```bash
uv run train example1                   # logs to W&B project eir_course
uv run train example1 --no-wandb        # offline: only writes logs/qlearning/example1/<time>/q_table.npz
uv run play example1                    # pretrained Q-table from checkpoints/example1/
uv run play example1 --checkpoint-file logs/qlearning/example1/<time>/q_table.npz
uv run play example1 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example1 --video            # writes videos/example1.mp4 instead of opening the viewer
```

## What to look at in W&B

- `return/mean`, `episode_length_s/mean`, `success_rate`: an episode succeeds if it ends with the pendulum within 0.2 rad of upright for the last 2 s.
- `q_table/visited_fraction`: how much of the table the agent has ever touched.
- `q_table/value_slice` and `q_table/greedy_action_slice`: max_a Q and the greedy action over α (rows) × α̇ (columns), with the arm centred and still. Grey cells were never visited.
- `greedy_rollout`: a video of the final greedy policy.

## Try this

- `--bins 15 15 9 9` (≈ 18 000 states): the table is 9× larger. Does `visited_fraction` keep up in the same number of episodes?
- `--bins 3 3 3 3`: too coarse to tell "falling left" from "upright". What happens to `success_rate`?
- `--action-scale 0.1`: the arm cannot move far enough to catch the pendulum.

Then move on to [example2](example2.md), which keeps everything the same but starts with the pendulum hanging down.
