# EIR course: reinforcement learning for robots

Official repository for the EIR course on reinforcement learning. Over ten hands-on examples you go from a pendulum learned with a lookup table to a humanoid that walks on rough terrain:

1. **example1–example2:** a Furuta (rotary inverted) pendulum learned with **tabular Q-learning**: first balancing, then swinging up from rest. The table grows quickly, which shows why tables stop scaling.
2. **example3–example9:** the **Booster K1 humanoid** learning to follow velocity commands with **PPO** and a neural network. Each example adds **two reward terms**, so you can see what each term does to the gait.
3. **example10:** the full K1 task as it is trained for the real robot: rough terrain, domain randomization, pushes and a terrain curriculum.

The lecture notes are in [docs/lecture.md](docs/lecture.md), and each example has a guide in [docs/examples/](docs/examples/).

| Example | Task | Method | New in this example | Train on |
| --- | --- | --- | --- | --- |
| [example1](docs/examples/example1.md) | Furuta pendulum: balance | Tabular Q-learning | 2025-state table, 7 position targets | GPU ~1 min, CPU ~5 min |
| [example2](docs/examples/example2.md) | Furuta pendulum: swing-up + balance | Tabular Q-learning | 15 435-state table (7.6×) | GPU ~6 min, CPU ~40 min |
| [example3](docs/examples/example3.md) | K1 velocity, flat | PPO | `track_linear_velocity`, `track_angular_velocity` | NVIDIA GPU |
| [example4](docs/examples/example4.md) | K1 velocity, flat | PPO | `upright`, `pose` | NVIDIA GPU |
| [example5](docs/examples/example5.md) | K1 velocity, flat | PPO | `action_rate_l2`, `dof_pos_limits` | NVIDIA GPU |
| [example6](docs/examples/example6.md) | K1 velocity, flat | PPO | `foot_clearance`, `foot_swing_height` | NVIDIA GPU |
| [example7](docs/examples/example7.md) | K1 velocity, flat | PPO | `foot_slip`, `soft_landing` | NVIDIA GPU |
| [example8](docs/examples/example8.md) | K1 velocity, flat | PPO | `air_time`, `self_collisions` | NVIDIA GPU |
| [example9](docs/examples/example9.md) | K1 velocity, flat | PPO | `body_ang_vel`, `angular_momentum` (full reward) | NVIDIA GPU |
| [example10](docs/examples/example10.md) | K1 velocity, rough terrain | PPO | Rough terrain, domain randomization, pushes, terrain curriculum | NVIDIA GPU |

The Furuta pendulum's arm is driven by the same motor model as the K1's knee, and both act through position targets at 50 Hz, so the same ideas carry over from example1 to example10.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13 (uv installs it for you).

```bash
git clone https://github.com/LIRA-UNAM/eir-course-reinforcement-learning.git
cd eir-course-reinforcement-learning
uv sync
```

The project is built on [mjlab](https://github.com/mujocolab/mjlab) 1.6.0 with MuJoCo 3.11.0 and MuJoCo Warp 3.11.0.

**Hardware.**

- **example1 and example2** train fastest on an NVIDIA GPU (about 1 and 6 minutes on a laptop GPU), and also train on a CPU: about 5 minutes for example1 and 40 minutes for example2.
- **example3–example10** need Linux with an NVIDIA GPU (CUDA 12.4+ recommended) to *train*. You can *play* the pretrained policies on any computer; without a GPU they run on the CPU, more slowly. You do not need a GPU to follow the course: every example ships with a trained policy.

## Run an example

```bash
uv run train example1                     # train
uv run play example1                      # watch the pretrained policy (checkpoints/example1/)

uv run train example3 --env.scene.num-envs 4096
uv run play example3                      # pretrained policy, Viser viewer with a velocity joystick
uv run play example3 --wandb-run-path <entity>/eir_course/<run_id>
uv run play example3 --checkpoint-file logs/rsl_rl/example3/<run>/model_9999.pt
```

- `play` looks for a policy in this order: `--wandb-run-path`, `--checkpoint-file`, then the pretrained file in `checkpoints/<example>/`.
- Add `--help` after an example name to see all options, for example `uv run train example2 --help`.
- All ten examples are mjlab tasks: `uv run list_envs` lists them. example1 and example2 plug a tabular Q-learning runner into mjlab's `train` and `play` in place of PPO. Its settings are CLI flags too, e.g. `uv run train example1 --agent.bins 15,15,9,9`.
- To train every K1 example back to back on a GPU machine: `uv run train_queue queues/course.train.yaml`.

## Weights & Biases

Training logs to the W&B project `eir_course`. Log in once with `uv run wandb login`, or train offline:

`WANDB_MODE=offline uv run train exampleN` trains without a W&B account.

Every example logs each reward term separately (`Episode_Reward/<term>`). The Furuta examples also log their success rate, Q-table coverage and heatmaps of the Q-table. Add `--video` to record training videos.

## Repository layout

```
src/eir_course/
  asset_zoo/robots/furuta/       Furuta pendulum MJCF (arm motor = K1 knee motor)
  asset_zoo/robots/k1/           Booster K1 model, motors and actuator config
  tasks/furuta/                  example1-2: mjlab env config, discretizer, Q-learning runner
  tasks/velocity/config/k1/      example3-10: env configs, reward groups, PPO config
  scripts/                       train / play dispatchers, train_queue
docs/lecture.md                  lecture notes (questions 0-8)
docs/examples/                   one guide per example
checkpoints/                     pretrained policies
queues/course.train.yaml         train example3-10 back to back
tests/                           uv run pytest
```

## Development

```bash
make sync       # install with dev tools
make format     # ruff
make test       # tests (make test-fast skips building simulation environments)
```

The test targets unset `PYTHONPATH`, so a sourced ROS installation does not leak its pytest plugins into the run.
