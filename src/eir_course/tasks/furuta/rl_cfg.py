"""Tabular Q-learning configuration for the Furuta examples (example1, example2)."""

import math
from dataclasses import dataclass

from mjlab.rl import RslRlBaseRunnerCfg

from eir_course.tasks.furuta.env_cfgs import SWING_UP_MAX_PENDULUM_SPEED

# The Furuta runs have their own W&B project, apart from the K1 runs (eir_course).
WANDB_PROJECT = "furuta_pendulum"


@dataclass
class QLearningRunnerCfg(RslRlBaseRunnerCfg):
  """Settings of ``QLearningRunner``; override any of them with ``--agent.<name>``.

  One iteration = ``num_steps_per_env`` steps of every environment, so
  ``num_envs * num_steps_per_env`` transitions update the table per iteration.
  """

  bins: tuple[int, int, int, int] = (9, 9, 5, 5)
  """Bins for (alpha, alpha_dot, theta, theta_dot). Try larger values to watch the table grow."""
  state_low: tuple[float, float, float, float] = (-0.4, -4.0, -0.8, -8.0)
  state_high: tuple[float, float, float, float] = (0.4, 4.0, 0.8, 8.0)
  sqrt_spacing: tuple[bool, bool, bool, bool] = (False, False, False, False)
  """Per dimension: pack the bin edges densely near 0."""
  num_actions: int = 7
  """Arm targets a in linspace(-1, 1, num_actions), scaled by the env's action scale."""
  learning_rate: float = 0.1
  gamma: float = 0.99
  epsilon_start: float = 1.0
  epsilon_end: float = 0.05
  epsilon_decay_fraction: float = 0.8
  """Fraction of the iterations over which epsilon decays linearly."""
  success_angle: float = 0.2
  """|alpha| below this (rad) counts as balanced."""
  success_time_s: float = 2.0
  """An episode succeeds if it ends with this long balanced."""
  image_interval: int = 50
  """Iterations between Q-table heatmaps logged to W&B."""


def furuta_balance_q_learning_cfg() -> QLearningRunnerCfg:
  """example1: a 9 x 9 x 5 x 5 = 2025-state table."""
  return QLearningRunnerCfg(
    experiment_name="example1",
    wandb_project=WANDB_PROJECT,
    max_iterations=300,
    save_interval=100,
  )


def furuta_swingup_q_learning_cfg() -> QLearningRunnerCfg:
  """example2: a 21 x 15 x 7 x 7 = 15 435-state table covering the whole circle of alpha."""
  return QLearningRunnerCfg(
    experiment_name="example2",
    wandb_project=WANDB_PROJECT,
    bins=(21, 15, 7, 7),
    state_low=(-math.pi, -SWING_UP_MAX_PENDULUM_SPEED, -1.5, -10.0),
    state_high=(math.pi, SWING_UP_MAX_PENDULUM_SPEED, 1.5, 10.0),
    # Wide bins where the pendulum swings through, narrow ones near upright.
    sqrt_spacing=(True, False, False, False),
    max_iterations=3000,
    save_interval=500,
    image_interval=250,
  )
