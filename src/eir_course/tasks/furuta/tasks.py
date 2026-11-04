"""Task specifications for the two Furuta pendulum examples.

Both examples share the robot, the motor and the action space. They differ in
where the pendulum starts and in how much of the state space the Q-table has to
cover, which is the point: example2 needs a much larger table and far more
experience than example1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Booster K1 knee pitch motor (ACTUATOR_E6416 in asset_zoo/robots/k1/k1_constants.py).
MOTOR_EFFORT_LIMIT = 112.0  # N*m
MOTOR_STIFFNESS = 80.0  # N*m/rad
# Same action-scale rule as every K1 joint: 0.25 * effort_limit / stiffness.
K1_ACTION_SCALE = 0.25 * MOTOR_EFFORT_LIMIT / MOTOR_STIFFNESS  # 0.35 rad


@dataclass(frozen=True)
class StateDim:
  """One discretized state variable."""

  name: str
  bins: int
  low: float
  high: float
  sqrt_spacing: bool = False
  """Place bin edges densely near 0 (spacing grows with sqrt(|x|))."""


@dataclass(frozen=True)
class FurutaTask:
  """Everything that differs between the balance and swing-up examples."""

  name: str
  description: str
  init_alpha_range: tuple[float, float]
  """Pendulum angle at reset, measured from upright (rad)."""
  state_dims: tuple[StateDim, ...]
  action_scale: float
  """Arm target = action_scale * a, with a in linspace(-1, 1, num_actions)."""
  alpha_limit: float | None
  """Terminate when |alpha| exceeds this (None: never)."""
  arm_limit: float
  """Terminate when |theta| exceeds this."""
  episode_length_s: float
  episodes: int
  swing_up_reward: bool
  num_actions: int = 7
  success_angle: float = 0.2
  """|alpha| below this counts as balanced."""
  success_time_s: float = 2.0
  """An episode succeeds if it ends with this long balanced."""


BALANCE = FurutaTask(
  name="example1",
  description="Furuta pendulum: keep it balanced, starting near upright.",
  init_alpha_range=(-math.radians(15.0), math.radians(15.0)),
  state_dims=(
    StateDim("alpha", 9, -0.4, 0.4),
    StateDim("alpha_dot", 9, -4.0, 4.0),
    StateDim("theta", 5, -0.8, 0.8),
    StateDim("theta_dot", 5, -8.0, 8.0),
  ),
  action_scale=K1_ACTION_SCALE,
  alpha_limit=0.4,
  arm_limit=1.2,
  episode_length_s=10.0,
  episodes=5_000,
  swing_up_reward=False,
)

SWING_UP = FurutaTask(
  name="example2",
  description="Furuta pendulum: swing up from hanging at rest, then balance.",
  init_alpha_range=(math.pi, math.pi),
  state_dims=(
    # Finer bins near upright, where balancing needs resolution.
    StateDim("alpha", 21, -math.pi, math.pi, sqrt_spacing=True),
    StateDim("alpha_dot", 15, -12.0, 12.0),
    StateDim("theta", 7, -1.5, 1.5),
    StateDim("theta_dot", 7, -10.0, 10.0),
  ),
  # Pumping energy into the pendulum needs a bigger arm swing than balancing.
  action_scale=1.0,
  alpha_limit=None,
  arm_limit=1.5,
  episode_length_s=15.0,
  episodes=30_000,
  swing_up_reward=True,
)

TASKS: dict[str, FurutaTask] = {t.name: t for t in (BALANCE, SWING_UP)}
