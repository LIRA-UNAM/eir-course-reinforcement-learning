"""Furuta pendulum model, discretizer and Q-learning loop."""

import math
from dataclasses import replace

import numpy as np
import pytest
from eir_course.asset_zoo.robots.k1.k1_constants import ACTUATOR_E6416
from eir_course.tasks.furuta.discretize import Discretizer, bin_edges
from eir_course.tasks.furuta.env import FurutaEnv, wrap_angle
from eir_course.tasks.furuta.q_learning import TrainArgs, load_policy, train
from eir_course.tasks.furuta.tasks import (
  BALANCE,
  K1_ACTION_SCALE,
  SWING_UP,
  TASKS,
  StateDim,
)


def test_arm_motor_is_the_k1_knee():
  m = FurutaEnv(BALANCE).model
  assert m.actuator_gainprm[0, 0] == pytest.approx(ACTUATOR_E6416.stiffness)
  assert m.actuator_biasprm[0, 1] == pytest.approx(-ACTUATOR_E6416.stiffness)
  assert m.actuator_biasprm[0, 2] == pytest.approx(-ACTUATOR_E6416.damping)
  assert tuple(m.actuator_forcerange[0]) == pytest.approx(
    (-ACTUATOR_E6416.effort_limit, ACTUATOR_E6416.effort_limit)
  )
  assert m.dof_armature[m.joint("arm").dofadr[0]] == pytest.approx(
    ACTUATOR_E6416.armature
  )
  assert K1_ACTION_SCALE == pytest.approx(
    0.25 * ACTUATOR_E6416.effort_limit / ACTUATOR_E6416.stiffness
  )


def test_pendulum_falls_from_upright():
  env = FurutaEnv(BALANCE)
  env.reset()
  env.data.qpos[env._pend] = 0.05
  for _ in range(50):  # 1 s with the arm held at 0
    env.step(env.num_actions // 2)
  assert abs(env.state()[0]) > 1.0


def test_wrap_angle():
  assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
  assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
  assert wrap_angle(0.3) == pytest.approx(0.3)


@pytest.mark.parametrize("task", TASKS.values(), ids=list(TASKS))
def test_discretizer_covers_the_table(task):
  disc = Discretizer(task.state_dims)
  assert disc.num_states == int(np.prod([d.bins for d in task.state_dims]))
  low = np.array([d.low for d in task.state_dims]) - 100.0
  high = np.array([d.high for d in task.state_dims]) + 100.0
  assert disc.index(low) == 0
  assert disc.index(high) == disc.num_states - 1
  # Upright and still lands in the middle bin of every dimension.
  assert disc.bins(np.zeros(4)) == tuple(d.bins // 2 for d in task.state_dims)


def test_sqrt_spacing_is_denser_near_upright():
  edges = bin_edges(StateDim("alpha", 21, -math.pi, math.pi, sqrt_spacing=True))
  assert len(edges) == 20
  assert np.allclose(edges, -edges[::-1])
  widths = np.diff(edges)
  assert widths[len(widths) // 2] < widths[0]


def test_example2_is_bigger_than_example1():
  assert (
    Discretizer(SWING_UP.state_dims).num_states
    > 5 * Discretizer(BALANCE.state_dims).num_states
  )


@pytest.mark.parametrize("task_name", list(TASKS))
def test_short_training_run_writes_a_policy(task_name, tmp_path):
  args = TrainArgs(episodes=20, wandb=False, video=False, log_root=str(tmp_path))
  path = train(task_name, args)
  q, dims, targets = load_policy(path)
  disc = Discretizer(dims)
  assert q.shape == (disc.num_states, len(targets))
  assert np.any(q != 0)
  assert dims == TASKS[task_name].state_dims


def test_bins_override(tmp_path):
  args = TrainArgs(
    episodes=2, bins=(3, 3, 3, 3), wandb=False, video=False, log_root=str(tmp_path)
  )
  q, dims, _ = load_policy(train("example1", args))
  assert q.shape[0] == 81
  assert dims == tuple(replace(d, bins=3) for d in BALANCE.state_dims)
