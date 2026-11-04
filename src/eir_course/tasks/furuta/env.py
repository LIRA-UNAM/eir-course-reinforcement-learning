"""Furuta pendulum environment on plain MuJoCo (CPU, one robot)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from eir_course.tasks.furuta.tasks import FurutaTask

FURUTA_XML: Path = (
  Path(__file__).resolve().parents[2] / "asset_zoo" / "robots" / "furuta" / "furuta.xml"
)

# Physics at 200 Hz, control at 50 Hz: the same rates as the K1 velocity task.
DECIMATION = 4


def wrap_angle(x: float) -> float:
  """Wrap an angle to [-pi, pi)."""
  return (x + np.pi) % (2 * np.pi) - np.pi


class FurutaEnv:
  """Discrete-action Furuta pendulum.

  State: (alpha, alpha_dot, theta, theta_dot). alpha is the pendulum angle
  from upright, wrapped to [-pi, pi); theta is the arm angle.

  Action: an index into ``action_targets``. The arm motor tracks the chosen
  absolute position target, the same way the K1 policy commands joint targets
  q* = q_default + scale * a.
  """

  def __init__(
    self, task: FurutaTask, action_scale: float | None = None, seed: int = 0
  ):
    self.task = task
    self.model = mujoco.MjModel.from_xml_path(str(FURUTA_XML))
    self.data = mujoco.MjData(self.model)
    scale = task.action_scale if action_scale is None else action_scale
    self.action_targets = scale * np.linspace(-1.0, 1.0, task.num_actions)
    self.control_dt = self.model.opt.timestep * DECIMATION
    self.max_steps = int(round(task.episode_length_s / self.control_dt))
    self.success_steps = int(round(task.success_time_s / self.control_dt))
    self.rng = np.random.default_rng(seed)
    self._arm = self.model.joint("arm").qposadr[0]
    self._pend = self.model.joint("pendulum").qposadr[0]
    self.steps = 0
    self.balanced_steps = 0

  @property
  def num_actions(self) -> int:
    return len(self.action_targets)

  def state(self) -> np.ndarray:
    d = self.data
    return np.array(
      [
        wrap_angle(float(d.qpos[self._pend])),
        d.qvel[self._pend],
        d.qpos[self._arm],
        d.qvel[self._arm],
      ]
    )

  def reset(self) -> np.ndarray:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[self._pend] = self.rng.uniform(*self.task.init_alpha_range)
    self.data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, self.data)
    self.steps = 0
    self.balanced_steps = 0
    return self.state()

  def reward(self, s: np.ndarray) -> float:
    alpha, _, theta, theta_dot = s
    if self.task.swing_up_reward:
      # In [0, 1] when the arm is still and centred: 0 hanging, 1 upright.
      # Kept non-negative so that ending the episode early never pays off.
      return 0.5 * (1.0 + np.cos(alpha)) - 0.1 * theta**2 - 0.001 * theta_dot**2
    # Alive bonus, minus a pull back towards the arm's centre.
    return 1.0 - 0.1 * theta**2

  def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
    """Advance one control step. Returns (state, reward, terminated, truncated)."""
    self.data.ctrl[0] = self.action_targets[action]
    mujoco.mj_step(self.model, self.data, nstep=DECIMATION)
    self.steps += 1

    s = self.state()
    alpha, _, theta, _ = s
    self.balanced_steps = (
      self.balanced_steps + 1 if abs(alpha) < self.task.success_angle else 0
    )

    terminated = abs(theta) > self.task.arm_limit or (
      self.task.alpha_limit is not None and abs(alpha) > self.task.alpha_limit
    )
    truncated = self.steps >= self.max_steps
    return s, self.reward(s), terminated, truncated

  @property
  def succeeded(self) -> bool:
    """The pendulum has been balanced for the last ``success_time_s``."""
    return self.balanced_steps >= self.success_steps
