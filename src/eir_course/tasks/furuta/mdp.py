"""Observation, reward and termination terms for the Furuta pendulum."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def arm_joint() -> SceneEntityCfg:
  """The arm joint. Pass it through a term's ``params`` so mjlab resolves it."""
  return SceneEntityCfg("robot", joint_names=("arm",))


def pendulum_joint() -> SceneEntityCfg:
  """The pendulum joint. Pass it through a term's ``params`` so mjlab resolves it."""
  return SceneEntityCfg("robot", joint_names=("pendulum",))


def _joint(
  env: ManagerBasedRlEnv, cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[cfg.name]
  return (
    asset.data.joint_pos[:, cfg.joint_ids].squeeze(-1),
    asset.data.joint_vel[:, cfg.joint_ids].squeeze(-1),
  )


def pendulum_angle(
  env: ManagerBasedRlEnv, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Pendulum angle from upright, wrapped to [-pi, pi). Shape: [num_envs]."""
  alpha, _ = _joint(env, pendulum_cfg)
  return torch.remainder(alpha + torch.pi, 2 * torch.pi) - torch.pi


# Observations.


def furuta_state(
  env: ManagerBasedRlEnv,
  arm_cfg: SceneEntityCfg,
  pendulum_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """(alpha, alpha_dot, theta, theta_dot). Shape: [num_envs, 4].

  alpha is the pendulum angle from upright, theta the arm angle. The Q-learning
  runner discretizes these four values, in this order.
  """
  _, alpha_dot = _joint(env, pendulum_cfg)
  theta, theta_dot = _joint(env, arm_cfg)
  alpha = pendulum_angle(env, pendulum_cfg)
  return torch.stack([alpha, alpha_dot, theta, theta_dot], dim=-1)


# Rewards.


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 for every step the episode keeps going."""
  return torch.ones(env.num_envs, device=env.device)


# dm_control's tolerance: 1 at 0, ``value_at_margin`` at |x| = margin.
_VALUE_AT_MARGIN = 0.1


def pendulum_upright(
  env: ManagerBasedRlEnv, pendulum_cfg: SceneEntityCfg, velocity_margin: float
) -> torch.Tensor:
  """Upright *and slow*: (1 + cos alpha) / 2 * (1 + slow) / 2, in [0, 1].

  ``slow`` is a Gaussian tolerance on the pendulum speed: 1 at rest, 0.1 at
  |alpha_dot| = ``velocity_margin``. Without it, spinning the pendulum through
  upright averages half the reward for free and the agent learns to spin. Same
  shape as the dm_control cartpole "smooth" reward (Tassa et al., 2018).
  """
  _, alpha_dot = _joint(env, pendulum_cfg)
  upright = 0.5 * (1.0 + torch.cos(pendulum_angle(env, pendulum_cfg)))
  scale = math.sqrt(-2.0 * math.log(_VALUE_AT_MARGIN)) / velocity_margin
  slow = torch.exp(-0.5 * torch.square(alpha_dot * scale))
  return upright * 0.5 * (1.0 + slow)


def arm_angle_l2(env: ManagerBasedRlEnv, arm_cfg: SceneEntityCfg) -> torch.Tensor:
  """theta^2: pulls the arm back towards its centre."""
  theta, _ = _joint(env, arm_cfg)
  return torch.square(theta)


def arm_velocity_l2(env: ManagerBasedRlEnv, arm_cfg: SceneEntityCfg) -> torch.Tensor:
  """theta_dot^2: discourages spinning the arm."""
  _, theta_dot = _joint(env, arm_cfg)
  return torch.square(theta_dot)


# Terminations.


def pendulum_fell(
  env: ManagerBasedRlEnv, limit: float, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The pendulum is more than ``limit`` rad from upright."""
  return torch.abs(pendulum_angle(env, pendulum_cfg)) > limit


def pendulum_too_fast(
  env: ManagerBasedRlEnv, limit: float, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The pendulum spins faster than ``limit`` rad/s."""
  _, alpha_dot = _joint(env, pendulum_cfg)
  return torch.abs(alpha_dot) > limit


def arm_out_of_range(
  env: ManagerBasedRlEnv, limit: float, arm_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The arm turned more than ``limit`` rad from its centre."""
  theta, _ = _joint(env, arm_cfg)
  return torch.abs(theta) > limit
