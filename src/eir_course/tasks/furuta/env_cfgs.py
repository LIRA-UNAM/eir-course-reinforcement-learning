"""Furuta pendulum environments: example1 (balance) and example2 (swing-up)."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_joints_by_offset, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from eir_course.asset_zoo.robots.furuta.furuta_constants import (
  FURUTA_ACTION_SCALE,
  get_furuta_cfg,
)
from eir_course.tasks.furuta import mdp

# Fastest pendulum speed (rad/s) example2 allows; also its Q-table's alpha_dot range.
# A free fall from upright peaks at ~10.6 rad/s, so a swing-up passes ~11 rad/s:
# 20 leaves headroom for that and still ends the ~40 rad/s spinning exploit.
SWING_UP_MAX_PENDULUM_SPEED = 20.0


def _joints() -> dict:
  return {"arm_cfg": mdp.arm_joint(), "pendulum_cfg": mdp.pendulum_joint()}


def _make_furuta_env_cfg(swing_up: bool) -> ManagerBasedRlEnvCfg:
  observations = {
    # One term with the four values the Q-table is indexed by. No noise: the
    # point of these examples is the table, not robustness.
    "actor": ObservationGroupCfg(
      {"state": ObservationTermCfg(func=mdp.furuta_state, params=_joints())},
      enable_corruption=False,
    ),
  }

  # Absolute arm position target q* = scale * a. The Q-learning runner only
  # ever sends a from a fixed set of levels in [-1, 1].
  actions: dict[str, ActionTermCfg] = {
    "arm": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=("arm",),
      # Pumping energy into the pendulum needs a bigger arm swing than balancing.
      scale=1.0 if swing_up else FURUTA_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  pendulum_range = (0.0, 0.0) if swing_up else (-math.radians(90), math.radians(90))
  events = {
    "reset_arm": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-math.pi, math.pi),
        "velocity_range": (-20.0, 20.0),
        "asset_cfg": mdp.arm_joint(),
      },
    ),
    "reset_pendulum": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": pendulum_range,
        "velocity_range": (-40.0, 40.0),
        "asset_cfg": mdp.pendulum_joint(),
      },
    ),
  }

  if swing_up:
    # Non-negative when the arm is centred: ending the episode early must
    # never pay off, or the agent learns to crash on purpose.
    rewards = {
      "pendulum_upright": RewardTermCfg(
        func=mdp.pendulum_upright,
        weight=1.0,
        params={"pendulum_cfg": mdp.pendulum_joint(), "velocity_margin": 5.0},
      ),
      "arm_angle": RewardTermCfg(
        func=mdp.arm_angle_l2, weight=-0.1, params={"arm_cfg": mdp.arm_joint()}
      ),
      "arm_velocity": RewardTermCfg(
        func=mdp.arm_velocity_l2, weight=-0.001, params={"arm_cfg": mdp.arm_joint()}
      ),
    }
  else:
    rewards = {
      "alive": RewardTermCfg(func=mdp.alive, weight=1.0),
      "arm_angle": RewardTermCfg(
        func=mdp.arm_angle_l2, weight=-0.1, params={"arm_cfg": mdp.arm_joint()}
      ),
    }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "arm_out_of_range": TerminationTermCfg(
      func=mdp.arm_out_of_range,
      params={"limit": 1.5 if swing_up else 1.2, "arm_cfg": mdp.arm_joint()},
    ),
  }
  if swing_up:
    # Spinning past the Q-table's alpha_dot range ends the episode:
    # it forfeits all future reward, and keeps the state inside the table.
    terminations["pendulum_too_fast"] = TerminationTermCfg(
      func=mdp.pendulum_too_fast,
      params={
        "limit": SWING_UP_MAX_PENDULUM_SPEED,
        "pendulum_cfg": mdp.pendulum_joint(),
      },
    )
  else:
    # Swinging up *is* falling, so only the balance task ends on a fall.
    terminations["pendulum_fell"] = TerminationTermCfg(
      func=mdp.pendulum_fell,
      params={"limit": 0.4, "pendulum_cfg": mdp.pendulum_joint()},
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_furuta_cfg(pendulum_angle=math.pi if swing_up else 0.0)},
      num_envs=256,
      env_spacing=1.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=1.8,
      elevation=-20.0,
      azimuth=135.0,
    ),
    # Physics at 200 Hz, control at 50 Hz: the same rates as the K1 examples.
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, disableflags=("contact",))),
    decimation=4,
    episode_length_s=15.0 if swing_up else 10.0,
  )


def _play(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 30.0
  return cfg


def furuta_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """example1: start within +-15 deg of upright and keep the pendulum balanced."""
  cfg = _make_furuta_env_cfg(swing_up=False)
  return _play(cfg) if play else cfg


def furuta_swingup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """example2: start hanging down at rest, swing up and balance."""
  cfg = _make_furuta_env_cfg(swing_up=True)
  return _play(cfg) if play else cfg
