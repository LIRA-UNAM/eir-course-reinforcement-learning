"""Course environments: the K1 velocity task built up two reward terms at a time.

example3-example9 train on flat ground with no domain randomization and no
pushes, so the only thing that changes between them is the reward. example10 is
the full rough-terrain task from ``env_cfgs.py``.

No example uses K1's speed curriculum: every example samples commands from the
fixed ``COMMAND_RANGES``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from eir_course.tasks.velocity.config.k1.env_cfgs import (
  k1_flat_env_cfg,
  k1_rough_env_cfg,
)

# Cumulative reward groups. Example ``FIRST_REWARD_EXAMPLE + i`` uses groups 0..i.
REWARD_GROUPS: tuple[tuple[str, str], ...] = (
  ("track_linear_velocity", "track_angular_velocity"),
  ("upright", "pose"),
  ("action_rate_l2", "dof_pos_limits"),
  ("foot_clearance", "foot_swing_height"),
  ("foot_slip", "soft_landing"),
  ("air_time", "self_collisions"),
  ("body_ang_vel", "angular_momentum"),
)

FIRST_REWARD_EXAMPLE = 3
FULL_TASK_EXAMPLE = FIRST_REWARD_EXAMPLE + len(REWARD_GROUPS)  # example10

# Sim-to-real randomization, removed from the reward-only examples.
_RANDOMIZATION_EVENTS = (
  "push_robot",
  "foot_friction",
  "encoder_bias",
  "base_com",
  "pd_gains",
  "trunk_inertia",
  "limb_inertia",
  "terrain_contact",
)


def reward_terms(num_groups: int) -> tuple[str, ...]:
  """Reward term names active when the first ``num_groups`` groups are enabled."""
  if not 1 <= num_groups <= len(REWARD_GROUPS):
    raise ValueError(
      f"num_groups must be in [1, {len(REWARD_GROUPS)}], got {num_groups}"
    )
  return tuple(name for group in REWARD_GROUPS[:num_groups] for name in group)


# Fixed command ranges for every K1 example (no speed curriculum).
COMMAND_RANGES: dict[str, tuple[float, float]] = {
  "lin_vel_x": (-1.0, 1.0),  # m/s
  "lin_vel_y": (-0.5, 0.5),  # m/s
  "ang_vel_z": (-1.2, 1.2),  # rad/s (yaw)
}


def _fix_command_ranges(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop the speed curriculum and sample commands from ``COMMAND_RANGES``."""
  cfg.curriculum.pop("command_vel", None)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  for field, value in COMMAND_RANGES.items():
    setattr(twist.ranges, field, value)


def course_flat_env_cfg(num_groups: int, play: bool = False) -> ManagerBasedRlEnvCfg:
  """K1 flat-ground velocity task with only the first ``num_groups`` reward groups."""
  cfg = k1_flat_env_cfg(play=play)

  keep = set(reward_terms(num_groups))
  cfg.rewards = {name: term for name, term in cfg.rewards.items() if name in keep}
  missing = keep - cfg.rewards.keys()
  assert not missing, f"Reward terms not in the K1 config: {sorted(missing)}"

  for name in _RANDOMIZATION_EVENTS:
    cfg.events.pop(name, None)

  _fix_command_ranges(cfg)
  return cfg


def course_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The full task: every reward term, rough terrain, randomization, terrain curriculum."""
  cfg = k1_rough_env_cfg(play=play)
  _fix_command_ranges(cfg)
  return cfg
