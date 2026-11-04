"""The K1 examples add exactly two reward terms each, on top of a fixed environment."""

import pytest
import torch
from eir_course.tasks.velocity.config.k1.course_cfgs import (
  FIRST_REWARD_EXAMPLE,
  FULL_TASK_EXAMPLE,
  REWARD_GROUPS,
  course_flat_env_cfg,
  reward_terms,
)
from eir_course.tasks.velocity.config.k1.env_cfgs import k1_rough_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

REWARD_EXAMPLES = range(FIRST_REWARD_EXAMPLE, FULL_TASK_EXAMPLE)
COMMAND_RANGES = {
  "lin_vel_x": (-1.0, 1.0),
  "lin_vel_y": (-0.5, 0.5),
  "ang_vel_z": (-1.2, 1.2),
}
RANDOMIZATION_EVENTS = {
  "push_robot",
  "foot_friction",
  "encoder_bias",
  "base_com",
  "pd_gains",
  "trunk_inertia",
  "limb_inertia",
  "terrain_contact",
}


def test_examples_are_registered():
  tasks = set(list_tasks())
  assert {f"example{n}" for n in range(3, 11)} <= tasks
  assert FULL_TASK_EXAMPLE == 10


def test_groups_cover_every_k1_reward_exactly_once():
  names = [name for group in REWARD_GROUPS for name in group]
  assert all(len(group) == 2 for group in REWARD_GROUPS)
  assert len(names) == len(set(names))
  assert set(names) == set(k1_rough_env_cfg().rewards)


@pytest.mark.parametrize("example", REWARD_EXAMPLES)
@pytest.mark.parametrize("play", [False, True])
def test_reward_examples_are_cumulative(example: int, play: bool):
  num_groups = example - FIRST_REWARD_EXAMPLE + 1
  cfg = load_env_cfg(f"example{example}", play=play)
  assert set(cfg.rewards) == set(reward_terms(num_groups))
  if num_groups > 1:
    previous = set(reward_terms(num_groups - 1))
    assert len(set(cfg.rewards) - previous) == 2


@pytest.mark.parametrize("example", REWARD_EXAMPLES)
def test_reward_examples_have_fixed_flat_env(example: int):
  cfg = load_env_cfg(f"example{example}")
  assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "plane"
  assert not RANDOMIZATION_EVENTS & set(cfg.events)
  assert not cfg.curriculum


@pytest.mark.parametrize("example", range(3, 11))
@pytest.mark.parametrize("play", [False, True])
def test_commands_are_fixed(example: int, play: bool):
  cfg = load_env_cfg(f"example{example}", play=play)
  assert "command_vel" not in cfg.curriculum
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  for field, value in COMMAND_RANGES.items():
    assert getattr(twist.ranges, field) == value


def test_reward_terms_keep_k1_weights():
  full = k1_rough_env_cfg().rewards
  last = course_flat_env_cfg(len(REWARD_GROUPS)).rewards
  assert {k: v.weight for k, v in last.items()} == {k: full[k].weight for k in last}


def test_full_task_is_the_k1_rough_task():
  cfg = load_env_cfg(f"example{FULL_TASK_EXAMPLE}")
  reference = k1_rough_env_cfg()
  assert set(cfg.rewards) == set(reference.rewards)
  assert set(cfg.events) == set(reference.events)
  assert set(cfg.curriculum) == set(reference.curriculum) - {"command_vel"}
  assert set(cfg.curriculum) == {"terrain_levels"}
  assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "generator"


@pytest.mark.parametrize("example", range(3, 11))
def test_runner_logs_to_course_project(example: int):
  rl_cfg = load_rl_cfg(f"example{example}")
  assert rl_cfg.wandb_project == "eir_course"
  assert rl_cfg.experiment_name == f"example{example}"
  assert rl_cfg.max_iterations == 10_000


@pytest.mark.slow
@pytest.mark.parametrize("example", [FIRST_REWARD_EXAMPLE, FULL_TASK_EXAMPLE])
def test_env_builds_and_steps_on_cpu(example: int):
  cfg = load_env_cfg(f"example{example}")
  cfg.scene.num_envs = 2
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    actions = torch.zeros(2, env.action_manager.total_action_dim)
    reward = torch.empty(0)
    for _ in range(3):
      _, reward, *_ = env.step(actions)
    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()
  finally:
    env.close()
