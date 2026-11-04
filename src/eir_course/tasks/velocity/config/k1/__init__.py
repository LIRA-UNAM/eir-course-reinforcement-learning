"""Register the K1 velocity examples (example3-example10) with mjlab."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .course_cfgs import (
  FIRST_REWARD_EXAMPLE,
  FULL_TASK_EXAMPLE,
  REWARD_GROUPS,
  course_flat_env_cfg,
  course_rough_env_cfg,
)
from .rl_cfg import course_ppo_runner_cfg

# Training budget (PPO iterations), the same for every K1 example.
K1_ITERATIONS = 10_000

for _num_groups in range(1, len(REWARD_GROUPS) + 1):
  _example = FIRST_REWARD_EXAMPLE + _num_groups - 1
  register_mjlab_task(
    task_id=f"example{_example}",
    env_cfg=course_flat_env_cfg(_num_groups),
    play_env_cfg=course_flat_env_cfg(_num_groups, play=True),
    rl_cfg=course_ppo_runner_cfg(_example, K1_ITERATIONS),
    runner_cls=VelocityOnPolicyRunner,
  )

register_mjlab_task(
  task_id=f"example{FULL_TASK_EXAMPLE}",
  env_cfg=course_rough_env_cfg(),
  play_env_cfg=course_rough_env_cfg(play=True),
  rl_cfg=course_ppo_runner_cfg(FULL_TASK_EXAMPLE, K1_ITERATIONS),
  runner_cls=VelocityOnPolicyRunner,
)
