"""Register the Furuta pendulum examples (example1, example2) with mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import furuta_balance_env_cfg, furuta_swingup_env_cfg
from .rl_cfg import furuta_balance_q_learning_cfg, furuta_swingup_q_learning_cfg
from .runner import QLearningRunner

register_mjlab_task(
  task_id="example1",
  env_cfg=furuta_balance_env_cfg(),
  play_env_cfg=furuta_balance_env_cfg(play=True),
  rl_cfg=furuta_balance_q_learning_cfg(),
  runner_cls=QLearningRunner,
)

register_mjlab_task(
  task_id="example2",
  env_cfg=furuta_swingup_env_cfg(),
  play_env_cfg=furuta_swingup_env_cfg(play=True),
  rl_cfg=furuta_swingup_q_learning_cfg(),
  runner_cls=QLearningRunner,
)
