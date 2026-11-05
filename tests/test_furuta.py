"""Furuta pendulum mjlab tasks (example1, example2) and the Q-learning runner."""

import math
from dataclasses import asdict, replace

import pytest
import torch
from eir_course.asset_zoo.robots.k1.k1_constants import ACTUATOR_E6416
from eir_course.tasks.furuta.discretize import Discretizer, bin_edges
from eir_course.tasks.furuta.rl_cfg import QLearningRunnerCfg
from eir_course.tasks.furuta.runner import QLearningRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

FURUTA_EXAMPLES = ("example1", "example2")


def _q_cfg(task: str) -> QLearningRunnerCfg:
  cfg = load_rl_cfg(task)
  assert isinstance(cfg, QLearningRunnerCfg)
  return cfg


def _arm_scale(task: str) -> float:
  action = load_env_cfg(task).actions["arm"]
  assert isinstance(action, JointPositionActionCfg)
  assert isinstance(action.scale, float)
  return action.scale


def _make_env(task: str, num_envs: int = 4) -> RslRlVecEnvWrapper:
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))


@pytest.fixture(scope="module")
def balance_env():
  env = _make_env("example1")
  yield env
  env.close()


def test_examples_use_the_q_learning_runner():
  for task in FURUTA_EXAMPLES:
    assert load_runner_cls(task) is QLearningRunner


@pytest.mark.parametrize("task", FURUTA_EXAMPLES)
def test_runs_log_to_the_furuta_project(task):
  cfg = _q_cfg(task)
  assert cfg.wandb_project == "furuta_pendulum"
  assert cfg.experiment_name == task


def test_arm_motor_is_the_k1_knee(balance_env):
  model = balance_env.unwrapped.sim.mj_model
  act = model.actuator(0)
  assert model.nu == 1
  assert act.gainprm[0] == pytest.approx(ACTUATOR_E6416.stiffness)
  assert act.biasprm[1] == pytest.approx(-ACTUATOR_E6416.stiffness)
  assert act.biasprm[2] == pytest.approx(-ACTUATOR_E6416.damping)
  assert tuple(act.forcerange) == pytest.approx(
    (-ACTUATOR_E6416.effort_limit, ACTUATOR_E6416.effort_limit)
  )
  arm_dof = model.jnt_dofadr[act.trnid[0]]
  assert model.dof_armature[arm_dof] == pytest.approx(ACTUATOR_E6416.armature)


def test_action_scale_follows_the_k1_rule():
  balance = _arm_scale("example1")
  assert balance == pytest.approx(
    0.25 * ACTUATOR_E6416.effort_limit / ACTUATOR_E6416.stiffness
  )
  assert _arm_scale("example2") == 1.0


def test_rewards_and_terminations():
  balance, swingup = load_env_cfg("example1"), load_env_cfg("example2")
  assert set(balance.rewards) == {"alive", "arm_angle"}
  assert set(swingup.rewards) == {"pendulum_upright", "arm_angle", "arm_velocity"}
  assert "pendulum_fell" in balance.terminations
  assert "pendulum_fell" not in swingup.terminations
  assert "pendulum_too_fast" in swingup.terminations
  assert "pendulum_too_fast" not in balance.terminations


def test_swingup_speed_limit_matches_the_table():
  limit = load_env_cfg("example2").terminations["pendulum_too_fast"].params["limit"]
  cfg = _q_cfg("example2")
  assert (cfg.state_low[1], cfg.state_high[1]) == (-limit, limit)


def test_upright_reward_pays_less_when_spinning(balance_env):
  from eir_course.tasks.furuta import mdp

  env = balance_env.unwrapped
  robot = env.scene["robot"]
  cfg = mdp.pendulum_joint()
  cfg.resolve(env.scene)
  joint = cfg.joint_ids
  for speed, expected in ((0.0, 1.0), (5.0, 0.55), (40.0, 0.5)):
    pos = torch.zeros_like(robot.data.joint_pos)
    vel = torch.zeros_like(robot.data.joint_vel)
    vel[:, joint] = speed
    robot.write_joint_state_to_sim(pos, vel)
    reward = mdp.pendulum_upright(env, cfg, velocity_margin=5.0)
    assert reward == pytest.approx(torch.full_like(reward, expected), abs=1e-3)


def test_pendulum_falls_from_near_upright(balance_env):
  obs, _ = balance_env.reset()
  hold = torch.zeros(balance_env.num_envs, 1)
  fell = torch.zeros(balance_env.num_envs, dtype=torch.bool)
  for _ in range(100):  # 2 s with the arm held at its centre
    obs, _, dones, extras = balance_env.step(hold)
    fell |= dones.bool() & ~extras["time_outs"].bool()
  assert fell.all()


def test_observation_is_the_discretized_state(balance_env):
  obs = balance_env.get_observations()["actor"]
  assert obs.shape == (balance_env.num_envs, 4)
  assert torch.all(torch.abs(obs[:, 0]) <= math.pi)


@pytest.mark.parametrize("task", FURUTA_EXAMPLES)
def test_discretizer_covers_the_table(task):
  cfg = _q_cfg(task)
  disc = Discretizer(cfg.bins, cfg.state_low, cfg.state_high, cfg.sqrt_spacing)
  assert disc.num_states == math.prod(cfg.bins)
  low = torch.tensor(cfg.state_low) - 100.0
  high = torch.tensor(cfg.state_high) + 100.0
  assert disc.index(low[None]).item() == 0
  assert disc.index(high[None]).item() == disc.num_states - 1
  # Upright and still lands in the middle bin of every dimension.
  assert disc.bins(torch.zeros(1, 4))[0].tolist() == [b // 2 for b in cfg.bins]


def test_sqrt_spacing_is_denser_near_upright():
  edges = bin_edges(21, -math.pi, math.pi, sqrt_spacing=True)
  assert len(edges) == 20
  assert torch.allclose(edges, -edges.flip(0))
  widths = torch.diff(edges)
  assert widths[len(widths) // 2] < widths[0]


def test_example2_table_is_bigger():
  b1, b2 = _q_cfg("example1").bins, _q_cfg("example2").bins
  assert math.prod(b2) > 5 * math.prod(b1)


@pytest.mark.slow
@pytest.mark.parametrize("task", FURUTA_EXAMPLES)
def test_short_training_run_saves_and_reloads(task, tmp_path):
  env = _make_env(task, num_envs=8)
  try:
    cfg = asdict(replace(_q_cfg(task), logger="tensorboard", save_interval=2))
    runner = QLearningRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=3)
    assert torch.any(runner.q != 0)
    assert int((runner.visits > 0).sum()) > 0
    checkpoints = sorted(tmp_path.glob("model_*.pt"))
    assert [p.name for p in checkpoints] == ["model_1.pt", "model_2.pt"]

    player = QLearningRunner(env, cfg, device="cpu")
    player.load(str(checkpoints[-1]))
    assert torch.equal(player.q, runner.q)
    actions = player.get_inference_policy()(env.get_observations())
    assert actions.shape == (env.num_envs, 1)
    assert torch.all(actions.abs() <= 1.0)
  finally:
    env.close()


@pytest.mark.slow
def test_bins_override_is_restored_on_load(tmp_path):
  env = _make_env("example1", num_envs=4)
  try:
    cfg = asdict(
      replace(load_rl_cfg("example1"), logger="tensorboard", bins=(3, 3, 3, 3))
    )
    runner = QLearningRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=1)
    player = QLearningRunner(env, asdict(load_rl_cfg("example1")), device="cpu")
    player.load(str(tmp_path / "model_0.pt"))
    assert player.q.shape == (81, 7)
  finally:
    env.close()
