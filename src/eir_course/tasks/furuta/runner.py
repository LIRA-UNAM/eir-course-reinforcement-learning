"""Tabular Q-learning runner for mjlab (example1, example2).

mjlab's ``train`` and ``play`` scripts drive any runner class registered with a
task. This one replaces PPO with a Q-table: no neural network, no gradients.
It provides the methods those scripts call: ``learn``, ``save``, ``load``,
``get_inference_policy`` and ``add_git_repo_to_log``.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from tensordict import TensorDict

from eir_course.tasks.furuta.discretize import Discretizer


class QLearningRunner:
  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.log_dir = log_dir
    self.device = device
    action_dim = env.unwrapped.action_manager.total_action_dim
    assert action_dim == 1, f"Q-learning needs a 1-D action, got {action_dim}"
    self.levels = torch.linspace(-1.0, 1.0, train_cfg["num_actions"], device=device)
    self._build_table(
      train_cfg["bins"],
      train_cfg["state_low"],
      train_cfg["state_high"],
      train_cfg["sqrt_spacing"],
    )
    self.current_learning_iteration = 0
    self.generator = torch.Generator(device=device).manual_seed(train_cfg["seed"])
    self.wandb_run = None
    self._learn_start_it = 0

  def _build_table(self, bins, low, high, sqrt_spacing) -> None:
    self.disc_settings = {
      "bins": tuple(bins),
      "state_low": tuple(low),
      "state_high": tuple(high),
      "sqrt_spacing": tuple(sqrt_spacing),
    }
    self.disc = Discretizer(
      tuple(bins), tuple(low), tuple(high), tuple(sqrt_spacing), self.device
    )
    num_actions = len(self.levels)
    self.q = torch.zeros(self.disc.num_states, num_actions, device=self.device)
    self.visits = torch.zeros(
      self.disc.num_states, dtype=torch.long, device=self.device
    )

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """Called by mjlab's train script; nothing to record for this runner."""
    del repo_file_path

  # Acting.

  @staticmethod
  def _states(obs: TensorDict) -> torch.Tensor:
    return obs["actor"]

  def get_inference_policy(self, device: str | None = None):
    """Greedy policy: a = argmax_a Q(s, a), sent to the env as an arm target in [-1, 1]."""
    del device

    def policy(obs: TensorDict) -> torch.Tensor:
      s = self.disc.index(self._states(obs))
      return self.levels[self.q[s].argmax(dim=-1)].unsqueeze(-1)

    return policy

  # Learning.

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    cfg = self.cfg
    env = self.env
    num_envs, num_actions = env.num_envs, len(self.levels)
    lr, gamma = cfg["learning_rate"], cfg["gamma"]
    dt = env.unwrapped.step_dt
    success_steps = round(cfg["success_time_s"] / dt)

    self._init_wandb()
    if init_at_random_ep_len:
      env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.unwrapped.max_episode_length)
      )

    print(
      f"[Q-learning] table: {self.disc.num_states} states x {num_actions} actions = "
      f"{self.disc.num_states * num_actions} entries; {num_envs} envs x "
      f"{cfg['num_steps_per_env']} steps per iteration"
    )

    s = self.disc.index(self._states(env.get_observations()))
    ep_return = torch.zeros(num_envs, device=self.device)
    ep_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    balanced = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    ones = torch.ones(num_envs, device=self.device)

    start_it = self._learn_start_it = self.current_learning_iteration
    decay_its = max(1, int(cfg["epsilon_decay_fraction"] * num_learning_iterations))
    total_steps = 0
    start_time = time.time()

    for it in range(start_it, start_it + num_learning_iterations):
      frac = min(1.0, (it - start_it) / decay_its)
      epsilon = cfg["epsilon_start"] + frac * (
        cfg["epsilon_end"] - cfg["epsilon_start"]
      )
      finished_return, finished_length, finished_success = [], [], []
      term_logs: dict[str, list[float]] = defaultdict(list)
      it_start = time.time()

      with torch.no_grad():
        for _ in range(cfg["num_steps_per_env"]):
          # Epsilon-greedy over the discrete arm targets.
          greedy = self.q[s].argmax(dim=-1)
          random = torch.randint(
            num_actions, (num_envs,), device=self.device, generator=self.generator
          )
          explore = (
            torch.rand(num_envs, device=self.device, generator=self.generator) < epsilon
          )
          a = torch.where(explore, random, greedy)

          obs, reward, dones, extras = env.step(self.levels[a].unsqueeze(-1))
          states = self._states(obs)
          s_next = self.disc.index(states)
          time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
          terminated = dones.bool() & ~time_outs

          # Q-learning target. A terminal state has no future. mjlab has already
          # reset finished envs, so s_next of a time-out is a fresh start: like
          # rsl_rl's PPO, bootstrap a time-out from the current state instead.
          next_value = self.q[s_next].max(dim=-1).values
          next_value = torch.where(time_outs, self.q[s].max(dim=-1).values, next_value)
          target = reward + gamma * next_value * (~terminated).float()

          # Q(s, a) += lr * (target - Q(s, a)). Envs that hit the same (s, a)
          # in this step share one update: the mean of their TD errors.
          cell = s * num_actions + a
          td = target - self.q.view(-1)[cell]
          td_sum = torch.zeros(self.q.numel(), device=self.device).index_add_(
            0, cell, td
          )
          count = torch.zeros(self.q.numel(), device=self.device).index_add_(
            0, cell, ones
          )
          hit = count > 0
          self.q.view(-1)[hit] += lr * td_sum[hit] / count[hit]
          self.visits.index_add_(0, s, torch.ones_like(s))

          # Episode statistics.
          ep_return += reward
          ep_steps += 1
          done = dones.bool()
          if done.any():
            finished_return += ep_return[done].tolist()
            finished_length += (ep_steps[done].float() * dt).tolist()
            finished_success += (balanced[done] >= success_steps).float().tolist()
            ep_return[done] = 0.0
            ep_steps[done] = 0
          upright = torch.abs(states[:, 0]) < cfg["success_angle"]
          balanced = torch.where(
            upright & ~done, balanced + 1, torch.zeros_like(balanced)
          )
          for key, value in extras.get("log", {}).items():
            term_logs[key].append(float(value))

          s = s_next
      total_steps += num_envs * cfg["num_steps_per_env"]
      self.current_learning_iteration = it + 1

      self._log(
        it,
        num_learning_iterations,
        epsilon,
        finished_return,
        finished_length,
        finished_success,
        term_logs,
        total_steps,
        steps_per_s=num_envs * cfg["num_steps_per_env"] / (time.time() - it_start),
        elapsed=time.time() - start_time,
      )
      if self.log_dir is not None and (it + 1) % cfg["save_interval"] == 0:
        self.save(str(Path(self.log_dir) / f"model_{it}.pt"))

    if self.log_dir is not None:
      self.save(
        str(Path(self.log_dir) / f"model_{self.current_learning_iteration - 1}.pt")
      )
    self._finish_wandb()

  # Logging.

  def _init_wandb(self) -> None:
    if (
      self.log_dir is None
      or self.cfg.get("logger") != "wandb"
      or self.wandb_run is not None
    ):
      return
    import wandb

    run_name = self.cfg.get("run_name") or Path(self.log_dir).name
    self.wandb_run = wandb.init(
      project=self.cfg["wandb_project"],
      name=run_name,
      group=self.cfg["experiment_name"],
      dir=self.log_dir,
      config=self.cfg,
    )
    self.wandb_run.summary.update(
      {
        "num_states": self.disc.num_states,
        "q_entries": self.q.numel(),
        "q_table_kib": self.q.numel() * 4 / 1024,
      }
    )

  def _log(
    self,
    it,
    num_learning_iterations,
    epsilon,
    finished_return,
    finished_length,
    finished_success,
    term_logs,
    total_steps,
    steps_per_s,
    elapsed,
  ) -> None:
    visited = float((self.visits > 0).float().mean())
    log: dict = {
      "epsilon": epsilon,
      "q_table/visited_fraction": visited,
      "env_steps": total_steps,
      "steps_per_s": steps_per_s,
    }
    if finished_return:
      log["return/mean"] = float(np.mean(finished_return))
      log["episode_length_s/mean"] = float(np.mean(finished_length))
      log["success_rate"] = float(np.mean(finished_success))
    log.update({k: float(np.mean(v)) for k, v in term_logs.items()})

    if it % 10 == 0 or it == self.current_learning_iteration - 1:
      ret = f"{log['return/mean']:7.3f}" if "return/mean" in log else "    n/a"
      success = f"{log['success_rate']:5.1%}" if "success_rate" in log else "  n/a"
      print(
        f"[Q-learning] it {it + 1:5d}/{num_learning_iterations}  eps {epsilon:.3f}  "
        f"return {ret}  success {success}  visited {visited:6.1%}  "
        f"{steps_per_s:9.0f} steps/s  {elapsed:6.0f} s"
      )
      # Same format as rsl_rl's logger, which train_queue reads for its progress bar.
      done = it + 1 - self._learn_start_it
      eta = elapsed / done * (num_learning_iterations - done)
      print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed))}")
      print(f"ETA: {datetime.timedelta(seconds=int(eta))}", flush=True)

    if self.wandb_run is None:
      return

    if (it + 1) % self.cfg["image_interval"] == 0:
      log.update(self._q_table_media())
    self.wandb_run.log(log, step=it)

  def _q_table_media(self) -> dict:
    import wandb

    q = self.q.cpu().numpy()
    visits = self.visits.cpu().numpy()
    caption = "rows: alpha bins, cols: alpha_dot bins; arm centred and still; grey = never visited"
    images = q_slice_images(q, visits, self.disc)
    media: dict[str, object] = {
      "q_table/value_slice": wandb.Image(images["value"], caption=caption),
      "q_table/greedy_action_slice": wandb.Image(
        images["greedy_action"], caption=caption
      ),
    }
    if visits.any():
      media["q_table/values"] = wandb.Histogram(
        np_histogram=np.histogram(q[visits > 0], bins=64)
      )
    return media

  def _finish_wandb(self) -> None:
    if self.wandb_run is None:
      return
    import wandb

    self.wandb_run.summary["q_table/visited_fraction"] = float(
      (self.visits > 0).float().mean()
    )
    # Videos recorded by `train --video` (mjlab writes them under the run's log dir).
    assert self.log_dir is not None
    for video in sorted((Path(self.log_dir) / "videos").rglob("*.mp4")):
      self.wandb_run.log({"videos/train": wandb.Video(str(video), format="mp4")})
    self.wandb_run.finish()
    self.wandb_run = None

  # Checkpoints.

  def save(self, path: str) -> None:
    torch.save(
      {
        "q": self.q.cpu(),
        "visits": self.visits.cpu(),
        "iteration": self.current_learning_iteration,
        "num_actions": len(self.levels),
        **self.disc_settings,
      },
      path,
    )
    if self.wandb_run is not None and self.cfg.get("upload_model", True):
      self.wandb_run.save(path, base_path=self.log_dir, policy="now")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> None:
    """Load a Q-table. The table's own bins win over the current config, so a
    policy trained with ``--agent.bins`` plays back without repeating the flag."""
    del load_cfg, strict
    data = torch.load(path, map_location=map_location or self.device)
    self.levels = torch.linspace(-1.0, 1.0, data["num_actions"], device=self.device)
    self._build_table(
      data["bins"], data["state_low"], data["state_high"], data["sqrt_spacing"]
    )
    self.q.copy_(data["q"].to(self.device))
    self.visits.copy_(data["visits"].to(self.device))
    self.current_learning_iteration = data["iteration"]


# W&B heatmaps, drawn straight from NumPy arrays (no plotting library).

_VIRIDIS = np.array(
  [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  dtype=np.float64,
)
_UNVISITED = np.array([90, 90, 90], dtype=np.uint8)


def _colorize(values: np.ndarray, mask: np.ndarray, scale: int = 24) -> np.ndarray:
  """Map a 2D array to an RGB image; cells where ``mask`` is False are grey."""
  lo, hi = (values[mask].min(), values[mask].max()) if mask.any() else (0.0, 1.0)
  t = np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * (len(_VIRIDIS) - 1)
  i = np.minimum(t.astype(int), len(_VIRIDIS) - 2)
  f = (t - i)[..., None]
  rgb = ((1 - f) * _VIRIDIS[i] + f * _VIRIDIS[i + 1]).astype(np.uint8)
  rgb[~mask] = _UNVISITED
  return np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))


def q_slice_images(
  q: np.ndarray, visits: np.ndarray, disc: Discretizer
) -> dict[str, np.ndarray]:
  """max_a Q and greedy action over (alpha, alpha_dot), with the arm centred and still."""
  zero = disc.bins(torch.zeros(1, len(disc.shape), device=disc.strides.device))[
    0
  ].tolist()
  i_theta, i_theta_dot = zero[2], zero[3]
  q4 = q.reshape(*disc.shape, q.shape[-1])[:, :, i_theta, i_theta_dot]
  seen = visits.reshape(disc.shape)[:, :, i_theta, i_theta_dot] > 0
  return {
    "value": _colorize(q4.max(axis=-1), seen),
    "greedy_action": _colorize(q4.argmax(axis=-1).astype(float), seen),
  }
