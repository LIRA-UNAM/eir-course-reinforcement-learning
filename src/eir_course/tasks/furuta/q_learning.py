"""Tabular Q-learning for the Furuta pendulum (example1 and example2).

Train:  uv run train example1 [--episodes N] [--bins 9 9 5 5] [--no-wandb]
Play:   uv run play example1 [--wandb-run-path entity/eir_course/run_id]
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import trange

from eir_course.tasks.furuta.discretize import Discretizer
from eir_course.tasks.furuta.env import FurutaEnv
from eir_course.tasks.furuta.tasks import TASKS, FurutaTask, StateDim

WANDB_PROJECT = "eir_course"
POLICY_FILE = "q_table.npz"
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainArgs:
  episodes: int | None = None
  """Training episodes (default: the task's own budget)."""
  bins: tuple[int, int, int, int] | None = None
  """Bins for (alpha, alpha_dot, theta, theta_dot); try larger values to watch the table grow."""
  action_scale: float | None = None
  """Arm target range in rad (default: the task's own value)."""
  learning_rate: float = 0.1
  gamma: float = 0.99
  epsilon_start: float = 1.0
  epsilon_end: float = 0.05
  epsilon_decay_fraction: float = 0.8
  """Fraction of the episodes over which epsilon decays linearly."""
  seed: int = 0
  wandb: bool = True
  """Log to W&B (project eir_course). Use --no-wandb to train offline."""
  wandb_entity: str | None = None
  video: bool = True
  """Log a video of the final greedy policy to W&B."""
  log_root: str = "logs/qlearning"


@dataclass
class PlayArgs:
  checkpoint_file: str | None = None
  """Local q_table.npz. Default: checkpoints/<example>/q_table.npz."""
  wandb_run_path: str | None = None
  """Download q_table.npz from a W&B run, e.g. entity/eir_course/abc123."""
  episodes: int = 5
  video: bool = False
  """Render to an mp4 instead of opening the viewer."""
  video_file: str | None = None
  seed: int = 1


##
# Policy file.
##


def save_policy(
  path: Path,
  task: FurutaTask,
  q: np.ndarray,
  visits: np.ndarray,
  action_targets: np.ndarray,
) -> None:
  meta = {
    "task": task.name,
    "state_dims": [asdict(d) for d in task.state_dims],
    "action_targets": action_targets.tolist(),
  }
  np.savez_compressed(path, q=q, visits=visits, meta=json.dumps(meta))


def load_policy(path: Path) -> tuple[np.ndarray, tuple[StateDim, ...], np.ndarray]:
  data = np.load(path)
  meta = json.loads(str(data["meta"]))
  dims = tuple(StateDim(**d) for d in meta["state_dims"])
  return data["q"], dims, np.asarray(meta["action_targets"])


##
# W&B images, drawn straight from NumPy (no plotting library).
##

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
  _, _, i_theta, i_theta_dot = disc.bins(np.zeros(4))
  q4 = q.reshape(*disc.shape, q.shape[-1])[:, :, i_theta, i_theta_dot]
  seen = visits.reshape(disc.shape)[:, :, i_theta, i_theta_dot] > 0
  return {
    "value": _colorize(q4.max(axis=-1), seen),
    "greedy_action": _colorize(q4.argmax(axis=-1).astype(float), seen),
  }


##
# Training.
##


def _resolve_task(task_name: str, args: TrainArgs) -> FurutaTask:
  task = TASKS[task_name]
  if args.bins is not None:
    dims = tuple(
      replace(d, bins=b) for d, b in zip(task.state_dims, args.bins, strict=True)
    )
    task = replace(task, state_dims=dims)
  if args.episodes is not None:
    task = replace(task, episodes=args.episodes)
  return task


def _record_rollout(
  env: FurutaEnv, q: np.ndarray, disc: Discretizer, path: Path
) -> bool:
  """Render one greedy episode to ``path``. Returns False if rendering is unavailable."""
  import mediapy

  try:
    renderer = mujoco.Renderer(env.model, height=480, width=640)
  except Exception as e:  # No OpenGL context (e.g. headless without EGL).
    print(f"[WARN] Skipping video: {e}")
    return False
  frames = []
  s = env.reset()
  done = False
  while not done:
    s, _, terminated, truncated = env.step(int(np.argmax(q[disc.index(s)])))
    renderer.update_scene(env.data, camera="front")
    frames.append(renderer.render())
    done = terminated or truncated
  renderer.close()
  mediapy.write_video(path, frames, fps=round(1.0 / env.control_dt))
  return True


def train(task_name: str, args: TrainArgs) -> Path:
  task = _resolve_task(task_name, args)
  env = FurutaEnv(task, action_scale=args.action_scale, seed=args.seed)
  disc = Discretizer(task.state_dims)
  rng = np.random.default_rng(args.seed)

  num_states, num_actions = disc.num_states, env.num_actions
  q = np.zeros((num_states, num_actions), dtype=np.float32)
  visits = np.zeros(num_states, dtype=np.int64)

  run_dir = (
    Path(args.log_root) / task.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  run_dir.mkdir(parents=True, exist_ok=True)

  table_info = {
    "num_states": num_states,
    "num_actions": num_actions,
    "q_entries": num_states * num_actions,
    "q_table_kib": q.nbytes / 1024,
    "state_bins": [d.bins for d in task.state_dims],
    "action_targets_rad": env.action_targets.round(3).tolist(),
  }
  print(f"[{task.name}] {task.description}")
  print(
    f"  Q-table: {num_states} states x {num_actions} actions = {num_states * num_actions} entries"
  )

  run = None
  if args.wandb:
    import wandb

    run = wandb.init(
      project=WANDB_PROJECT,
      entity=args.wandb_entity,
      name=task.name,
      group=task.name,
      config={"task": task.name, **asdict(args), **table_info},
    )
    run.summary.update(table_info)

  log_every = max(1, task.episodes // 500)
  image_every = max(1, task.episodes // 10)
  decay_episodes = max(1, int(args.epsilon_decay_fraction * task.episodes))
  window_returns, window_lengths, window_success = [], [], []
  total_steps = 0
  start = time.time()

  for episode in trange(task.episodes, desc=task.name, unit="ep"):
    frac = min(1.0, episode / decay_episodes)
    epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

    s = env.reset()
    si = disc.index(s)
    episode_return, done = 0.0, False
    while not done:
      if rng.random() < epsilon:
        a = int(rng.integers(num_actions))
      else:
        a = int(np.argmax(q[si]))
      s_next, r, terminated, truncated = env.step(a)
      si_next = disc.index(s_next)

      # Q-learning update. A terminal state has no future; a time-out does.
      target = r if terminated else r + args.gamma * q[si_next].max()
      q[si, a] += args.learning_rate * (target - q[si, a])
      visits[si] += 1

      episode_return += r
      si = si_next
      done = terminated or truncated
    total_steps += env.steps

    window_returns.append(episode_return)
    window_lengths.append(env.steps * env.control_dt)
    window_success.append(float(env.succeeded))

    if run is not None and (episode + 1) % log_every == 0:
      log = {
        "episode": episode + 1,
        "return/mean": np.mean(window_returns),
        "episode_length_s/mean": np.mean(window_lengths),
        "success_rate": np.mean(window_success),
        "epsilon": epsilon,
        "q_table/visited_fraction": np.count_nonzero(visits) / num_states,
        "env_steps": total_steps,
        "wall_clock_s": time.time() - start,
      }
      if (episode + 1) % image_every == 0:
        import wandb

        images = q_slice_images(q, visits, disc)
        caption = "rows: alpha bins, cols: alpha_dot bins; arm centred and still; grey = never visited"
        log["q_table/values"] = (
          wandb.Histogram(np_histogram=np.histogram(q[visits > 0], bins=64))
          if visits.any()
          else None
        )
        log["q_table/value_slice"] = wandb.Image(images["value"], caption=caption)
        log["q_table/greedy_action_slice"] = wandb.Image(
          images["greedy_action"], caption=caption
        )
      run.log({k: v for k, v in log.items() if v is not None}, step=episode + 1)
      window_returns, window_lengths, window_success = [], [], []

  policy_path = run_dir / POLICY_FILE
  save_policy(policy_path, task, q, visits, env.action_targets)
  visited = np.count_nonzero(visits) / num_states
  print(
    f"  Visited {visited:.1%} of the states in {total_steps} steps ({time.time() - start:.0f} s)."
  )
  print(f"  Saved {policy_path}")

  if run is not None:
    import wandb

    run.summary.update({"q_table/visited_fraction": visited, "env_steps": total_steps})
    run.save(str(policy_path), base_path=str(run_dir), policy="now")
    if args.video:
      video_path = run_dir / "greedy_rollout.mp4"
      if _record_rollout(env, q, disc, video_path):
        run.log({"greedy_rollout": wandb.Video(str(video_path), format="mp4")})
    run.finish()
  return policy_path


##
# Playing.
##


def _find_policy(task_name: str, args: PlayArgs) -> Path:
  if args.wandb_run_path:
    import wandb

    out = Path(tempfile.mkdtemp(prefix=f"{task_name}_"))
    wandb.Api().run(args.wandb_run_path).file(POLICY_FILE).download(
      root=str(out), replace=True
    )
    return out / POLICY_FILE
  if args.checkpoint_file:
    return Path(args.checkpoint_file)
  for root in (Path.cwd(), REPO_ROOT):
    candidate = root / "checkpoints" / task_name / POLICY_FILE
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
    f"No policy for {task_name}: pass --checkpoint-file or --wandb-run-path, or train it "
    f"with `uv run train {task_name}`."
  )


def play(task_name: str, args: PlayArgs) -> None:
  policy_path = _find_policy(task_name, args)
  q, dims, action_targets = load_policy(policy_path)
  task = replace(TASKS[task_name], state_dims=dims)
  env = FurutaEnv(task, seed=args.seed)
  env.action_targets = action_targets
  disc = Discretizer(dims)
  print(
    f"[{task_name}] Playing {policy_path} ({disc.num_states} states x {len(action_targets)} actions)"
  )

  def greedy(s: np.ndarray) -> int:
    return int(np.argmax(q[disc.index(s)]))

  if args.video:
    import mediapy

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []
    for _ in range(args.episodes):
      s, done = env.reset(), False
      while not done:
        s, _, terminated, truncated = env.step(greedy(s))
        renderer.update_scene(env.data, camera="front")
        frames.append(renderer.render())
        done = terminated or truncated
    renderer.close()
    out = Path(args.video_file or f"videos/{task_name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out, frames, fps=round(1.0 / env.control_dt))
    print(f"Wrote {out}")
    return

  from mujoco import viewer as mj_viewer

  with mj_viewer.launch_passive(env.model, env.data) as viewer:
    for episode in range(args.episodes):
      s, done, ret = env.reset(), False, 0.0
      terminated = False
      while not done and viewer.is_running():
        tick = time.time()
        s, r, terminated, truncated = env.step(greedy(s))
        ret += r
        done = terminated or truncated
        viewer.sync()
        time.sleep(max(0.0, env.control_dt - (time.time() - tick)))
      if not viewer.is_running():
        break
      outcome = (
        "balanced" if env.succeeded else ("fell" if terminated else "not balanced")
      )
      print(
        f"  episode {episode + 1}: return {ret:.1f}, {env.steps * env.control_dt:.1f} s, {outcome}"
      )


##
# CLI entry points (called by eir_course.scripts.train / play).
##


def train_main(task_name: str, argv: list[str]) -> None:
  args = tyro.cli(TrainArgs, args=argv, prog=f"train {task_name}")
  train(task_name, args)


def play_main(task_name: str, argv: list[str]) -> None:
  args = tyro.cli(PlayArgs, args=argv, prog=f"play {task_name}")
  play(task_name, args)
