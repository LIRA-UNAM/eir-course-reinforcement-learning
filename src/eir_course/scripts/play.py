"""``uv run play <example>``.

example1 and example2 (Furuta pendulum) run the greedy Q-table policy in the
MuJoCo viewer. Every other task ID runs mjlab's play script with the Viser
viewer by default (``--viewer native`` for a desktop window), which adds an
interactive velocity-command joystick for the K1 examples.

Policy source, in order: ``--wandb-run-path``, ``--checkpoint-file``, then the
pretrained checkpoint shipped in ``checkpoints/<example>/``.
"""

import sys
from pathlib import Path

from eir_course.tasks.furuta.tasks import TASKS as FURUTA_TASKS

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_NAME = "model.pt"


def _has_flag(args: list[str], *flags: str) -> bool:
  return any(a == f or a.startswith(f + "=") for a in args for f in flags)


def _shipped_checkpoint(task_id: str) -> Path | None:
  for root in (Path.cwd(), REPO_ROOT):
    path = root / "checkpoints" / task_id / CHECKPOINT_NAME
    if path.exists():
      return path
  return None


def _play_mjlab(task_id: str, args: list[str]) -> None:
  import mjlab
  import tyro
  from mjlab.scripts.play import PlayConfig, run_play

  dummy_agent = any(a in ("zero", "random") for a in args)
  has_source = _has_flag(
    args,
    "--wandb-run-path",
    "--wandb_run_path",
    "--checkpoint-file",
    "--checkpoint_file",
  )
  if not dummy_agent and not has_source:
    checkpoint = _shipped_checkpoint(task_id)
    if checkpoint is not None:
      print(f"[INFO] Using the pretrained checkpoint {checkpoint}")
      args = [*args, "--checkpoint-file", str(checkpoint)]

  cfg = tyro.cli(
    PlayConfig,
    args=args,
    default=PlayConfig(viewer="viser"),
    prog=f"play {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(task_id, cfg)


def main() -> None:
  if len(sys.argv) > 1 and sys.argv[1] in FURUTA_TASKS:
    from eir_course.tasks.furuta.q_learning import play_main

    play_main(sys.argv[1], sys.argv[2:])
    return

  import mjlab.tasks  # noqa: F401  (populates the task registry)
  from mjlab.scripts._cli import maybe_print_top_level_help
  from mjlab.tasks.registry import list_tasks

  maybe_print_top_level_help("play")
  tasks = list_tasks()
  if len(sys.argv) < 2 or sys.argv[1] not in tasks:
    names = ", ".join(
      sorted(
        [*FURUTA_TASKS, *(t for t in tasks if t.startswith("example"))],
        key=lambda s: int(s[7:]),
      )
    )
    sys.exit(f"usage: play <task> [options]\n  course examples: {names}")
  _play_mjlab(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
  main()
