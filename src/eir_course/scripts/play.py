"""Course defaults for ``uv run play <example>``.

``play`` is mjlab's own script (``mjlab.scripts.play``). mjlab imports this
package through its ``mjlab.tasks`` entry point before it parses the command
line, and ``eir_course/__init__.py`` then calls :func:`patch_play_argv`, which
adjusts ``sys.argv`` in place:

- ``--checkpoint X`` is a short form of ``--checkpoint-file X``, so course
  attendees can run ``uv run play exampleN --checkpoint models/exampleN.pt``.
- With no ``--wandb-run-path`` / ``--checkpoint-file`` (and a trained agent),
  the pretrained ``models/<example>.pt`` is loaded.
- The Viser viewer is the default (``--viewer native`` for a desktop window);
  it adds an interactive velocity-command joystick for the K1 examples.

Policy source, in order: ``--wandb-run-path``, ``--checkpoint`` /
``--checkpoint-file``, then ``models/<example>.pt``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = "models"
_HELP = ("-h", "--help")


def _has_flag(args: list[str], *flags: str) -> bool:
  return any(a == f or a.startswith(f + "=") for a in args for f in flags)


def _expand_checkpoint_alias(args: list[str]) -> list[str]:
  """Rewrite ``--checkpoint X`` / ``--checkpoint=X`` to ``--checkpoint-file X``."""
  out: list[str] = []
  for a in args:
    if a == "--checkpoint":
      out.append("--checkpoint-file")
    elif a.startswith("--checkpoint="):
      out.append("--checkpoint-file=" + a.split("=", 1)[1])
    else:
      out.append(a)
  return out


def _shipped_checkpoint(task_id: str) -> Path | None:
  for root in (Path.cwd(), REPO_ROOT):
    path = root / MODELS_DIR / f"{task_id}.pt"
    if path.exists():
      return path
  return None


def course_play_args(task_id: str, args: list[str]) -> list[str]:
  """Return ``args`` (everything after the task name) with the course defaults."""
  args = _expand_checkpoint_alias(args)
  if any(a in _HELP for a in args):
    return args
  if not _has_flag(args, "--viewer"):
    args = [*args, "--viewer", "viser"]
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
  return args


def patch_play_argv(argv: list[str] = sys.argv) -> None:
  """Apply :func:`course_play_args` to ``argv`` in place when running ``play``."""
  if len(argv) < 2 or Path(argv[0]).name not in ("play", "play.exe"):
    return
  if argv[1].startswith("-"):
    return
  argv[2:] = course_play_args(argv[1], argv[2:])


def main() -> None:
  """``python -m eir_course.scripts.play <example> [options]``."""
  from mjlab.scripts.play import main as mjlab_play

  sys.argv[0] = "play"
  patch_play_argv(sys.argv)
  mjlab_play()


if __name__ == "__main__":
  main()
