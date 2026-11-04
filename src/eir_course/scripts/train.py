"""``uv run train <example>``.

example1 and example2 (Furuta pendulum) train with tabular Q-learning on the
CPU. Every other task ID goes to mjlab's own training script unchanged, so
``uv run train example3 --env.scene.num-envs 4096`` behaves exactly like mjlab.
"""

import sys

from eir_course.tasks.furuta.tasks import TASKS as FURUTA_TASKS


def main() -> None:
  if len(sys.argv) > 1 and sys.argv[1] in FURUTA_TASKS:
    from eir_course.tasks.furuta.q_learning import train_main

    train_main(sys.argv[1], sys.argv[2:])
    return

  from mjlab.scripts.train import main as mjlab_train

  mjlab_train()


if __name__ == "__main__":
  main()
