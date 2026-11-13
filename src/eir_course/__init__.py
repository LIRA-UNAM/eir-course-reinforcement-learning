"""EIR course on reinforcement learning.

Importing this package registers the course's mjlab tasks: example1-example2
(Furuta pendulum, tabular Q-learning) and example3-example10 (K1 velocity, PPO).

mjlab's ``play`` script imports this package (via the ``mjlab.tasks`` entry
point) before parsing the command line, so this is also where the course
defaults for ``play`` are applied: ``--checkpoint``, ``models/<example>.pt``
and the Viser viewer (see ``eir_course/scripts/play.py``). It is a no-op for
every other program.
"""

from eir_course.scripts.play import patch_play_argv as _patch_play_argv
from eir_course.tasks.furuta import *  # noqa: F401, F403
from eir_course.tasks.velocity.config.k1 import *  # noqa: F401, F403

_patch_play_argv()
