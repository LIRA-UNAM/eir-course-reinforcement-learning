"""EIR course on reinforcement learning.

Importing this package registers the mjlab tasks example3-example10 (K1
velocity). example1 and example2 (Furuta pendulum, tabular Q-learning) are
plain MuJoCo + NumPy and are dispatched by ``eir_course.scripts``.
"""

from eir_course.tasks.velocity.config.k1 import *  # noqa: F401, F403
