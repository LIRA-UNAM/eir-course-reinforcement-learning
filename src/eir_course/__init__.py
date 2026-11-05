"""EIR course on reinforcement learning.

Importing this package registers the course's mjlab tasks: example1-example2
(Furuta pendulum, tabular Q-learning) and example3-example10 (K1 velocity, PPO).
"""

from eir_course.tasks.furuta import *  # noqa: F401, F403
from eir_course.tasks.velocity.config.k1 import *  # noqa: F401, F403
