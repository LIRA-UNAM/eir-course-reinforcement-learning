"""Map a continuous state to a single Q-table row index."""

from __future__ import annotations

import numpy as np

from eir_course.tasks.furuta.tasks import StateDim


def bin_edges(dim: StateDim) -> np.ndarray:
  """Interior edges of ``dim`` (``bins - 1`` values between ``low`` and ``high``)."""
  if not dim.sqrt_spacing:
    return np.linspace(dim.low, dim.high, dim.bins + 1)[1:-1]
  # Uniform in u = sign(x) * sqrt(|x| / h), mapped back to x = sign(u) * u^2 * h.
  # This assumes a symmetric range (low = -high).
  h = dim.high
  u = np.linspace(-1.0, 1.0, dim.bins + 1)[1:-1]
  return np.sign(u) * u**2 * h


class Discretizer:
  """Tabular state index from (alpha, alpha_dot, theta, theta_dot).

  Values outside a dimension's range fall into its first or last bin.
  """

  def __init__(self, dims: tuple[StateDim, ...]):
    self.dims = dims
    self.edges = [bin_edges(d) for d in dims]
    self.shape = tuple(d.bins for d in dims)
    self.num_states = int(np.prod(self.shape))

  def bins(self, state: np.ndarray) -> tuple[int, ...]:
    """Per-dimension bin indices."""
    return tuple(
      int(np.searchsorted(e, x, side="right"))
      for e, x in zip(self.edges, state, strict=True)
    )

  def index(self, state: np.ndarray) -> int:
    """Flat Q-table row for ``state``."""
    return int(np.ravel_multi_index(self.bins(state), self.shape))
