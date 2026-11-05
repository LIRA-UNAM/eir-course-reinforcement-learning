"""Map batches of continuous states to Q-table row indices."""

from __future__ import annotations

import torch


def bin_edges(
  bins: int, low: float, high: float, sqrt_spacing: bool = False
) -> torch.Tensor:
  """The ``bins - 1`` interior edges between ``low`` and ``high``.

  With ``sqrt_spacing`` the edges are uniform in u = sign(x) sqrt(|x| / high),
  which packs them densely near 0. That assumes a symmetric range (low = -high).
  """
  if not sqrt_spacing:
    return torch.linspace(low, high, bins + 1)[1:-1]
  u = torch.linspace(-1.0, 1.0, bins + 1)[1:-1]
  return torch.sign(u) * u**2 * high


class Discretizer:
  """Tabular state index for each row of a [num_envs, num_dims] state batch.

  Values outside a dimension's range fall into its first or last bin.
  """

  def __init__(
    self,
    bins: tuple[int, ...],
    low: tuple[float, ...],
    high: tuple[float, ...],
    sqrt_spacing: tuple[bool, ...],
    device: str | torch.device = "cpu",
  ):
    self.shape = tuple(bins)
    self.num_states = 1
    for b in self.shape:
      self.num_states *= b
    self.edges = [
      bin_edges(b, lo, hi, sq).to(device)
      for b, lo, hi, sq in zip(bins, low, high, sqrt_spacing, strict=True)
    ]
    # Row-major strides, so index = sum(bin_i * stride_i).
    strides = [1] * len(self.shape)
    for i in range(len(self.shape) - 2, -1, -1):
      strides[i] = strides[i + 1] * self.shape[i + 1]
    self.strides = torch.tensor(strides, device=device)

  def bins(self, states: torch.Tensor) -> torch.Tensor:
    """Per-dimension bin indices. Shape: [num_envs, num_dims]."""
    return torch.stack(
      [
        torch.bucketize(states[:, i].contiguous(), edges, right=True)
        for i, edges in enumerate(self.edges)
      ],
      dim=-1,
    )

  def index(self, states: torch.Tensor) -> torch.Tensor:
    """Flat Q-table row for each state. Shape: [num_envs]."""
    return (self.bins(states) * self.strides).sum(dim=-1)
