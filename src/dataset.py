"""Uniform sampler / TensorDataset of SU(2) Euler angles (theta, phi, alpha)."""

from __future__ import annotations

import math
import torch
from torch.utils.data import TensorDataset, random_split


def sample_angles(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    u = torch.rand(n, 3, generator=generator)
    u[:, 0] *= math.pi
    u[:, 1] *= 2 * math.pi
    u[:, 2] *= 2 * math.pi
    return u.to(torch.float32)


def make_dataset(n: int, generator: torch.Generator | None = None) -> TensorDataset:
    return TensorDataset(sample_angles(n, generator))


def split_dataset(ds: TensorDataset,
                  val_fraction: float,
                  generator: torch.Generator | None = None
                  ) -> tuple[TensorDataset, TensorDataset]:
    n = len(ds)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    train, val = random_split(ds, [n_train, n_val], generator=generator)

    # Re-wrap into TensorDataset for cheap indexing/serialization.
    def _wrap(subset):
        idx = torch.tensor(subset.indices)
        return TensorDataset(ds.tensors[0][idx])

    return _wrap(train), _wrap(val)
