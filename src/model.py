"""Encoder-Decoder pulse generator phi(t) = NN(U_target).

Encoder: 3 -> 2L -> 2L -> 2L -> 2L -> 2L -> L (ReLU, no bias).
Decoder: per-T nn.Linear(L, L) head, ModuleList indexed by T-position.
The pulse is mapped to [0, 2*pi] via 2*pi*sigmoid(.).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn, Tensor


def _angle_features(angles: Tensor) -> Tensor:
    """Map (theta, phi, alpha) -> [sin theta, cos theta, sin phi, cos phi, sin alpha, cos alpha]."""
    s = torch.sin(angles)
    c = torch.cos(angles)
    # Interleave so each angle's (sin, cos) are adjacent.
    return torch.stack([s, c], dim=-1).reshape(*angles.shape[:-1], -1)


class Encoder(nn.Module):
    def __init__(self, L: int, angle_features: bool = True) -> None:
        super().__init__()
        self.L = L
        self.angle_features = angle_features
        in_dim = 6 if angle_features else 3
        h = 2 * L
        # Spec uses bias=False throughout.
        self.net = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, L, bias=False),
        )

    def forward(self, angles: Tensor) -> Tensor:
        x = _angle_features(angles) if self.angle_features else angles
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, L: int, n_runtimes: int) -> None:
        super().__init__()
        self.L = L
        self.heads = nn.ModuleList([nn.Linear(L, L) for _ in range(n_runtimes)])

    def forward(self, logit: Tensor, t_index: int) -> Tensor:
        out = self.heads[t_index](logit)
        return 2.0 * math.pi * torch.sigmoid(out)


class PulseModel(nn.Module):
    def __init__(self, L: int, T_list: Sequence[float]) -> None:
        super().__init__()
        self._L = int(L)
        self._T_list = tuple(float(t) for t in T_list)
        self.encoder = Encoder(L=self._L)
        self.decoder = Decoder(L=self._L, n_runtimes=len(self._T_list))

    @property
    def T_list(self) -> tuple[float, ...]:
        return self._T_list

    @property
    def L(self) -> int:
        return self._L

    def forward(self, angles: Tensor, t_index: int) -> Tensor:
        return self.decoder(self.encoder(angles), t_index)
