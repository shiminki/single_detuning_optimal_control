"""Small helpers: config loading, seeding, device selection, autocast dtype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random
import yaml
import numpy as np
import torch


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            return cls(yaml.safe_load(f))

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None
