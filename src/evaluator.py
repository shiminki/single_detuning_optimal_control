"""Evaluation utilities: per-T fidelity statistics and the runtime/fidelity plot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from .model import PulseModel
from .quantum import target_unitary, evolve, haar_fidelity


@dataclass
class FidelityStats:
    T: float
    mean: float
    min: float
    max: float
    std: float


@torch.no_grad()
def evaluate_fidelity(model: PulseModel,
                      loader: DataLoader,
                      *,
                      delta_samples: int,
                      device: torch.device,
                      ) -> list[FidelityStats]:
    """Per-target fidelity is averaged over `delta_samples` δ draws first; then
    we report mean / min / max / std across the dataset, for every T."""
    model.eval().to(device)

    n_T = len(model.T_list)
    chunks: list[list[Tensor]] = [[] for _ in range(n_T)]

    for (angles,) in loader:
        angles = angles.to(device, non_blocking=True)
        theta, phi, alpha = angles.unbind(-1)
        U_t = target_unitary(theta, phi, alpha).unsqueeze(1)            # (B, 1, 2, 2)
        delta = torch.randn(angles.shape[0], delta_samples,
                            device=device, dtype=angles.dtype)
        logit = model.encoder(angles)
        for t_idx, T in enumerate(model.T_list):
            pulse = model.decoder(logit, t_idx)
            pulse_m = pulse.unsqueeze(1).expand(-1, delta_samples, -1)
            U_out = evolve(pulse_m, delta, T)
            F = haar_fidelity(U_out, U_t).mean(dim=1)                    # (B,)
            chunks[t_idx].append(F.detach().cpu())

    out: list[FidelityStats] = []
    for t_idx, T in enumerate(model.T_list):
        f = torch.cat(chunks[t_idx])
        out.append(FidelityStats(
            T=float(T),
            mean=f.mean().item(),
            min=f.min().item(),
            max=f.max().item(),
            std=f.std().item(),
        ))
    return out


def plot_fidelity_vs_runtime(stats: list[FidelityStats], out_path: Path) -> None:
    Ts = [s.T for s in stats]
    means = [s.mean for s in stats]
    mins = [s.min for s in stats]
    maxs = [s.max for s in stats]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(Ts, mins, maxs, alpha=0.20, label="min/max range")
    ax.plot(Ts, means, "o-", linewidth=2, label="mean")
    ax.plot(Ts, mins, "v--", alpha=0.7, label="min")
    ax.plot(Ts, maxs, "^--", alpha=0.7, label="max")

    ax.set_xlabel(r"Pulse runtime $T \cdot \Omega$")
    ax.set_ylabel("Haar-averaged fidelity")
    ax.set_title("Fidelity vs runtime (single-qubit, strong disorder)")
    ax.set_ylim(min(mins) - 0.02, 1.005)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Secondary x-axis showing T in units of pi.
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / math.pi, lambda x: x * math.pi),
    )
    ax2.set_xlabel(r"$T / \pi$")

    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
