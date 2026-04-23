"""Training loop for the encoder-decoder pulse model.

Loss
----
    L = sum_{T in T_list} E_target E_delta [ 1 - F_haar( evolve(model(angles, T), delta, T), U_target ) ]

Disorder is freshly sampled each step (Monte Carlo over delta ~ N(0,1)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import PulseModel
from .quantum import target_unitary, evolve, haar_fidelity


@dataclass
class TrainState:
    step: int
    epoch: int
    loss: float
    per_T_infidelity: dict[float, float] = field(default_factory=dict)


def _expand_pulse(pulse: Tensor, M: int) -> Tensor:
    # (B, L) -> (B, M, L)
    return pulse.unsqueeze(1).expand(-1, M, -1)


def _step_loss(model: PulseModel, angles: Tensor, M: int) -> tuple[Tensor, dict[int, float]]:
    """Compute total loss summed over T and report per-T infidelity."""
    theta, phi, alpha = angles.unbind(-1)
    U_t = target_unitary(theta, phi, alpha)               # (B, 2, 2)
    U_t = U_t.unsqueeze(1)                                # (B, 1, 2, 2)
    delta = torch.randn(angles.shape[0], M,
                        device=angles.device, dtype=angles.dtype)

    total = angles.new_zeros(())
    per_T: dict[int, float] = {}
    logit = model.encoder(angles)
    for t_idx, T in enumerate(model.T_list):
        pulse = model.decoder(logit, t_idx)               # (B, L)
        pulse_m = _expand_pulse(pulse, M)                 # (B, M, L)
        U_out = evolve(pulse_m, delta, T)                 # (B, M, 2, 2)
        F = haar_fidelity(U_out, U_t)                     # (B, M)
        infid = (1.0 - F).mean()
        total = total + infid
        per_T[t_idx] = infid.detach().item()
    return total, per_T


def _eval_loss(model: PulseModel,
               loader: Iterable,
               M: int,
               device: torch.device) -> dict[int, float]:
    model.eval()
    sums: dict[int, float] = {i: 0.0 for i in range(len(model.T_list))}
    counts = 0
    with torch.no_grad():
        for (angles,) in loader:
            angles = angles.to(device, non_blocking=True)
            theta, phi, alpha = angles.unbind(-1)
            U_t = target_unitary(theta, phi, alpha).unsqueeze(1)
            delta = torch.randn(angles.shape[0], M,
                                device=device, dtype=angles.dtype)
            logit = model.encoder(angles)
            for t_idx, T in enumerate(model.T_list):
                pulse = model.decoder(logit, t_idx)
                pulse_m = _expand_pulse(pulse, M)
                U_out = evolve(pulse_m, delta, T)
                F = haar_fidelity(U_out, U_t)
                sums[t_idx] += (1.0 - F).sum().item()
            counts += angles.shape[0] * M
    model.train()
    return {i: s / counts for i, s in sums.items()}


def train(model: PulseModel,
          train_loader: DataLoader,
          val_loader: DataLoader,
          *,
          epochs: int,
          lr: float,
          weight_decay: float,
          delta_samples: int,
          grad_clip: float,
          warmup_steps: int,
          autocast_dtype: torch.dtype | None,
          device: torch.device,
          checkpoint_path: Path,
          log_every: int = 50,
          ) -> list[TrainState]:
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
                     * 0.5 * (1.0 + torch.cos(torch.tensor(
                         3.141592653589793 * min(step, total_steps) / total_steps)).item()),
    )

    use_autocast = autocast_dtype is not None and device.type == "cuda"
    history: list[TrainState] = []
    step = 0
    best_val = float("inf")

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for (angles,) in pbar:
            angles = angles.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss, per_T = _step_loss(model, angles, delta_samples)
            else:
                loss, per_T = _step_loss(model, angles, delta_samples)

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()

            if step % log_every == 0:
                history.append(TrainState(
                    step=step, epoch=epoch, loss=loss.item(),
                    per_T_infidelity={float(model.T_list[i]): v for i, v in per_T.items()},
                ))
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
            step += 1

        # End-of-epoch eval + checkpoint.
        val = _eval_loss(model, val_loader, delta_samples, device)
        val_total = sum(val.values())
        print(f"[epoch {epoch}] val_infid_sum={val_total:.4f} per_T="
              + ", ".join(f"T={float(model.T_list[i]):.2f}:{v:.4f}"
                          for i, v in val.items()))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "T_list": list(model.T_list),
            "L": model.L,
            "epoch": epoch,
            "val_infid": val,
        }, checkpoint_path)
        if val_total < best_val:
            best_val = val_total
            torch.save({
                "model": model.state_dict(),
                "T_list": list(model.T_list),
                "L": model.L,
                "epoch": epoch,
                "val_infid": val,
            }, checkpoint_path.with_name(checkpoint_path.stem + "_best.pt"))

    return history
