"""Smoke test: a tiny model on a tiny dataset trains down on a single T."""

import math
import torch
from torch.utils.data import DataLoader

from src.dataset import make_dataset
from src.model import PulseModel
from src.quantum import target_unitary, evolve, haar_fidelity


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    L = 32
    T_list = [4 * math.pi]
    model = PulseModel(L=L, T_list=T_list)

    g = torch.Generator().manual_seed(0)
    ds = make_dataset(256, g)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss_fn():
        losses = []
        for (angles,) in loader:
            theta, phi, alpha = angles.unbind(-1)
            U_t = target_unitary(theta, phi, alpha)
            pulse = model(angles, 0)
            delta = torch.randn(angles.shape[0], 4)
            U_out = evolve(pulse.unsqueeze(1).expand(-1, 4, -1),
                           delta, T=T_list[0])
            F = haar_fidelity(U_out, U_t.unsqueeze(1))
            losses.append((1 - F).mean())
        return torch.stack(losses).mean()

    initial = loss_fn().item()
    for _ in range(20):
        for (angles,) in loader:
            theta, phi, alpha = angles.unbind(-1)
            U_t = target_unitary(theta, phi, alpha)
            pulse = model(angles, 0)
            delta = torch.randn(angles.shape[0], 4)
            U_out = evolve(pulse.unsqueeze(1).expand(-1, 4, -1),
                           delta, T=T_list[0])
            F = haar_fidelity(U_out, U_t.unsqueeze(1))
            loss = (1 - F).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    final = loss_fn().item()
    assert final < initial - 0.02, f"loss did not decrease enough: {initial:.4f} -> {final:.4f}"
