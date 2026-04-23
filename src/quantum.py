"""Quantum primitives: Pauli ops, target SU(2), pulse evolution, fidelity.

Conventions
-----------
* All operators are 2x2 complex matrices stored as torch.complex64 tensors.
* `phi` (lowercase) refers to the control phase, never the SU(2) azimuthal angle.
* Disorder is the dimensionless `delta = delta_phys / Omega`, drawn ~ N(0,1).
* Pulse vectors have shape (..., L) of real dtype with values in [0, 2*pi].
* Time `T` is dimensionless: T_phys * Omega.
"""

from __future__ import annotations

import torch
from torch import Tensor


def pauli_matrices(device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.complex64
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    I = torch.eye(2, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return I, X, Y, Z


def target_unitary(theta: Tensor, phi: Tensor, alpha: Tensor) -> Tensor:
    """U = cos(alpha/2) I - i sin(alpha/2) (sin theta cos phi X + sin theta sin phi Y + cos theta Z)."""
    theta = torch.as_tensor(theta)
    phi = torch.as_tensor(phi)
    alpha = torch.as_tensor(alpha)

    c = torch.cos(alpha / 2)
    s = torch.sin(alpha / 2)
    nx = torch.sin(theta) * torch.cos(phi)
    ny = torch.sin(theta) * torch.sin(phi)
    nz = torch.cos(theta)

    # 2x2 matrix written component-wise.
    U00 = c - 1j * s * nz
    U01 = -1j * s * (nx - 1j * ny)
    U10 = -1j * s * (nx + 1j * ny)
    U11 = c + 1j * s * nz

    U = torch.stack([torch.stack([U00, U01], dim=-1),
                     torch.stack([U10, U11], dim=-1)], dim=-2)
    return U.to(torch.complex64)


def step_unitaries(pulse: Tensor, delta: Tensor, T: float) -> Tensor:
    """V_i = cos(a) I - i sin(a)/r * (cos phi_i X + sin phi_i Y + delta Z),

    where r = sqrt(1 + delta^2) and a = r * dt / 2, dt = T / L.
    """
    L = pulse.shape[-1]
    dt = T / L

    # Broadcast delta against pulse: pulse is (..., L), delta is (...,) or scalar.
    delta_b = torch.as_tensor(delta, dtype=pulse.dtype, device=pulse.device)
    # ensure delta has same leading dims as pulse[..., 0]
    while delta_b.dim() < pulse.dim() - 1:
        delta_b = delta_b.unsqueeze(-1)
    delta_b = delta_b.unsqueeze(-1)  # (..., 1) so it broadcasts over L

    r = torch.sqrt(1.0 + delta_b ** 2)             # (..., 1)
    a = r * (dt / 2.0)                             # (..., 1)
    cos_a = torch.cos(a)                           # (..., 1)
    sin_over_r = torch.sin(a) / r                  # (..., 1)

    cos_phi = torch.cos(pulse)                     # (..., L)
    sin_phi = torch.sin(pulse)                     # (..., L)

    cos_a_b = cos_a.expand_as(pulse)               # (..., L)
    sin_or_b = sin_over_r.expand_as(pulse)         # (..., L)
    delta_l = delta_b.expand_as(pulse)             # (..., L)

    # Components (real parts and imag parts separated to avoid spurious complex casts).
    V00_re = cos_a_b
    V00_im = -sin_or_b * delta_l
    V11_re = cos_a_b
    V11_im = sin_or_b * delta_l
    V01_re = -sin_or_b * sin_phi      # Im of (cos - i sin) -> -sin
    V01_im = -sin_or_b * cos_phi
    V10_re = sin_or_b * sin_phi       # +sin from (cos + i sin)
    V10_im = -sin_or_b * cos_phi

    V00 = torch.complex(V00_re, V00_im)
    V01 = torch.complex(V01_re, V01_im)
    V10 = torch.complex(V10_re, V10_im)
    V11 = torch.complex(V11_re, V11_im)

    V = torch.stack([torch.stack([V00, V01], dim=-1),
                     torch.stack([V10, V11], dim=-1)], dim=-2)
    return V  # (..., L, 2, 2)


def reduce_unitaries(V: Tensor) -> Tensor:
    """Pairwise tree reduction: U = V_{L-1} ... V_1 V_0."""
    # Move L axis to a known position (-3).
    L = V.shape[-3]
    while L > 1:
        if L % 2 == 1:
            # Pad with identity at the end (position L), so its product with V_{L-1} is V_{L-1}.
            eye = torch.eye(2, dtype=V.dtype, device=V.device)
            pad_shape = list(V.shape)
            pad_shape[-3] = 1
            eye_pad = eye.expand(pad_shape)
            V = torch.cat([V, eye_pad], dim=-3)
            L = V.shape[-3]

        even = V[..., 0::2, :, :]   # V_0, V_2, ...
        odd = V[..., 1::2, :, :]    # V_1, V_3, ...
        V = odd @ even              # (V_1 V_0), (V_3 V_2), ...
        L = V.shape[-3]

    return V[..., 0, :, :]


def evolve(pulse: Tensor, delta: Tensor, T: float) -> Tensor:
    return reduce_unitaries(step_unitaries(pulse, delta, T))


def haar_fidelity(U_out: Tensor, U_target: Tensor) -> Tensor:
    """(|Tr(U_out^dagger U_target)|^2 + 2) / 6."""
    M = U_out.conj().transpose(-1, -2) @ U_target
    tr = M[..., 0, 0] + M[..., 1, 1]
    return (tr.real ** 2 + tr.imag ** 2 + 2.0) / 6.0
