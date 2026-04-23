import math

import torch
import pytest

from src.quantum import (
    pauli_matrices,
    target_unitary,
    step_unitaries,
    reduce_unitaries,
    evolve,
    haar_fidelity,
)


def test_pauli_algebra():
    I, X, Y, Z = pauli_matrices()
    eye = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(X @ X, eye)
    assert torch.allclose(Y @ Y, eye)
    assert torch.allclose(Z @ Z, eye)
    # XY = iZ
    assert torch.allclose(X @ Y, 1j * Z)
    assert torch.allclose(Y @ Z, 1j * X)
    assert torch.allclose(Z @ X, 1j * Y)


def test_target_unitary_known_cases():
    # alpha = 0 -> identity
    U = target_unitary(torch.tensor(0.4), torch.tensor(1.2), torch.tensor(0.0))
    assert torch.allclose(U, torch.eye(2, dtype=torch.complex64), atol=1e-6)

    # theta=pi/2, phi=0, alpha=pi -> exp(-i pi/2 X) = -i X
    U = target_unitary(torch.tensor(math.pi / 2), torch.tensor(0.0), torch.tensor(math.pi))
    _, X, _, _ = pauli_matrices()
    assert torch.allclose(U, -1j * X, atol=1e-5)

    # theta=0 -> rotation about Z by alpha: exp(-i alpha/2 Z)
    alpha = 0.7
    U = target_unitary(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(alpha))
    expected = torch.diag(torch.tensor([math.cos(alpha / 2) - 1j * math.sin(alpha / 2),
                                        math.cos(alpha / 2) + 1j * math.sin(alpha / 2)],
                                       dtype=torch.complex64))
    assert torch.allclose(U, expected, atol=1e-6)


def test_target_unitary_is_unitary_batched():
    torch.manual_seed(0)
    theta = torch.rand(64) * math.pi
    phi = torch.rand(64) * 2 * math.pi
    alpha = torch.rand(64) * 2 * math.pi
    U = target_unitary(theta, phi, alpha)
    UU = U @ U.conj().transpose(-1, -2)
    eye = torch.eye(2, dtype=UU.dtype).expand_as(UU)
    assert torch.allclose(UU, eye, atol=1e-5)
    # det = 1
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_step_unitaries_zero_delta_resonant_pulse():
    # delta=0, constant phi -> evolve by total angle T about (cos phi)X+(sin phi)Y
    L = 200
    T = 1.234
    phi_val = 0.7
    pulse = torch.full((1, L), phi_val)
    delta = torch.zeros(1)
    V = step_unitaries(pulse, delta, T)
    U = reduce_unitaries(V)

    # Closed form: exp(-i T/2 (cos phi X + sin phi Y))
    _, X, Y, _ = pauli_matrices()
    H = 0.5 * (math.cos(phi_val) * X + math.sin(phi_val) * Y)
    expected = torch.matrix_exp(-1j * T * H).unsqueeze(0)
    assert torch.allclose(U, expected, atol=1e-4)


def test_evolve_is_unitary():
    torch.manual_seed(1)
    B, M, L = 4, 3, 50
    pulse = torch.rand(B, M, L) * 2 * math.pi
    delta = torch.randn(B, M)
    U = evolve(pulse, delta, T=2.0)
    UU = U @ U.conj().transpose(-1, -2)
    eye = torch.eye(2, dtype=UU.dtype).expand_as(UU)
    assert torch.allclose(UU, eye, atol=1e-4)


def test_haar_fidelity_self_is_one():
    torch.manual_seed(2)
    theta = torch.rand(8) * math.pi
    phi = torch.rand(8) * 2 * math.pi
    alpha = torch.rand(8) * 2 * math.pi
    U = target_unitary(theta, phi, alpha)
    F = haar_fidelity(U, U)
    assert torch.allclose(F, torch.ones_like(F), atol=1e-5)


def test_haar_fidelity_random_lower_bound():
    """A truly random Haar pair has E[F] = 1/2 in d=2."""
    torch.manual_seed(3)

    def haar_su2(n):
        # Random SU(2) via Euler-like sampling that *is* Haar (not param-uniform).
        # We sample Z = N(0,I)+iN(0,I), QR, fix det=1 phase.
        Z = torch.randn(n, 2, 2, dtype=torch.cfloat) + 1j * torch.randn(n, 2, 2, dtype=torch.cfloat)
        Q, R = torch.linalg.qr(Z)
        # Make Haar by absorbing phase of diag(R)
        d = torch.diagonal(R, dim1=-2, dim2=-1)
        ph = d / d.abs()
        Q = Q * ph.unsqueeze(-2)
        det = Q[..., 0, 0] * Q[..., 1, 1] - Q[..., 0, 1] * Q[..., 1, 0]
        Q = Q * (det.conj().sqrt()).unsqueeze(-1).unsqueeze(-1)
        return Q

    A = haar_su2(20000)
    B = haar_su2(20000)
    F = haar_fidelity(A, B).mean().item()
    # Analytical Haar average for d=2 is 1/2.
    assert abs(F - 0.5) < 0.01


def test_pulse_recovers_target_no_disorder():
    """With delta=0 and a sufficiently long constant-phi pulse, we should hit
    a known SU(2) rotation about the equator."""
    L = 400
    phi_val = 1.1
    alpha = 0.9
    T = alpha  # in our units evolve applies rotation by total angle T
    pulse = torch.full((1, L), phi_val)
    U = evolve(pulse, torch.zeros(1), T=T)
    target = target_unitary(torch.tensor(math.pi / 2),
                            torch.tensor(phi_val),
                            torch.tensor(alpha)).unsqueeze(0)
    F = haar_fidelity(U, target)
    assert F.item() > 0.999
