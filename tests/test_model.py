import math
import torch

from src.model import Encoder, Decoder, PulseModel


def test_encoder_shape():
    enc = Encoder(L=64)
    x = torch.randn(7, 3)
    out = enc(x)
    assert out.shape == (7, 64)


def test_decoder_per_T_independence():
    dec = Decoder(L=32, n_runtimes=3)
    z = torch.randn(4, 32)
    out0 = dec(z, 0)
    out1 = dec(z, 1)
    assert out0.shape == (4, 32) and out1.shape == (4, 32)
    # Heads are independent linear layers, so outputs should differ.
    assert not torch.allclose(out0, out1)
    # Output is in [0, 2*pi].
    assert (out0 >= 0).all() and (out0 <= 2 * math.pi).all()


def test_pulse_model_encoder_shared():
    T_list = [4 * math.pi, 7 * math.pi]
    model = PulseModel(L=16, T_list=T_list)
    angles = torch.rand(5, 3)

    # Snapshot encoder params before/after a backward through one head:
    pulse0 = model(angles, 0)
    pulse1 = model(angles, 1)
    assert pulse0.shape == (5, 16)
    assert pulse1.shape == (5, 16)
    assert (pulse0 >= 0).all() and (pulse0 <= 2 * math.pi).all()

    # Encoder gradient must flow when we hit either head.
    loss = pulse0.sum()
    loss.backward()
    enc_grads_present = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()  # type: ignore[attr-defined]
    )
    assert enc_grads_present


def test_T_list_recoverable():
    T_list = [4 * math.pi, 7 * math.pi, 10 * math.pi]
    m = PulseModel(L=8, T_list=T_list)
    assert tuple(m.T_list) == tuple(T_list)
    assert m.L == 8
