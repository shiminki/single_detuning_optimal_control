# Single-Detuning Optimal Control

Neural-network controller `phi(t) = NN(U_target)` that synthesizes
disorder-robust single-qubit gates in the strong-disorder regime
`delta / Omega ~ N(0, 1)`.

## Problem

Hamiltonian (in units where Omega = 1):

```
H(t) = (1/2) (cos phi(t) X + sin phi(t) Y) + (delta/2) Z
```

Goal: maximize the Haar-averaged fidelity

```
F = E_{delta ~ N(0,1)} ( |Tr(U_out^† U_target)|^2 + 2 ) / 6
```

simultaneously across `T ∈ {4π, 7π, 10π, 13π, 16π, 20π}`, with one
shared encoder and per-T linear decoder heads.

## Layout

```
src/
  quantum.py    Pauli ops, target SU(2), step unitaries, log-depth product, Haar fidelity
  dataset.py    Uniform sampler over (theta, phi, alpha)
  model.py      Encoder (6-layer MLP) + per-T linear decoder heads
  trainer.py    Multi-T training loop with Monte-Carlo disorder
  evaluator.py  Per-T fidelity stats and plotting
  utils.py      Config / seeding / device helpers

scripts/
  train.py      Driver: train and produce fidelity-vs-runtime plot
  evaluate.py   Driver: load checkpoint and re-plot

tests/
  test_quantum.py    Pauli algebra, closed-form V_i, Haar mean = 1/2
  test_dataset.py    Sample ranges and marginals
  test_model.py      Forward shapes, per-T head independence
  test_pipeline.py   Smoke test: 20 steps reduces loss
```

## Usage

```bash
pip install -r requirements.txt
pytest -q                                                    # tests
python scripts/train.py --config configs/default.yaml --name run1
python scripts/evaluate.py --checkpoint outputs/checkpoints/run1_best.pt --name run1
```

`configs/default.yaml` is tuned for an A100 80 GB. For a CPU dry-run override
`device: cpu`, drop `dataset.size` to ~10⁴, and reduce `model.L`.
