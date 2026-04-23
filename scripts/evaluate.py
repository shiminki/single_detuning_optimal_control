"""Driver: load a trained model and re-produce the fidelity-vs-runtime plot."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils import Config, seed_everything, pick_device
from src.dataset import make_dataset
from src.model import PulseModel
from src.evaluator import evaluate_fidelity, plot_fidelity_vs_runtime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--name", type=str, default="eval")
    p.add_argument("--n-samples", type=int, default=20000,
                   help="Number of evaluation targets to draw fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    seed_everything(int(cfg["seed"]))
    device = pick_device(cfg["device"])

    ckpt = torch.load(args.checkpoint, map_location=device)
    L = int(ckpt["L"])
    T_list = list(ckpt["T_list"])
    model = PulseModel(L=L, T_list=T_list).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint epoch={ckpt.get('epoch')} L={L} T_list={T_list}")

    g = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    ds = make_dataset(args.n_samples, g)
    loader = DataLoader(ds,
                        batch_size=int(cfg["evaluation"]["batch_size"]),
                        shuffle=False)

    stats = evaluate_fidelity(
        model, loader,
        delta_samples=int(cfg["evaluation"]["delta_samples"]),
        device=device,
    )
    for s in stats:
        print(f"T={s.T:.3f} ({s.T/math.pi:.2f}π)  "
              f"mean={s.mean:.4f}  min={s.min:.4f}  max={s.max:.4f}  std={s.std:.4f}")

    plot_path = Path(cfg["paths"]["plot_dir"]) / f"{args.name}_fidelity_vs_runtime.png"
    plot_fidelity_vs_runtime(stats, plot_path)
    print(f"saved plot to {plot_path}")


if __name__ == "__main__":
    main()
