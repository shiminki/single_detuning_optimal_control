"""Driver: train the encoder-decoder pulse model and produce a fidelity plot."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils import Config, seed_everything, pick_device, autocast_dtype
from src.dataset import make_dataset, split_dataset
from src.model import PulseModel
from src.trainer import train
from src.evaluator import evaluate_fidelity, plot_fidelity_vs_runtime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--name", type=str, default="run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)

    seed_everything(int(cfg["seed"]))
    device = pick_device(cfg["device"])
    print(f"device = {device}")

    # Dataset.
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    ds = make_dataset(int(cfg["dataset"]["size"]), g)
    train_ds, val_ds = split_dataset(
        ds, val_fraction=float(cfg["dataset"]["val_fraction"]), generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    # Model.
    L = int(cfg["model"]["L"])
    T_list = [float(t) * math.pi for t in cfg["model"]["T_list_pi"]]
    model = PulseModel(L=L, T_list=T_list).to(device)
    print(f"model: L={L}, T_list (in pi) = {cfg['model']['T_list_pi']}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # Train.
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"{args.name}.pt"

    train(
        model,
        train_loader,
        val_loader,
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        delta_samples=int(cfg["training"]["delta_samples"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        warmup_steps=int(cfg["training"]["warmup_steps"]),
        autocast_dtype=autocast_dtype(cfg["training"]["precision"]),
        device=device,
        checkpoint_path=ckpt_path,
        log_every=int(cfg["training"]["log_every"]),
    )

    # Final evaluation + plot.
    stats = evaluate_fidelity(
        model, val_loader,
        delta_samples=int(cfg["evaluation"]["delta_samples"]),
        device=device,
    )
    for s in stats:
        print(f"T={s.T:.3f}  mean={s.mean:.4f}  min={s.min:.4f}  max={s.max:.4f}  std={s.std:.4f}")

    plot_path = Path(cfg["paths"]["plot_dir"]) / f"{args.name}_fidelity_vs_runtime.png"
    plot_fidelity_vs_runtime(stats, plot_path)
    print(f"saved plot to {plot_path}")


if __name__ == "__main__":
    main()
