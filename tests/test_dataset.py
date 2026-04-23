import math
import torch

from src.dataset import sample_angles, make_dataset, split_dataset


def test_sample_ranges():
    g = torch.Generator().manual_seed(0)
    a = sample_angles(50_000, g)
    assert a.shape == (50_000, 3)
    assert (a[:, 0] >= 0).all() and (a[:, 0] <= math.pi).all()
    assert (a[:, 1] >= 0).all() and (a[:, 1] <= 2 * math.pi).all()
    assert (a[:, 2] >= 0).all() and (a[:, 2] <= 2 * math.pi).all()


def test_sample_marginals_uniform():
    g = torch.Generator().manual_seed(1)
    a = sample_angles(200_000, g)
    # Mean should approach midpoint of each range.
    means = a.mean(dim=0)
    assert abs(means[0].item() - math.pi / 2) < 0.02
    assert abs(means[1].item() - math.pi) < 0.04
    assert abs(means[2].item() - math.pi) < 0.04


def test_split_disjoint():
    g = torch.Generator().manual_seed(2)
    ds = make_dataset(1000, g)
    train, val = split_dataset(ds, val_fraction=0.1, generator=g)
    assert len(train) + len(val) == 1000
    assert len(val) == 100
