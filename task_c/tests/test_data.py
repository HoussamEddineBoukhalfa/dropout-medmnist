from __future__ import annotations

import numpy as np
import pytest
import torch

from cifar_control.config import DataConfig
from cifar_control.data import TEST, TRAIN, VAL, balanced_indices, load_dataset, split_indices


def test_balanced_indices_counts_and_exclusion():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(5), 100)
    first = balanced_indices(labels, 50, rng)
    second = balanced_indices(labels, 100, rng, exclude=first)
    assert np.bincount(labels[first]).tolist() == [10] * 5
    assert np.bincount(labels[second]).tolist() == [20] * 5
    assert not set(first) & set(second)
    with pytest.raises(ValueError):
        balanced_indices(labels, 51, rng)


def test_cifar10_split_sizes_reproducible_and_disjoint():
    labels = np.repeat(np.arange(10), 5000)  # CIFAR-10: 5,000 training images per class
    cfg = DataConfig()
    a, b = split_indices(labels, cfg), split_indices(labels, cfg)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not set(a[TRAIN]) & set(a[VAL])
    assert np.bincount(labels[a[TRAIN]]).tolist() == [450] * 10
    assert np.bincount(labels[a[VAL]]).tolist() == [800] * 10


def test_standardised_with_train_stats_and_test_hidden_by_default():
    cfg = DataConfig(name="synthetic", n_train=200, n_val=100, image_size=28)
    bundle = load_dataset(cfg)
    assert set(bundle.splits) == {TRAIN, VAL}
    x = bundle.splits[TRAIN].x
    assert x.shape[1:] == (3, 28, 28)  # 32x32 resized to 28x28
    assert torch.allclose(x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(x.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-3)
    with_test = load_dataset(cfg, include_test=True)
    assert TEST in with_test.splits
    assert torch.equal(with_test.splits[TRAIN].x, bundle.splits[TRAIN].x)  # same split and stats either way
