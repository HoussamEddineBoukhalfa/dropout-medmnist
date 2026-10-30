"""Reduced CIFAR-10 with the PathMNIST split protocol.

* train : balanced subset of 4,500 images (450 per class) from the official training set,
          drawn with ``split_seed``. It is the only training regime.
* val   : 8,000 images (800 per class) from the training images not used for training.
          Tuning only. CIFAR-10 has no validation split, so this replaces PathMNIST's.
* test  : the official 10,000-image test set, loaded only by the final evaluation
          (``include_test=True``). CIFAR-10 has no second centre, so there is one test set.

Images are resized once 32x32 -> 28x28 (bilinear, antialiased) so the network is exactly the
PathMNIST one, scaled to [0, 1] and standardised per channel with statistics of the training
subset only. No other transformation is applied.

``synthetic`` gives small learnable random images in the same layout, for offline tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DataConfig

log = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@dataclass
class Split:
    x: torch.Tensor  # float32 (N, C, H, W), standardised
    y: torch.Tensor  # int64 (N,)

    def __len__(self) -> int:
        return len(self.y)

    def to(self, device: torch.device | str) -> Split:
        return Split(self.x.to(device), self.y.to(device))


@dataclass
class DatasetBundle:
    name: str
    class_names: tuple[str, ...]
    splits: dict[str, Split]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def in_channels(self) -> int:
        return self.splits[TRAIN].x.shape[1]

    @property
    def image_size(self) -> int:
        return self.splits[TRAIN].x.shape[-1]

    def to(self, device: torch.device | str) -> DatasetBundle:
        return DatasetBundle(self.name, self.class_names, {k: v.to(device) for k, v in self.splits.items()},
                             self.mean, self.std)


def balanced_indices(labels: np.ndarray, n_total: int, rng: np.random.Generator,
                     exclude: np.ndarray | None = None) -> np.ndarray:
    """Draw ``n_total / n_classes`` indices per class without replacement, skipping ``exclude``."""
    classes = np.unique(labels)
    if n_total % len(classes):
        raise ValueError(f"n={n_total} is not divisible by the {len(classes)} classes")
    per_class = n_total // len(classes)
    available = np.ones(len(labels), dtype=bool)
    if exclude is not None:
        available[exclude] = False
    chosen = []
    for c in classes:
        pool = np.flatnonzero((labels == c) & available)
        if len(pool) < per_class:
            raise ValueError(f"class {c}: {len(pool)} images available, {per_class} requested")
        chosen.append(rng.choice(pool, per_class, replace=False))
    return np.sort(np.concatenate(chosen))


def split_indices(train_labels: np.ndarray, cfg: DataConfig) -> dict[str, np.ndarray]:
    """Training subset and validation draw, both from the official training set and disjoint."""
    rng = np.random.default_rng(cfg.split_seed)
    train = balanced_indices(train_labels, cfg.n_train, rng)
    val = balanced_indices(train_labels, cfg.n_val, rng, exclude=train)
    return {TRAIN: train, VAL: val}


def _load_cifar10(cfg: DataConfig):
    from torchvision.datasets import CIFAR10

    Path(cfg.root).mkdir(parents=True, exist_ok=True)
    tr = CIFAR10(cfg.root, train=True, download=True)
    te = CIFAR10(cfg.root, train=False, download=True)
    return ((tr.data, np.asarray(tr.targets, dtype=np.int64)),
            (te.data, np.asarray(te.targets, dtype=np.int64)), CIFAR10_CLASSES)


def _load_synthetic(cfg: DataConfig, n_classes: int = 10, size: int = 32):
    """Learnable toy data: one mean colour per class plus heavy pixel noise."""
    rng = np.random.default_rng(1234)
    centres = rng.uniform(60, 195, size=(n_classes, 3))

    def make(n_per_class: int):
        labels = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
        noise = rng.normal(0, 70, size=(len(labels), size, size, 3))
        return np.clip(centres[labels][:, None, None, :] + noise, 0, 255).astype(np.uint8), labels

    per_class_pool = (cfg.n_train + cfg.n_val) // n_classes + 20
    return make(per_class_pool), make(60), tuple(f"class_{i}" for i in range(n_classes))


_LOADERS = {"cifar10": _load_cifar10, "synthetic": _load_synthetic}


def _to_tensor(imgs: np.ndarray, image_size: int, chunk: int = 4096) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW in [0, 1], resized if needed."""
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = torch.cat([
            F.interpolate(x[i:i + chunk], size=(image_size, image_size), mode="bilinear",
                          align_corners=False, antialias=True)
            for i in range(0, len(x), chunk)
        ]).clamp_(0.0, 1.0)
    return x.contiguous()


def load_dataset(cfg: DataConfig, include_test: bool = False) -> DatasetBundle:
    """Load train + val, and the test set only when ``include_test`` (final evaluation)."""
    if cfg.name not in _LOADERS:
        raise ValueError(f"unknown dataset '{cfg.name}', expected one of {sorted(_LOADERS)}")
    (train_imgs, train_labels), (test_imgs, test_labels), class_names = _LOADERS[cfg.name](cfg)
    idx = split_indices(train_labels, cfg)

    splits = {s: Split(_to_tensor(train_imgs[i], cfg.image_size), torch.from_numpy(train_labels[i]).long())
              for s, i in idx.items()}
    if include_test:
        splits[TEST] = Split(_to_tensor(test_imgs, cfg.image_size), torch.from_numpy(test_labels).long())

    mean = splits[TRAIN].x.mean(dim=(0, 2, 3))
    std = splits[TRAIN].x.std(dim=(0, 2, 3))
    for split in splits.values():
        split.x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    for name, split in splits.items():
        counts = np.bincount(split.y.numpy(), minlength=len(class_names))
        log.info("%s %-5s n=%-6d per-class min/max=%d/%d", cfg.name, name, len(split), counts.min(), counts.max())
    return DatasetBundle(cfg.name, tuple(class_names), splits, tuple(mean.tolist()), tuple(std.tolist()))
