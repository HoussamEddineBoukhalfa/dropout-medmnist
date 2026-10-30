from __future__ import annotations

import pytest

from cifar_control.config import ExperimentConfig, from_dict


@pytest.fixture
def tiny_cfg() -> ExperimentConfig:
    """Synthetic data, 3 epochs: fast enough to run on CPU."""
    return from_dict({
        "name": "baseline",
        "data": {"name": "synthetic", "n_train": 200, "n_val": 100, "image_size": 28},
        "train": {"epochs": 3, "batch_size": 64, "lr": 0.02},
    })
