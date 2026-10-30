"""Final evaluation on the CIFAR-10 test set.

The test set is used once, at the end: results are written to ``test_metrics.json`` in the run
directory and reused afterwards (``force=True`` recomputes). Class probabilities are saved next to
them (``probs_test.npz``) so reliability diagrams never need the model again.

"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig
from .data import TEST, TRAIN, VAL
from .metrics import DEFAULT_ECE_BINS, all_metrics
from .models import STOCHASTIC_LAYERS
from .runs import get_data, load_trained, run_dir
from .utils import load_json, save_json

log = logging.getLogger(__name__)


@torch.no_grad()
def _softmax(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.cat([F.softmax(model(x[i:i + batch_size]).double(), dim=1) for i in range(0, len(x), batch_size)])


def predict_proba(model: nn.Module, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Deterministic prediction: every stochastic layer off."""
    model.eval()
    return _softmax(model, x, batch_size).cpu().numpy()


def mc_predict_proba(model: nn.Module, x: torch.Tensor, n_samples: int, seed: int = 0,
                     batch_size: int = 1024) -> np.ndarray:
    """Monte Carlo dropout (Gal & Ghahramani, 2016): dropout-type layers stay active at test time
    and the softmax is averaged over ``n_samples`` passes. Everything else stays in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, STOCHASTIC_LAYERS):
            module.train()
    torch.manual_seed(seed)
    total = None
    for _ in range(n_samples):
        probs = _softmax(model, x, batch_size)
        total = probs if total is None else total + probs
    model.eval()
    return (total / n_samples).cpu().numpy()


def evaluate_run(cfg: ExperimentConfig, seed: int, out_root: str | Path, device: torch.device,
                 mc_samples: int = 0, n_bins: int = DEFAULT_ECE_BINS, force: bool = False) -> dict:
    path = run_dir(out_root, cfg, seed)
    cache = path / "test_metrics.json"
    if cache.exists() and not force:
        cached = load_json(cache)
        if cached.get("mc_samples", 0) == mc_samples and cached.get("ece_bins") == n_bins:
            return cached

    bundle = get_data(cfg.data, device, include_test=True)
    model = load_trained(cfg, seed, out_root, bundle, device)
    row: dict = {"run_id": cfg.run_id, "seed": seed, "mc_samples": mc_samples, "ece_bins": n_bins}
    for split in (TRAIN, VAL):
        data = bundle.splits[split]
        row[f"{split}_acc"] = float((predict_proba(model, data.x).argmax(1) == data.y.cpu().numpy()).mean())

    test = bundle.splits[TEST]
    y = test.y.cpu().numpy()
    probs = predict_proba(model, test.x)
    row.update({f"test_{k}": v for k, v in all_metrics(probs, y, n_bins).items()})
    arrays = {"probs": probs.astype(np.float32), "y": y}
    if mc_samples > 0 and model.has_stochastic_layers():
        mc_probs = mc_predict_proba(model, test.x, mc_samples, seed=seed)
        row.update({f"test_mc_{k}": v for k, v in all_metrics(mc_probs, y, n_bins).items()})
        arrays["mc_probs"] = mc_probs.astype(np.float32)
    np.savez_compressed(path / "probs_test.npz", **arrays)

    row["gen_gap"] = row["train_acc"] - row["test_acc"]
    save_json(row, cache)
    log.info("[%s seed %d] test evaluated: acc %.4f", cfg.name, seed, row["test_acc"])
    return row
