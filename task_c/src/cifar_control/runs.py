"""Run directories, caching, sweeps and selection on validation.

A run is (config identity, seed) and lives in ``<out>/runs/<dataset>/<run_id>/seed<k>/``:

    config.yaml   resolved config
    meta.json     parameter count, device, versions, git commit, wall time
    history.json  per-epoch losses and accuracies
    model.pt      last-epoch weights (written last: its presence marks a finished run)

Finished runs are never retrained, so re-running a script resumes where it stopped
(useful when Colab disconnects) and identical configs from different stages share one run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import DataConfig, ExperimentConfig, save_config, with_param
from .data import TRAIN, VAL, DatasetBundle, load_dataset
from .models import SmallCNN, build_model, count_parameters
from .train import train_model
from .utils import environment, load_json, save_json, set_seed

log = logging.getLogger(__name__)

_DATA_CACHE: dict[tuple, DatasetBundle] = {}


def get_data(cfg: DataConfig, device: torch.device, include_test: bool = False) -> DatasetBundle:
    """Load a dataset once per process and keep it on ``device``."""
    key = (json.dumps(asdict(cfg), sort_keys=True), include_test, str(device))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_dataset(cfg, include_test=include_test).to(device)
    return _DATA_CACHE[key]


def run_dir(out_root: str | Path, cfg: ExperimentConfig, seed: int) -> Path:
    return Path(out_root) / "runs" / cfg.data.name / cfg.run_id / f"seed{seed}"


def is_finished(path: Path) -> bool:
    return (path / "model.pt").exists() and (path / "history.json").exists()


def make_model(cfg: ExperimentConfig, bundle: DatasetBundle) -> SmallCNN:
    return build_model(cfg.model, bundle.in_channels, bundle.n_classes, bundle.image_size)


def train_run(cfg: ExperimentConfig, seed: int, out_root: str | Path, device: torch.device) -> dict:
    """Train (or load the cached history of) one run. Uses train and val only."""
    path = run_dir(out_root, cfg, seed)
    if is_finished(path):
        log.info("[%s seed %d] cached run %s", cfg.name, seed, cfg.run_id)
        return load_json(path / "history.json")

    bundle = get_data(cfg.data, device)
    set_seed(seed)
    model = make_model(cfg, bundle).to(device)
    log.info("[%s seed %d] training run %s (%d parameters)", cfg.name, seed, cfg.run_id, count_parameters(model))
    history = train_model(model, bundle.splits[TRAIN], bundle.splits[VAL], cfg.train, seed)

    path.mkdir(parents=True, exist_ok=True)
    save_config(cfg, path / "config.yaml")
    save_json({"run_id": cfg.run_id, "seed": seed, "name": cfg.name, "parameters": count_parameters(model),
               "device": str(device), "seconds": history["seconds"], **environment()}, path / "meta.json")
    save_json(history, path / "history.json")
    torch.save(model.state_dict(), path / "model.pt")
    return history


def load_trained(cfg: ExperimentConfig, seed: int, out_root: str | Path, bundle: DatasetBundle,
                 device: torch.device) -> SmallCNN:
    path = run_dir(out_root, cfg, seed)
    if not is_finished(path):
        raise FileNotFoundError(f"run {cfg.run_id} seed {seed} is not trained yet ({path})")
    model = make_model(cfg, bundle).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    return model.eval()


def run_sweep(cfg: ExperimentConfig, param: str, values: list[Any], seed: int, out_root: str | Path,
              device: torch.device) -> pd.DataFrame:
    """One run per value with the tuning seed; scores are last-epoch validation metrics."""
    rows = []
    for value in values:
        run_cfg = with_param(cfg, param, value)
        history = train_run(run_cfg, seed, out_root, device)
        rows.append({
            "experiment": cfg.name, "param": param, "value": value, "run_id": run_cfg.run_id, "seed": seed,
            "val_acc": history["val_acc"][-1], "val_loss": history["val_loss"][-1],
            "min_val_loss": min(history["val_loss"]), "train_loss": history["train_loss"][-1],
            "diverged": history["diverged"],
        })
    return pd.DataFrame(rows)


def select_best(sweep: pd.DataFrame) -> pd.Series:
    """Highest last-epoch validation accuracy; ties broken by lower validation loss."""
    ok = sweep[~sweep["diverged"]]
    if ok.empty:
        raise RuntimeError(f"every run of sweep '{sweep['experiment'].iloc[0]}' diverged")
    return ok.sort_values(["val_acc", "val_loss"], ascending=[False, True]).iloc[0]


def overfitting_check(history: dict) -> dict[str, Any]:
    """The baseline must clearly overfit: validation loss goes back up while training loss keeps falling."""
    val_loss, train_loss = history["val_loss"], history["train_loss"]
    best_epoch = int(min(range(len(val_loss)), key=val_loss.__getitem__)) + 1
    rise = val_loss[-1] - min(val_loss)
    return {
        "best_val_epoch": best_epoch, "min_val_loss": min(val_loss), "last_val_loss": val_loss[-1],
        "val_loss_rise": rise, "last_train_loss": train_loss[-1],
        "train_acc_last": history["train_acc"][-1], "val_acc_last": history["val_acc"][-1],
        "overfits": bool(rise > 0.1 * min(val_loss) and train_loss[-1] < train_loss[best_epoch - 1]),
    }
