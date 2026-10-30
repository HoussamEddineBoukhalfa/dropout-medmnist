"""The stages of task C, run in order by ``scripts/run.py``.

Results layout (small text files and figures, committed):

    results/<dataset>/
        curves/      baseline learning curves + overfitting check
        sweeps/      one CSV per tuning sweep (validation only)
        selected/    resolved config of every selected method
        final/       per-seed test metrics
        figures/     PNG figures for the report
        summary.csv  summary.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, save_config, with_param
from .evaluate import evaluate_run
from .metrics import DEFAULT_ECE_BINS
from .reporting import (aggregate, markdown_table, method_colors, plot_final, plot_learning_curves,
                        plot_reliability, plot_sweeps)
from .runs import overfitting_check, run_sweep, select_best, train_run
from .utils import load_json, save_json

log = logging.getLogger(__name__)

SWEEP_PANELS = [
    {"sweep": "dropout_rate", "method": "dropout", "title": "Dropout (dense)", "xlabel": "Dropout rate p"},
    {"sweep": "dropblock_rate", "method": "dropblock", "title": "DropBlock (block 3)", "xlabel": "Drop rate p"},
]


def baseline_stage(cfg: ExperimentConfig, seed: int, out_root: str | Path, results: Path,
                   device: torch.device) -> dict:
    history = train_run(cfg, seed, out_root, device)
    check = overfitting_check(history)
    save_json(history, results / "curves" / f"{cfg.name}_seed{seed}.json")
    save_json(check, results / "curves" / f"{cfg.name}_overfitting_check.json")
    if check["overfits"]:
        log.info("overfitting check OK: val loss min at epoch %d (%.3f), last %.3f; train loss %.4f",
                 check["best_val_epoch"], check["min_val_loss"], check["last_val_loss"], check["last_train_loss"])
    else:
        log.warning("baseline does NOT clearly overfit (val loss min %.3f at epoch %d, last %.3f). "
                    "The brief requires a clear overfit before comparing methods.",
                    check["min_val_loss"], check["best_val_epoch"], check["last_val_loss"])
    return history


def tune_stage(cfg: ExperimentConfig, param: str, values: list[Any], seed: int, out_root: str | Path,
               results: Path, device: torch.device, sweep_name: str, selected_name: str) -> ExperimentConfig:
    """Grid on validation with one seed; writes the sweep CSV and the selected config."""
    sweep = run_sweep(cfg, param, values, seed, out_root, device)
    best = select_best(sweep)
    sweep["selected"] = sweep["value"] == best["value"]
    (results / "sweeps").mkdir(parents=True, exist_ok=True)
    sweep.to_csv(results / "sweeps" / f"{sweep_name}.csv", index=False)
    selected = with_param(cfg, param, best["value"], name=selected_name)
    save_config(selected, results / "selected" / f"{selected_name}.yaml")
    log.info("[%s] selected %s=%s (val acc %.4f)\n%s", sweep_name, param, best["value"], best["val_acc"],
             sweep[["value", "val_acc", "val_loss", "min_val_loss", "diverged"]].to_string(index=False))
    return selected


def final_stage(methods: dict[str, ExperimentConfig], seeds: list[int], out_root: str | Path,
                device: torch.device) -> None:
    for cfg in methods.values():
        for seed in seeds:
            train_run(cfg, seed, out_root, device)


def evaluate_stage(methods: dict[str, ExperimentConfig], seeds: list[int], out_root: str | Path, results: Path,
                   device: torch.device, mc_samples: int = 0, n_bins: int = DEFAULT_ECE_BINS,
                   force: bool = False) -> pd.DataFrame:
    rows = []
    for method, cfg in methods.items():
        for seed in seeds:
            row = evaluate_run(cfg, seed, out_root, device, mc_samples=mc_samples, n_bins=n_bins, force=force)
            rows.append({"method": method, "dataset": cfg.data.name, **row})
    per_seed = pd.DataFrame(rows)
    (results / "final").mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(results / "final" / "per_seed.csv", index=False)
    return per_seed


def _pooled_probs(per_seed: pd.DataFrame, method: str, out_root: str | Path, key: str):
    probs, labels = [], []
    for _, row in per_seed[per_seed["method"] == method].iterrows():
        path = Path(out_root) / "runs" / row["dataset"] / row["run_id"] / f"seed{row['seed']}" / "probs_test.npz"
        if not path.exists():
            return None
        data = np.load(path)
        if key not in data:
            return None
        probs.append(data[key].astype(np.float64))
        labels.append(data["y"])
    return (np.concatenate(probs), np.concatenate(labels)) if probs else None


def build_report(results: Path, out_root: str | Path, baseline: str = "baseline") -> pd.DataFrame:
    """Summary table and figures from what the earlier stages wrote to ``results``."""
    per_seed = pd.read_csv(results / "final" / "per_seed.csv")
    methods = list(dict.fromkeys(per_seed["method"]))
    summary = aggregate(per_seed, baseline=baseline)
    summary.to_csv(results / "summary.csv", index=False)
    (results / "summary.md").write_text(markdown_table(summary, baseline=baseline))
    figures = results / "figures"

    curve = results / "curves" / f"{baseline}_seed0.json"
    base_val = None
    if curve.exists():
        history = load_json(curve)
        base_val = history["val_acc"][-1]
        plot_learning_curves(history, figures / "baseline_learning_curves.png",
                             title=f"Baseline ({per_seed['dataset'].iloc[0]})")

    colors = method_colors(methods)
    panels = []
    for panel in SWEEP_PANELS:
        path = results / "sweeps" / f"{panel['sweep']}.csv"
        if path.exists():
            df = pd.read_csv(path)
            panels.append({"title": panel["title"], "xlabel": panel["xlabel"], "sweep": df,
                           "selected": df.loc[df["selected"], "value"].iloc[0],
                           "color": colors.get(panel["method"], "#2a78d6"), "baseline_val_acc": base_val})
    if panels:
        plot_sweeps(panels, figures / "tuning_sweeps.png")

    plot_final(per_seed, methods, ["test_acc", "gen_gap", "test_ece"], figures / "final_metrics.png")

    for key, name, suffix in (("probs", "reliability.png", ""), ("mc_probs", "reliability_mc.png", " (MC dropout)")):
        curves = {m: c for m in methods if (c := _pooled_probs(per_seed, m, out_root, key)) is not None}
        if curves:
            plot_reliability(curves, methods, figures / name, n_bins=DEFAULT_ECE_BINS, title_suffix=suffix)

    log.info("report written to %s\n%s", results, (results / "summary.md").read_text())
    return summary
