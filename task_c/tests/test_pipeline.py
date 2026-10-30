"""End to end on synthetic data: train, cache, sweep, evaluate once, report."""

from __future__ import annotations

import pandas as pd
import torch

from cifar_control.config import with_param
from cifar_control.data import TEST
from cifar_control.evaluate import evaluate_run, mc_predict_proba, predict_proba
from cifar_control.pipeline import build_report, evaluate_stage, tune_stage
from cifar_control.runs import get_data, load_trained, run_dir, train_run

CPU = torch.device("cpu")


def test_training_is_cached_and_reproducible(tiny_cfg, tmp_path):
    first = train_run(tiny_cfg, 0, tmp_path / "a", CPU)
    assert (run_dir(tmp_path / "a", tiny_cfg, 0) / "model.pt").exists()
    assert train_run(tiny_cfg, 0, tmp_path / "a", CPU) == first  # cached
    again = train_run(tiny_cfg, 0, tmp_path / "b", CPU)  # retrained from scratch, same seed
    assert again["val_loss"] == first["val_loss"]
    assert len(first["val_acc"]) == tiny_cfg.train.epochs and not first["diverged"]


def test_mc_dropout_differs_from_deterministic(tiny_cfg, tmp_path):
    cfg = with_param(tiny_cfg, "model.dropout_p", 0.5)
    train_run(cfg, 0, tmp_path, CPU)
    bundle = get_data(cfg.data, CPU, include_test=True)
    model = load_trained(cfg, 0, tmp_path, bundle, CPU)
    x = bundle.splits[TEST].x[:32]
    det, mc = predict_proba(model, x), mc_predict_proba(model, x, n_samples=5)
    assert det.shape == mc.shape and abs(det - mc).max() > 0
    assert all(not m.training for m in model.modules())  # restored to eval mode


def test_full_pipeline(tiny_cfg, tmp_path):
    out, results = tmp_path / "out", tmp_path / "results" / "synthetic"
    selected = tune_stage(tiny_cfg, "model.dropout_p", [0.1, 0.3], 0, out, results, CPU,
                          sweep_name="dropout_rate", selected_name="dropout")
    assert selected.model.dropout_p in (0.1, 0.3) and selected.sweep is None
    assert pd.read_csv(results / "sweeps" / "dropout_rate.csv")["selected"].sum() == 1

    methods = {"baseline": tiny_cfg, "dropout": selected}
    for cfg in methods.values():
        for seed in (0, 1):
            train_run(cfg, seed, out, CPU)
    per_seed = evaluate_stage(methods, [0, 1], out, results, CPU, mc_samples=3)
    assert {"test_acc", "test_ece", "test_auc", "gen_gap"} <= set(per_seed.columns)
    assert per_seed.loc[per_seed["method"] == "dropout", "test_mc_acc"].notna().all()

    # the test set is evaluated once, then read from the cache
    cached = evaluate_run(selected, 0, out, CPU, mc_samples=3)
    assert cached["test_acc"] == per_seed.query("method == 'dropout' and seed == 0")["test_acc"].item()

    summary = build_report(results, out)
    assert "test_acc_delta" in summary.columns
    for name in ("final_metrics.png", "reliability.png", "reliability_mc.png", "tuning_sweeps.png"):
        assert (results / "figures" / name).exists(), name
    assert (results / "summary.md").read_text().startswith("| Method")
