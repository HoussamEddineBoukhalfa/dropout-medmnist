"""Ablations around the selected configurations.

Each variant changes one thing from its reference (the configuration selected on validation), all
other settings fixed. Ablations analyse the selected methods; they are not a new tuning round. They
are therefore measured on the **validation set only**, with 3 seeds, and never touch the test set:
the final comparison with the baseline remains the one in ``summary.md``.

Caveat: the references' hyperparameter was selected on this validation set with seed 0, so their
seed-0 validation score is slightly optimistic compared with the variants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .config import ExperimentConfig, load_config, with_params
from .evaluate import evaluate_validation
from .reporting import method_colors, plot_ablations
from .runs import train_run

log = logging.getLogger(__name__)

METRICS = ["val_acc", "val_nll", "val_ece", "val_auc", "train_acc", "gen_gap_val"]
BASELINE = "baseline"


@dataclass
class Variant:
    group: str
    label: str
    cfg: ExperimentConfig
    reference: bool = False


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def apply_changes(cfg: ExperimentConfig, changes: dict[str, Any], name: str) -> ExperimentConfig:
    """Copy of ``cfg`` with several fields changed, renamed and without a sweep spec."""
    return with_params(cfg, changes, name)


def load_references(spec: dict, configs_dir: str | Path, overrides: list[str] | None = None,
                    selected_dir: str | Path | None = None) -> dict[str, ExperimentConfig]:
    """References are the configurations selected by run.py (``{selected: name}`` ->
    ``<selected_dir>/<name>.yaml``), or a base config plus fixed values (``{config: file, set: {...}}``)."""
    references = {}
    for name, ref in spec["references"].items():
        if "selected" in ref:
            if selected_dir is None:
                raise ValueError(f"reference '{name}' needs the selected configurations (selected_dir)")
            path = Path(selected_dir) / f"{ref['selected']}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run scripts/run.py first (same --results)")
            cfg = load_config(path, overrides)
        else:
            cfg = load_config(Path(configs_dir) / ref["config"], overrides)
        references[name] = apply_changes(cfg, ref.get("set", {}), name)
    return references


def build_variants(spec: dict, references: dict[str, ExperimentConfig]) -> list[Variant]:
    variants = []
    for group in spec["groups"]:
        ref = references[group["reference"]]
        variants.append(Variant(group["name"], group.get("reference_label", "selected"), ref, reference=True))
        for label, changes in group["variants"].items():
            variants.append(Variant(group["name"], label, apply_changes(ref, changes, f"{group['name']}/{label}")))
    return variants


def run_ablations(variants: list[Variant], baseline: ExperimentConfig, seeds: list[int], out_root: str | Path,
                  device: torch.device) -> pd.DataFrame:
    """Train (or reuse) every variant for every seed and collect validation metrics."""
    rows = []
    for variant in [Variant(BASELINE, BASELINE, baseline, reference=True), *variants]:
        for seed in seeds:
            train_run(variant.cfg, seed, out_root, device)
            metrics = evaluate_validation(variant.cfg, seed, out_root, device)
            rows.append({"group": variant.group, "variant": variant.label, "reference": variant.reference,
                         "run_id": variant.cfg.run_id, "seed": seed, **{m: metrics[m] for m in METRICS}})
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per variant; validation-accuracy deltas vs the group reference and the baseline."""
    grouped = per_seed.groupby(["group", "variant"], sort=False)
    out = pd.concat({"mean": grouped[METRICS].mean(), "std": grouped[METRICS].std(ddof=1)}, axis=1)
    out.columns = [f"{metric}_{stat}" for stat, metric in out.columns]
    out["n_seeds"] = grouped.size()
    out["reference"] = grouped["reference"].first()
    out = out.reset_index()

    base = out[out["group"] == BASELINE].iloc[0]
    out["delta_vs_baseline"] = out["val_acc_mean"] - base["val_acc_mean"]
    refs = out[out["reference"] & (out["group"] != BASELINE)].set_index("group")
    ref_mean = out["group"].map(refs["val_acc_mean"])
    ref_std = out["group"].map(refs["val_acc_std"])
    out["delta_vs_reference"] = out["val_acc_mean"] - ref_mean
    noise = np.fmax(out["val_acc_std"], ref_std)
    out["conclusive_vs_reference"] = (out["delta_vs_reference"].abs() > noise) & ~out["reference"]
    return out


def _pm(mean: float, std: float, scale: float = 100.0) -> str:
    return f"{mean * scale:.2f}" + ("" if pd.isna(std) else f" ± {std * scale:.2f}")


def markdown(summary: pd.DataFrame, spec: dict) -> str:
    base = summary[summary["group"] == BASELINE].iloc[0]
    lines = ["# Ablations (validation set, reduced CIFAR-10)", "",
             f"Baseline: validation accuracy {_pm(base['val_acc_mean'], base['val_acc_std'])} %, "
             f"ECE {_pm(base['val_ece_mean'], base['val_ece_std'])} %.", ""]
    for group in spec["groups"]:
        rows = summary[summary["group"] == group["name"]]
        lines += [f"## {group['title']}", "",
                  "| Variant | Val acc (%) | Δ vs selected (conclusive?) | Δ vs baseline (pp) | Gen. gap (pp) | ECE (%) |",
                  "|---|---|---|---|---|---|"]
        for _, r in rows.iterrows():
            delta = "-" if r["reference"] else (f"{r['delta_vs_reference'] * 100:+.2f} pp "
                                                f"({'yes' if r['conclusive_vs_reference'] else 'no'})")
            lines.append(f"| {r['variant']} | {_pm(r['val_acc_mean'], r['val_acc_std'])} | {delta} | "
                         f"{r['delta_vs_baseline'] * 100:+.2f} | {_pm(r['gen_gap_val_mean'], r['gen_gap_val_std'])} | "
                         f"{_pm(r['val_ece_mean'], r['val_ece_std'])} |")
        lines.append("")
    lines += ["Mean ± sample std over seeds, last-epoch models, validation set (the test set is not used). "
              "Conclusive when |Δ| > max(std of the variant, std of the selected configuration). "
              "The selected configurations were chosen on this validation set with seed 0, so their score is "
              "slightly optimistic.", ""]
    return "\n".join(lines)


def write_report(per_seed: pd.DataFrame, spec: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "per_seed.csv", index=False)
    summary = summarize(per_seed)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "ablations.md").write_text(markdown(summary, spec))
    colors = method_colors([BASELINE, *spec["references"]])
    panels = [{"title": g["title"], "per_seed": per_seed[per_seed["group"] == g["name"]],
               "color": colors[g["reference"]]} for g in spec["groups"]]
    base = per_seed[per_seed["group"] == BASELINE]["val_acc"].mean()
    plot_ablations(panels, base, out_dir / "ablations.png")
    return summary