"""Summary table and figures.

Table: mean +/- sample std (ddof=1) over seeds, difference to the baseline, and a conclusiveness
flag following the brief: a difference smaller than the standard deviation does not allow a
conclusion (we require |delta| > max(std_method, std_baseline)).

Figures use one fixed colour per method (identity follows the method, never its rank), thin
marks and recessive axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .metrics import reliability_bins  # noqa: E402

# Reference palette (light mode): categorical slots in fixed order, ink and chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

METRIC_LABELS = {
    "test_acc": "Test accuracy",
    "gen_gap": "Generalisation gap (train - test)",
    "test_ece": "Expected calibration error",
    "test_auc": "Macro AUC",
}


def method_colors(methods: list[str]) -> dict[str, str]:
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "legend.labelcolor": INK_2, "lines.linewidth": 1.8, "font.family": "sans-serif",
    })


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- table


def aggregate(per_seed: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """One row per method: mean and std of every metric, plus the test-accuracy delta vs the baseline."""
    numeric = [c for c in per_seed.columns
               if c not in ("method", "seed", "run_id", "mc_samples", "ece_bins")
               and pd.api.types.is_numeric_dtype(per_seed[c])]
    grouped = per_seed.groupby("method", sort=False)[numeric]
    out = pd.concat({"mean": grouped.mean(), "std": grouped.std(ddof=1)}, axis=1)
    out.columns = [f"{metric}_{stat}" for stat, metric in out.columns]
    out.insert(0, "n_seeds", per_seed.groupby("method", sort=False).size())
    if baseline in out.index:
        for col in ("test_acc", "test_mc_acc"):
            if f"{col}_mean" not in out:
                continue
            delta = out[f"{col}_mean"] - out.loc[baseline, "test_acc_mean"]
            noise = np.maximum(out[f"{col}_std"], out.loc[baseline, "test_acc_std"])
            out[f"{col}_delta"] = delta
            out[f"{col}_conclusive"] = (delta.abs() > noise) & (out.index != baseline)
    return out.reset_index()


def _pm(row: pd.Series, col: str, scale: float = 100.0, digits: int = 2) -> str:
    mean, std = row.get(f"{col}_mean"), row.get(f"{col}_std")
    if mean is None or pd.isna(mean):
        return "-"
    std_txt = "" if pd.isna(std) else f" ± {std * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f}{std_txt}"


def _delta(row: pd.Series, col: str, baseline: str) -> str:
    if row["method"] == baseline or pd.isna(row.get(f"{col}_delta")):
        return "-"
    flag = "yes" if row[f"{col}_conclusive"] else "no"
    return f"{row[f'{col}_delta'] * 100:+.2f} pp ({flag})"


def markdown_table(summary: pd.DataFrame, baseline: str = "baseline") -> str:
    """The results table, ready to paste into the report."""
    has_mc = "test_mc_acc_mean" in summary
    header = ["Method", "Seeds", "Test acc (%)", "Δ vs baseline (conclusive?)", "Gen. gap (pp)", "ECE (%)",
              "Macro AUC"]
    if has_mc:
        header += ["MC dropout acc (%)", "MC dropout ECE (%)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in summary.iterrows():
        cells = [row["method"], str(int(row["n_seeds"])), _pm(row, "test_acc"), _delta(row, "test_acc", baseline),
                 _pm(row, "gen_gap"), _pm(row, "test_ece"), _pm(row, "test_auc", 1.0, 3)]
        if has_mc:
            cells += [_pm(row, "test_mc_acc"), _pm(row, "test_mc_ece")]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\nReduced CIFAR-10, last-epoch models. Mean ± sample std over seeds. A difference is marked "
            "conclusive when |Δ| > max(std of the method, std of the baseline).\n")
    return "\n".join(lines) + "\n" + note


# ----------------------------------------------------------------------------- figures


def plot_learning_curves(history: dict, path: str | Path, title: str = "Baseline") -> Path:
    """Loss and accuracy per epoch: the overfitting check of the baseline."""
    _style()
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, key, label in ((axes[0], "loss", "Cross-entropy"), (axes[1], "acc", "Accuracy")):
        for i, (split, name) in enumerate((("train", "train (running)"), ("val", "validation"))):
            values = history[f"{split}_{key}"]
            ax.plot(epochs, values, color=SERIES[i], label=name)
            ax.annotate(name, (epochs[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.12)
    val_loss = history["val_loss"]
    best = int(np.argmin(val_loss))
    axes[0].plot(epochs[best], val_loss[best], "o", ms=6, mfc=SURFACE, mec=SERIES[1], mew=1.6)
    axes[0].annotate(f"min val loss (epoch {epochs[best]})", (epochs[best], val_loss[best]), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    axes[0].legend(loc="upper right")
    fig.suptitle(f"{title}: training vs validation", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_sweeps(panels: list[dict], path: str | Path) -> Path:
    """Validation accuracy (last epoch) against the tuned value, one panel per method.

    Each panel dict: title, xlabel, sweep (DataFrame with value, val_acc), selected, color,
    baseline_val_acc (optional).
    """
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.3), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        df = panel["sweep"].sort_values("value")
        x, y = df["value"].astype(float).to_numpy(), df["val_acc"].to_numpy() * 100
        ax.plot(x, y, color=panel["color"], marker="o", ms=5)
        sel = df[df["value"] == panel["selected"]]
        sx, sy = float(sel["value"].iloc[0]), sel["val_acc"].iloc[0] * 100
        ax.plot(sx, sy, "o", ms=10, mfc="none", mec=panel["color"], mew=1.6)
        ax.annotate("selected", (sx, sy), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_2)
        if panel.get("baseline_val_acc") is not None:
            base = panel["baseline_val_acc"] * 100
            ax.axhline(base, color=MUTED, lw=1.2, ls=(0, (4, 3)))
            ax.annotate("baseline", (x.max(), base), xytext=(0, 4), textcoords="offset points",
                        ha="right", fontsize=8, color=MUTED)
        ax.margins(x=0.08, y=0.18)
        ax.set_title(panel["title"], loc="left")
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation accuracy, last epoch (%)")
        ax.set_xticks(x)
    fig.tight_layout()
    return _save(fig, path)


def plot_final(per_seed: pd.DataFrame, methods: list[str], metrics: list[str], path: str | Path) -> Path:
    """Dot plot per metric: each seed as a small dot, mean +/- std as a larger mark."""
    _style()
    colors = method_colors(methods)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for i, method in enumerate(methods):
            vals = per_seed.loc[per_seed["method"] == method, metric].to_numpy() * 100
            if len(vals) == 0:
                continue
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(i + jitter, vals, s=16, color=colors[method], alpha=0.45, linewidths=0, zorder=2)
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.errorbar(i + 0.3, vals.mean(), yerr=std, fmt="o", ms=6, color=colors[method],
                        ecolor=colors[method], elinewidth=1.6, capsize=0, zorder=3)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_xlim(-0.5, len(methods) - 0.3)
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left", fontsize=9.5)
        ax.set_ylabel("percentage points" if metric.endswith("_gap") else "%")
    fig.tight_layout()
    return _save(fig, path)


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], methods: list[str], path: str | Path,
                     n_bins: int = 15, title_suffix: str = "") -> Path:
    """Reliability diagram per method (small multiples), seeds pooled."""
    _style()
    colors = method_colors(methods)
    present = [m for m in methods if m in curves]
    fig, axes = plt.subplots(1, len(present), figsize=(3.3 * len(present), 3.3), squeeze=False)
    for ax, method in zip(axes[0], present):
        probs, y = curves[method]
        bins = reliability_bins(probs, y, n_bins)
        centres = (bins["edges"][:-1] + bins["edges"][1:]) / 2
        filled = bins["count"] > 0
        ax.bar(centres[filled], bins["accuracy"][filled], width=1.0 / n_bins - 0.006, color=colors[method],
               alpha=0.85, zorder=2)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{method}{title_suffix}", loc="left")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, path)
