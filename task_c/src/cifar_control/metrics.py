"""Evaluation metrics, computed on predicted class probabilities.

* accuracy
* nll       : mean negative log-likelihood of the true class
* ece       : expected calibration error, top-label, ``n_bins`` equal-width bins on (lo, hi]
* auc_macro : one-vs-rest ROC AUC averaged over classes
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_ECE_BINS = 15


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, None)).mean())


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_ECE_BINS) -> dict[str, np.ndarray]:
    """Per bin: mean confidence, accuracy and count (top-label)."""
    confidence = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.searchsorted(edges[1:-1], confidence, side="left")
    counts = np.bincount(bin_of, minlength=n_bins)
    conf_sum = np.bincount(bin_of, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(bin_of, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"edges": edges, "count": counts, "confidence": conf_sum / counts, "accuracy": acc_sum / counts}


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_ECE_BINS) -> float:
    bins = reliability_bins(probs, y, n_bins)
    filled = bins["count"] > 0
    gap = np.abs(bins["accuracy"][filled] - bins["confidence"][filled])
    return float((bins["count"][filled] * gap).sum() / len(y))


def auc_macro(probs: np.ndarray, y: np.ndarray) -> float:
    probs = probs / probs.sum(1, keepdims=True)
    try:
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=np.arange(probs.shape[1])))
    except ValueError:  # a class absent from y
        return float("nan")


def all_metrics(probs: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_ECE_BINS) -> dict[str, float]:
    return {"acc": accuracy(probs, y), "nll": nll(probs, y), "ece": ece(probs, y, n_bins), "auc": auc_macro(probs, y)}
