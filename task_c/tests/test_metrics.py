from __future__ import annotations

import numpy as np

from cifar_control.metrics import accuracy, auc_macro, ece, nll, reliability_bins


def test_perfect_confident_predictions():
    y = np.array([0, 1, 2, 1])
    probs = np.eye(3)[y]
    assert accuracy(probs, y) == 1.0
    assert ece(probs, y) == 0.0
    assert nll(probs, y) < 1e-9
    assert auc_macro(probs, y) == 1.0


def test_ece_known_value():
    # all predictions at confidence 0.8, half of them correct -> ECE = |0.5 - 0.8| = 0.3
    probs = np.tile([0.8, 0.2], (10, 1))
    y = np.array([0] * 5 + [1] * 5)
    assert np.isclose(ece(probs, y, n_bins=10), 0.3)


def test_reliability_bins_upper_edge_inclusive():
    probs = np.array([[1.0, 0.0], [0.6, 0.4]])
    bins = reliability_bins(probs, np.array([0, 0]), n_bins=5)
    assert bins["count"][-1] == 1 and bins["count"][2] == 1  # 1.0 in the last bin, 0.6 in (0.4, 0.6]


def test_auc_with_missing_class_is_nan():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    assert np.isnan(auc_macro(probs, np.array([0, 1])))
