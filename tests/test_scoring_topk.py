import numpy as np

from fraud_detection.serving.custom_scoring import apply_top_k_threshold


def test_top_k_threshold_basic():
    probs = np.array([0.9, 0.1, 0.8])
    preds, threshold, num_alerts = apply_top_k_threshold(probs, alert_cap=2)
    assert sum(preds) == 2
    assert num_alerts == 2
    assert threshold == 0.8


def test_top_k_threshold_ties_only_selects_k():
    probs = np.array([0.5, 0.5, 0.4])
    preds, threshold, num_alerts = apply_top_k_threshold(probs, alert_cap=1)
    assert sum(preds) == 1
    assert num_alerts == 1
    assert threshold == 0.5


def test_top_k_threshold_k_greater_than_n():
    probs = np.array([0.2, 0.3])
    preds, threshold, num_alerts = apply_top_k_threshold(probs, alert_cap=5)
    assert sum(preds) == 2
    assert num_alerts == 2
    assert threshold == 0.2


def test_top_k_threshold_empty():
    preds, threshold, num_alerts = apply_top_k_threshold(np.array([]), alert_cap=3)
    assert preds == []
    assert threshold == 1.0
    assert num_alerts == 0
