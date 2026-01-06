import numpy as np
import pytest

from fraud_detection.serving.custom_scoring import apply_top_k_threshold
from fraud_detection.serving.online_endpoint import _build_initial_traffic, _validate_traffic


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


def test_build_initial_traffic_first_deployment():
    traffic = _build_initial_traffic(
        current_traffic={},
        deployment_name="blue",
        low_traffic_percent=10,
    )

    assert traffic == {"blue": 100}


def test_build_initial_traffic_rebalances_existing():
    traffic = _build_initial_traffic(
        current_traffic={"blue": 70, "green": 30},
        deployment_name="canary",
        low_traffic_percent=10,
    )

    assert traffic == {"blue": 63, "green": 27, "canary": 10}


def test_build_initial_traffic_even_split_when_existing_zero():
    traffic = _build_initial_traffic(
        current_traffic={"blue": 0, "green": 0},
        deployment_name="canary",
        low_traffic_percent=20,
    )

    assert traffic == {"blue": 40, "green": 40, "canary": 20}


def test_validate_traffic_accepts_valid_payload():
    _validate_traffic({"blue": 100})


@pytest.mark.parametrize(
    "traffic, message",
    [
        ({}, "traffic mapping is empty"),
        ({1: 100}, "traffic keys must be deployment names"),
        ({"blue": 99.5}, "traffic percentage for 'blue' must be an int"),
        ({"blue": -1, "green": 101}, "traffic percentage for 'blue' must be >= 0"),
        ({"blue": 50, "green": 40}, "traffic must sum to 100"),
    ],
)
def test_validate_traffic_rejects_invalid_payloads(traffic, message):
    with pytest.raises(ValueError, match=message):
        _validate_traffic(traffic)
