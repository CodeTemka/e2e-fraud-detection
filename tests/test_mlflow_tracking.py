from types import SimpleNamespace

from fraud_detection.mlflow import tracking


def test_set_tracking_uri_from_workspace(monkeypatch):
    recorded = {}

    def fake_set_tracking_uri(uri):
        recorded["uri"] = uri

    workspace = SimpleNamespace(mlflow_tracking_uri="https://workspace")
    ml_client = SimpleNamespace(workspace_name="ws", workspaces=SimpleNamespace(get=lambda name: workspace))

    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", fake_set_tracking_uri)

    uri = tracking.set_tracking_uri_from_workspace(ml_client)

    assert uri == "https://workspace"
    assert recorded["uri"] == "https://workspace"
