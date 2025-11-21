import pytest

from fraud_detection.config import AzureMLConfig


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_ID", "sub-id")
    monkeypatch.setenv("RESOURCE_GROUP", "rg")
    monkeypatch.setenv("WORKSPACE_NAME", "ws")

    config = AzureMLConfig.from_env()
    assert config.subscription_id == "sub-id"
    assert config.resource_group == "rg"
    assert config.workspace_name == "ws"


def test_config_missing_env(monkeypatch):
    monkeypatch.delenv("SUBSCRIPTION_ID", raising=False)
    monkeypatch.delenv("RESOURCE_GROUP", raising=False)
    monkeypatch.delenv("WORKSPACE_NAME", raising=False)

    with pytest.raises(OSError) as exc:
        AzureMLConfig.from_env()

    assert "Missing required environment variables" in str(exc.value)
