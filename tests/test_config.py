import importlib

import pytest

from fraud_detection import config


def test_settings_loads_env(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_ID", "sub-id")
    monkeypatch.setenv("RESOURCE_GROUP", "rg")
    monkeypatch.setenv("WORKSPACE_NAME", "ws")
    monkeypatch.setenv("AML_COMPUTE", "cpu-cluster")
    monkeypatch.setenv("AML_DATASET", "azureml:dataset:1")

    importlib.reload(config)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.subscription_id == "sub-id"
    assert settings.resource_group == "rg"
    assert settings.workspace_name == "ws"
    assert settings.default_compute == "cpu-cluster"
    assert settings.default_dataset == "azureml:dataset:1"


def test_settings_allows_missing_azure_env(monkeypatch):
    monkeypatch.delenv("SUBSCRIPTION_ID", raising=False)
    monkeypatch.delenv("RESOURCE_GROUP", raising=False)
    monkeypatch.delenv("WORKSPACE_NAME", raising=False)

    importlib.reload(config)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.subscription_id is None
    assert settings.resource_group is None
    assert settings.workspace_name is None

    with pytest.raises(ValueError, match="SUBSCRIPTION_ID"):
        settings.require_azure()
