import importlib

from fraud_detection import config


def test_settings_loads_env(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_ID", "sub-id")
    monkeypatch.setenv("RESOURCE_GROUP", "rg")
    monkeypatch.setenv("WORKSPACE_NAME", "ws")
    monkeypatch.setenv("LOCATION", "eastus")
    monkeypatch.setenv("AML_COMPUTE", "cpu-cluster")
    monkeypatch.setenv("DEPLOYMENT_INSTANCE_TYPE", "Standard_D2_v2")
    monkeypatch.setenv("AML_DATASET", "azureml:dataset:1")
    monkeypatch.setenv("GITHUB_OWNER", "octocat")
    monkeypatch.setenv("GITHUB_REPO", "fraud")

    importlib.reload(config)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.subscription_id == "sub-id"
    assert settings.resource_group == "rg"
    assert settings.workspace_name == "ws"
    assert settings.training_compute == "cpu-cluster"
    assert settings.compute_cluster == "cpu-cluster"
    assert settings.deployment_instance_type == "Standard_D2_v2"
    assert settings.dataset_name == "azureml:dataset:1"


def test_settings_allows_missing_azure_env(monkeypatch):
    monkeypatch.delenv("SUBSCRIPTION_ID", raising=False)
    monkeypatch.delenv("RESOURCE_GROUP", raising=False)
    monkeypatch.delenv("WORKSPACE_NAME", raising=False)
    monkeypatch.delenv("LOCATION", raising=False)
    monkeypatch.setenv("GITHUB_OWNER", "octocat")
    monkeypatch.setenv("GITHUB_REPO", "fraud")

    importlib.reload(config)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.subscription_id == config.DEFAULT_SUBSCRIPTION_ID
    assert settings.resource_group == config.DEFAULT_RESOURCE_GROUP
    assert settings.workspace_name == config.DEFAULT_WORKSPACE_NAME
