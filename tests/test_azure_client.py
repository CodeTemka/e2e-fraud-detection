from types import SimpleNamespace

import pytest

from fraud_detection.azure import client


class DummySettings:
    def __init__(self):
        self.subscription_id = "sub-id"
        self.resource_group = "rg-name"
        self.workspace_name = "ws-name"
        self.require_called = False

    def require_azure(self):
        self.require_called = True
        return self


def test_build_default_credential_uses_non_interactive(monkeypatch):
    captured_kwargs = {}

    class DummyCredential:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(client, "DefaultAzureCredential", DummyCredential)

    credential = client.build_default_credential()

    assert isinstance(credential, DummyCredential)
    assert captured_kwargs == {"exclude_interactive_browser_credential": True}


def test_get_ml_client_uses_defaults(monkeypatch):
    settings = DummySettings()
    credential = object()
    captured_kwargs = {}

    def fake_get_settings():
        return settings

    def fake_build_default_credential():
        return credential

    class DummyMLClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(client, "get_settings", fake_get_settings)
    monkeypatch.setattr(client, "build_default_credential", fake_build_default_credential)
    monkeypatch.setattr(client, "MLClient", DummyMLClient)

    returned_client = client.get_ml_client()

    assert isinstance(returned_client, DummyMLClient)
    assert settings.require_called is True
    assert captured_kwargs == {
        "credential": credential,
        "subscription_id": settings.subscription_id,
        "resource_group_name": settings.resource_group,
        "workspace_name": settings.workspace_name,
    }


def test_get_ml_client_respects_overrides(monkeypatch):
    settings = DummySettings()
    custom_credential = object()
    captured_kwargs = {}

    monkeypatch.setattr(
        client,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("should not call get_settings")),
    )
    monkeypatch.setattr(
        client,
        "build_default_credential",
        lambda: (_ for _ in ()).throw(AssertionError("should not build default credential")),
    )

    class DummyMLClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(client, "MLClient", DummyMLClient)

    returned_client = client.get_ml_client(settings=settings, credential=custom_credential)

    assert isinstance(returned_client, DummyMLClient)
    assert captured_kwargs["credential"] is custom_credential
    assert captured_kwargs["subscription_id"] == settings.subscription_id
    assert captured_kwargs["resource_group_name"] == settings.resource_group
    assert captured_kwargs["workspace_name"] == settings.workspace_name
