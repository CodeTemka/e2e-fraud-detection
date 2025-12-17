from types import SimpleNamespace

import pytest

from fraud_detection.registry import register_from_run


class _Model:
    def __init__(self, name="model", version="1", tags=None):
        self.name = name
        self.version = version
        self.tags = tags or {}


class _Models:
    def __init__(self):
        self.created = []
        self.existing = []

    def list(self, name=None):
        return self.existing

    def create_or_update(self, model):
        self.created.append(model)
        return _Model(name=model.name, version="2", tags=model.tags)

    def get(self, name, version):
        return _Model(name=name, version=version)


@pytest.fixture
def ml_client():
    models = _Models()
    return SimpleNamespace(models=models)


def test_find_existing_version_returns_match(ml_client):
    ml_client.models.existing = [_Model(tags={"source_run_id": "run-123", "other": "x"})]

    found = register_from_run.find_existing_version_for_run(
        ml_client, model_name="fraud", run_id="run-123"
    )

    assert found == "1"


def test_register_model_from_run_reuses_existing_version(ml_client):
    ml_client.models.existing = [_Model(version="5", tags={"source_run_id": "run-1"})]

    model = register_from_run.register_model_from_run(
        ml_client, model_name="fraud", run_id="run-1", description="desc"
    )

    assert model.version == "5"


def test_register_model_from_run_tries_candidates_in_order(ml_client, monkeypatch):
    # Fail first path, succeed second
    calls = []

    def fake_create_or_update(model):
        calls.append(model.path)
        if len(calls) == 1:
            raise RuntimeError("bad path")
        return _Model(name=model.name, version="3", tags=model.tags)

    ml_client.models.create_or_update = fake_create_or_update

    model = register_from_run.register_model_from_run(
        ml_client, model_name="fraud", run_id="run-2", artifact_paths=["bad/", "good/"]
    )

    assert calls == ["azureml://jobs/run-2/bad/", "azureml://jobs/run-2/good/"]
    assert model.version == "3"
    assert model.tags["source_run_id"] == "run-2"


def test_register_model_from_run_raises_after_exhausting(monkeypatch, ml_client):
    def always_fail(model):
        raise RuntimeError("nope")

    ml_client.models.create_or_update = always_fail

    with pytest.raises(RuntimeError):
        register_from_run.register_model_from_run(
            ml_client, model_name="fraud", run_id="run-3", artifact_paths=["one/"]
        )
