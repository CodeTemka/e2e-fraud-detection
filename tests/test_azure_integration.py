import os
from types import SimpleNamespace

import pytest
from azure.ai.ml.constants import AssetTypes

from fraud_detection.training import automl
from fraud_detection.training import submit_lgbm, submit_xgb

REQUIRED_AZURE_ENV_VARS = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_WORKSPACE_NAME",
)
HAS_AZURE_ENV = all(os.getenv(name) for name in REQUIRED_AZURE_ENV_VARS)
SKIP_IN_CI = bool(os.getenv("CI")) and not HAS_AZURE_ENV


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IN_CI, reason="Azure ML env vars not set in CI")
def test_submit_job_uses_mocked_ml_client():
    captured = {}

    class FakeJobs:
        def create_or_update(self, job):
            captured["job"] = job
            return SimpleNamespace(name="job-123")

    ml_client = SimpleNamespace(jobs=FakeJobs())
    job = SimpleNamespace(name="local-job")

    result = automl.submit_job(ml_client, job)

    assert result == "job-123"
    assert captured["job"] is job


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IN_CI, reason="Azure ML env vars not set in CI")
@pytest.mark.parametrize(
    "resolver",
    [submit_xgb._resolve_data_input, submit_lgbm._resolve_data_input],
)
def test_data_input_resolves_registered_asset(resolver):
    data_input = resolver("azureml:creditcard:1")

    assert data_input.type == AssetTypes.MLTABLE
    assert data_input.path == "azureml:creditcard:1"


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IN_CI, reason="Azure ML env vars not set in CI")
@pytest.mark.parametrize(
    "resolver",
    [submit_xgb._resolve_data_input, submit_lgbm._resolve_data_input],
)
def test_data_input_resolves_mltable_folder(tmp_path, resolver):
    mltable_dir = tmp_path / "mltable"
    mltable_dir.mkdir()
    (mltable_dir / "MLTable").write_text("paths:\n- file: ./data.csv\n")

    data_input = resolver(str(mltable_dir))

    assert data_input.type == AssetTypes.MLTABLE
    assert data_input.path == str(mltable_dir)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_IN_CI, reason="Azure ML env vars not set in CI")
@pytest.mark.parametrize(
    "resolver",
    [submit_xgb._resolve_data_input, submit_lgbm._resolve_data_input],
)
def test_data_input_resolves_csv_file(tmp_path, resolver):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    data_input = resolver(str(csv_path))

    assert data_input.type == AssetTypes.URI_FILE
    assert data_input.path == str(csv_path)
