# Troubleshooting Azure ML Failures

## Authentication errors
**Symptoms**
- `ClientAuthenticationError`, `DefaultAzureCredential` failures, or 401/403 errors.

**Checks**
1. Confirm `.env` values match the intended subscription/workspace.
2. Ensure `az login` or managed identity is available in the environment.
3. Verify service principal secrets if using non-interactive auth.

**Fix**
- Re-run `az login` or refresh service principal credentials.
- Validate the `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET` variables.

## Workspace not found
**Symptoms**
- `ResourceNotFoundError` when loading the workspace.

**Checks**
1. Confirm `subscription_id`, `resource_group`, and `workspace_name` in `.env`.
2. Verify access rights for the calling identity.

**Fix**
- Update `.env` and re-run `python scripts/az_prep.py`.

## Compute quota or cluster errors
**Symptoms**
- Jobs stuck in `Queued`, or errors referencing quota/region capacity.

**Checks**
1. Inspect the target compute cluster in Azure ML Studio.
2. Confirm VM size is available in the region.

**Fix**
- Resize the cluster or request quota increase.
- Update the compute name in `.env` if switching clusters.

## Data asset registration failures
**Symptoms**
- Errors when running `register_local_data.py` or MLTable registration.

**Checks**
1. Confirm the local path exists (`data/creditcard.csv` by default).
2. Check storage permissions for the workspace.

**Fix**
- Update the dataset path or register a different data asset version.
- Ensure the workspace storage account allows the current identity.

## Job submission failures
**Symptoms**
- `ValidationException`, environment build errors, or job schema errors.

**Checks**
1. Validate the CLI command arguments (metric, algorithms, etc.).
2. Inspect the environment definition files in `src/fraud_detection/training/*_env.yaml`.

**Fix**
- Re-run with `--dry-run` when available to validate job payloads.
- Pin compatible dependency versions in the environment file.

## Model registration failures
**Symptoms**
- Errors registering best runs or missing metrics.

**Checks**
1. Confirm the experiment has completed successfully.
2. Validate the metric name passed to the CLI.

**Fix**
- Use `fraud-cli register-best-automl-model --best-child-run` if the parent run lacks metrics.
- Re-run registration with the correct metric name.

## Endpoint deployment failures
**Symptoms**
- Deployment stuck in `Provisioning` or `Failed`.

**Checks**
1. Confirm the endpoint exists (`fraud-cli endpoint-create`).
2. Ensure model version is registered and accessible.
3. Inspect `azureml-logs` in Azure ML Studio for container errors.

**Fix**
- Re-deploy with a supported instance type or update `DEPLOYMENT_INSTANCE_TYPE`.
- For custom models, verify `src/fraud_detection/serving/custom_scoring.py` dependencies.
