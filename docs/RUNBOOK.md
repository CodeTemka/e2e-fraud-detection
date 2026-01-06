# Operational Runbook

## Owners
- **Primary owner**: ML Platform team
- **Secondary owner**: Fraud Analytics team
- **Escalation**: Azure ML workspace administrators

## Prerequisites
- Azure credentials available via `DefaultAzureCredential`.
- `.env` populated from `.env.example`.
- Dataset available locally (default `data/creditcard.csv`) or registered as a data asset.

## Training workflow (AutoML)
1. **Prep resources** (one-time per workspace or when compute changes):
   ```bash
   python scripts/az_prep.py
   ```
2. **Register dataset**:
   ```bash
   python scripts/register_local_data.py
   ```
3. **Submit training job** (recall-focused by default):
   ```bash
   fraud-cli submit-automl --metric recall
   ```
4. **Monitor runs** in Azure ML studio or via CLI logs.

## Training workflow (XGBoost sweep)
1. **Validate local data** (optional but recommended):
   ```bash
   fraud-cli validate-data --sample-rows 2000
   ```
2. **Submit the sweep**:
   ```bash
   fraud-cli submit-xgb-sweep --metric average_precision
   ```
3. **Confirm completion** in Azure ML studio and capture the best run ID.

## Promotion workflow
1. **Register + promote AutoML model** (best metric across runs):
   ```bash
   fraud-cli register-best-automl-model --best-by-metric --metric norm-macro-recall
   ```
2. **Register + promote custom model**:
   ```bash
   fraud-cli register-best-custom-model --metric average_precision_score_macro
   ```
3. **Verify model version** in the Azure ML registry.

## Deployment workflow
1. **Create endpoint** (if not already present):
   ```bash
   fraud-cli endpoint-create
   ```
2. **Deploy current production model**:
   ```bash
   fraud-cli deploy
   ```
3. **Smoke test invocation**:
   ```bash
   fraud-cli serve-invoke
   ```

## Rollback procedure
1. **Identify last known-good model** in the registry.
2. **Redeploy that version**:
   - The default `fraud-cli deploy` command always deploys the latest registered version.
   - To rollback, update the deployment in Azure ML Studio (or `az ml` CLI) to the previous
     model version, then verify the deployment succeeds.
3. **Validate endpoint health** via `fraud-cli serve-invoke`.
4. **Document the incident** and notify the owners above.

## Operational checks
- Confirm Azure ML compute is running and has available quota.
- Confirm dataset registration points to the correct storage path.
- Verify model version numbers in the registry before promotion.
