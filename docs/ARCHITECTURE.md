# Architecture

## System boundaries
The fraud-detection system consists of two major boundaries:

- **Local developer/CI boundary**: CLI, config files, and scripts that submit jobs and manage models.
- **Azure ML boundary**: managed compute, storage, experiment tracking, model registry, and online endpoints.

External dependencies include Azure Active Directory (auth), Azure Blob Storage (datasets), and
Azure ML-managed compute clusters.

## High-level flow
_Diagram format:_ Mermaid (replace with an image if preferred).

```mermaid
flowchart LR
  subgraph Local[Local dev + CI]
    CLI[fraud-cli]
    Scripts[scripts/*.py]
    Config[.env + config files]
  end

  subgraph AzureML[Azure ML Workspace]
    Compute[Compute Cluster]
    Experiments[Experiments + Runs]
    Registry[Model Registry]
    Endpoint[Online Endpoint]
    DataAssets[Data Assets]
  end

  Auth[Azure AD]
  Storage[(Blob Storage)]

  CLI -->|submit jobs| Experiments
  Scripts -->|prep resources| AzureML
  CLI -->|register data| DataAssets
  DataAssets -->|mount/download| Compute
  Compute -->|train| Experiments
  Experiments -->|register best| Registry
  Registry -->|deploy model| Endpoint
  CLI -->|invoke| Endpoint
  Auth -->|token| CLI
  Storage -->|dataset| DataAssets
```

## Key components
- **CLI (`fraud-cli`)**: Submits AutoML and custom training jobs, registers best models, and manages
  endpoints. Source: `src/fraud_detection/cli/*`.
- **Training utilities**: AutoML and XGBoost training assets plus environment definitions.
  Source: `src/fraud_detection/training/*`.
- **Azure helpers**: Authentication, workspace access, MLflow registration.
  Source: `src/fraud_detection/azure/*`.

## Data & model lifecycle
1. Register raw datasets as Azure ML data assets.
2. Submit training jobs (AutoML or custom XGBoost sweep).
3. Register the best model in the Azure ML registry.
4. Promote the best model to production (if metrics improve).
5. Deploy to an online endpoint and invoke for scoring.
