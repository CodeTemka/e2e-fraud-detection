from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

ID_COLUMNS = ("id", "transaction_id", "row_id")


@dataclass
class ModelAssets:
    model_type: str
    model: Any
    amount_scaler: Any
    time_scaler: Any
    feature_columns: list[str]
    raw_feature_columns: list[str]
    label_column: str


MODEL_ASSETS: ModelAssets | None = None
DEFAULT_ALERT_CAP = 100
APP_INSIGHTS_CONFIGURED = False
VERBOSE_LOGGING_ENV = "FRAUD_LOG_VERBOSE"


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in model dir: {model_dir}")
    return json.loads(metadata_path.read_text())


def _load_model(model_dir: Path, model_type: str) -> Any:
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        model_path = model_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"xgboost model.json not found in {model_dir}")
        model = XGBClassifier()
        model.load_model(str(model_path))
        return model

    if model_type == "lightgbm":
        from lightgbm import Booster

        model_path = model_dir / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"lightgbm model.txt not found in {model_dir}")
        return Booster(model_file=str(model_path))

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _get_app_insights_connection_string() -> str | None:
    for env_key in (
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "APPINSIGHTS_CONNECTION_STRING",
        "AZURE_MONITOR_CONNECTION_STRING",
    ):
        value = os.environ.get(env_key)
        if value:
            return value
    return None


def _configure_app_insights_logger() -> None:
    global APP_INSIGHTS_CONFIGURED

    if APP_INSIGHTS_CONFIGURED:
        return

    connection_string = _get_app_insights_connection_string()
    if not connection_string:
        return

    if importlib.util.find_spec("opencensus.ext.azure.log_exporter") is None:
        logger.info(
            "Application Insights SDK not installed; skipping log handler setup.",
            extra={"app_insights_configured": False},
        )
        APP_INSIGHTS_CONFIGURED = True
        return

    module = importlib.import_module("opencensus.ext.azure.log_exporter")
    azure_handler = module.AzureLogHandler(connection_string=connection_string)
    azure_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger("fraud_detection")
    root_logger.addHandler(azure_handler)
    APP_INSIGHTS_CONFIGURED = True
    logger.info(
        "Application Insights log handler configured.",
        extra={"app_insights_configured": True},
    )


def _predict_proba(model: Any, model_type: str, features: pd.DataFrame) -> np.ndarray:
    if model_type == "xgboost":
        proba = model.predict_proba(features)[:, 1]
        return np.asarray(proba, dtype=float)

    if model_type == "lightgbm":
        proba = model.predict(features)
        return np.asarray(proba, dtype=float)

    raise ValueError(f"Unsupported model_type '{model_type}'.")


def _parse_alert_cap(raw_data: dict[str, Any] | None) -> int | None:
    if not raw_data:
        return None
    cap = raw_data.get("alert_cap")
    if cap is None:
        return None
    try:
        return max(int(cap), 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid alert_cap '{cap}'. Must be an integer.") from exc


def _parse_request_id(raw_data: Any) -> str:
    if raw_data is None:
        return str(uuid.uuid4())

    parsed = raw_data
    if isinstance(raw_data, (str, bytes)):
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError:
            return str(uuid.uuid4())

    if isinstance(parsed, dict):
        for key in ("request_id", "requestId", "correlation_id", "correlationId"):
            value = parsed.get(key)
            if value:
                return str(value)
    return str(uuid.uuid4())


def _is_verbose_logging_enabled() -> bool:
    value = os.environ.get(VERBOSE_LOGGING_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _coerce_records(raw_data: Any) -> tuple[pd.DataFrame, int | None]:
    if raw_data is None:
        return pd.DataFrame(), None

    if isinstance(raw_data, (str, bytes)):
        raw_data = json.loads(raw_data)

    if isinstance(raw_data, dict) and "input_data" in raw_data:
        payload = raw_data["input_data"] or {}
        columns = payload.get("columns") or []
        data_rows = payload.get("data") or []
        df = pd.DataFrame(data_rows, columns=columns)
        return df, _parse_alert_cap(raw_data)

    if isinstance(raw_data, dict) and "data" in raw_data:
        data = raw_data.get("data")
        df = _records_to_df(data)
        return df, _parse_alert_cap(raw_data)

    if isinstance(raw_data, dict):
        df = _records_to_df(raw_data)
        return df, _parse_alert_cap(raw_data)

    return _records_to_df(raw_data), None


def _records_to_df(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if isinstance(data[0], dict):
            return pd.DataFrame(data)
    raise ValueError("Input data must be a dict, list of dicts, or AzureML input_data format.")


def _extract_row_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Any] | None]:
    for col in ID_COLUMNS:
        if col in df.columns:
            row_ids = df[col].tolist()
            return df.drop(columns=[col]), row_ids
    return df, None


def _prepare_features(df: pd.DataFrame, assets: ModelAssets) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if assets.label_column in df.columns:
        df = df.drop(columns=[assets.label_column])

    missing = [c for c in assets.raw_feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[assets.raw_feature_columns]

    df["scaled_amount"] = assets.amount_scaler.transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = assets.time_scaler.transform(df["Time"].values.reshape(-1, 1))
    df = df.drop(columns=["Amount", "Time"])

    missing_features = [c for c in assets.feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns after scaling: {missing_features}")

    return df[assets.feature_columns]


def apply_top_k_threshold(probabilities: np.ndarray, alert_cap: int) -> tuple[list[int], float, int]:
    if probabilities.size == 0:
        return [], 1.0, 0

    n = probabilities.size
    k = min(max(alert_cap, 0), n)
    if k == 0:
        return [0] * n, 1.0, 0

    top_indices = np.argpartition(-probabilities, k - 1)[:k]
    preds = np.zeros(n, dtype=int)
    preds[top_indices] = 1
    threshold = float(np.min(probabilities[top_indices]))
    return preds.tolist(), threshold, int(k)


def init() -> None:
    global MODEL_ASSETS
    global DEFAULT_ALERT_CAP

    _configure_app_insights_logger()

    model_dir = Path(os.environ.get("AZUREML_MODEL_DIR", ".")).resolve()
    metadata = _load_metadata(model_dir)
    model_type = metadata.get("model_type")
    if not model_type:
        raise ValueError("metadata.json missing required field 'model_type'.")

    model = _load_model(model_dir, model_type=model_type)
    amount_scaler = load(model_dir / "scaler_amount.pkl")
    time_scaler = load(model_dir / "scaler_time.pkl")

    MODEL_ASSETS = ModelAssets(
        model_type=model_type,
        model=model,
        amount_scaler=amount_scaler,
        time_scaler=time_scaler,
        feature_columns=list(metadata.get("feature_columns") or []),
        raw_feature_columns=list(metadata.get("raw_feature_columns") or []),
        label_column=str(metadata.get("label_column") or "Class"),
    )

    env_alert_cap = os.environ.get("ALERT_CAP")
    if env_alert_cap is not None:
        try:
            DEFAULT_ALERT_CAP = max(int(env_alert_cap), 0)
        except (TypeError, ValueError):
            logger.warning("Invalid ALERT_CAP env var '%s'; defaulting to %s.", env_alert_cap, DEFAULT_ALERT_CAP)

    logger.info(
        "Model assets loaded",
        extra={
            "model_type": MODEL_ASSETS.model_type,
            "feature_count": len(MODEL_ASSETS.feature_columns),
            "raw_feature_count": len(MODEL_ASSETS.raw_feature_columns),
            "default_alert_cap": DEFAULT_ALERT_CAP,
        },
    )


def run(raw_data: Any) -> dict[str, Any]:
    if MODEL_ASSETS is None:
        raise RuntimeError("Model assets not initialized. Call init() first.")

    start_time = time.perf_counter()
    request_id = _parse_request_id(raw_data)
    df, alert_cap_override = _coerce_records(raw_data)
    df, row_ids = _extract_row_ids(df)
    features = _prepare_features(df, MODEL_ASSETS)
    verbose_logging = _is_verbose_logging_enabled()

    if features.empty:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_payload = {
            "request_id": request_id,
            "record_count": len(df),
            "alert_cap_override": alert_cap_override,
            "alert_cap": 0,
            "num_alerts": 0,
            "duration_ms": duration_ms,
        }
        logger.info("inference_empty", extra=log_payload)
        response = {
            "predictions": [],
            "probabilities": [],
            "threshold": 1.0,
            "alert_cap": 0,
            "num_alerts": 0,
        }
        if row_ids is not None:
            response["row_ids"] = row_ids
        return response

    alert_cap = alert_cap_override if alert_cap_override is not None else DEFAULT_ALERT_CAP
    alert_cap = min(max(int(alert_cap), 0), len(features))

    probabilities = _predict_proba(MODEL_ASSETS.model, MODEL_ASSETS.model_type, features)
    predictions, threshold, num_alerts = apply_top_k_threshold(probabilities, alert_cap)

    duration_ms = (time.perf_counter() - start_time) * 1000
    log_payload = {
        "request_id": request_id,
        "record_count": len(df),
        "alert_cap_override": alert_cap_override,
        "alert_cap": alert_cap,
        "num_alerts": num_alerts,
        "threshold": threshold,
        "duration_ms": duration_ms,
    }
    if verbose_logging:
        log_payload.update(
            {
                "avg_score": float(np.mean(probabilities)),
                "min_score": float(np.min(probabilities)),
                "max_score": float(np.max(probabilities)),
            }
        )
    logger.info("inference", extra=log_payload)

    response = {
        "predictions": predictions,
        "probabilities": probabilities.tolist(),
        "threshold": threshold,
        "alert_cap": alert_cap,
        "num_alerts": num_alerts,
    }
    if row_ids is not None:
        response["row_ids"] = row_ids
    return response
