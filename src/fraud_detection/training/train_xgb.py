from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to a CSV file (mounted/downloaded by Azure ML).")
    p.add_argument("--label_col", type=str, default="Class", help="Label column name in the CSV.")

    # Model hyperparameters (these will be tuned by sweep)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)

    # Imbalance helper (optional)
    p.add_argument("--scale_pos_weight", type=float, default=1.0)

    # Training controls
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2)  # fraction of TRAIN split used for validation
    p.add_argument("--random_state", type=int, default=42)

    # Early stopping (optional)
    p.add_argument("--early_stopping_rounds", type=int, default=50)

    # Outputs (Azure ML captures ./outputs automatically)
    p.add_argument("--output_dir", type=str, default="outputs")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.train_data)
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found. Columns: {list(df.columns)[:30]} ...")

    y = df[args.label_col].astype(int).to_numpy()
    X = df.drop(columns=[args.label_col]).to_numpy()

    # Split: train vs test, then train -> train/val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.val_size,
        stratify=y_train,
        random_state=args.random_state,
    )

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=args.scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",  # internal eval metric; we'll still log sklearn metrics explicitly
        tree_method="hist",
        n_jobs=-1,
        random_state=args.random_state,
    )

    # Azure ML sweep expects the primary metric name to match EXACTLY what you log. :contentReference[oaicite:1]{index=1}
    if mlflow.active_run() is None:
        mlflow.start_run()

    # Log params (nice to see in the sweep UI)
    mlflow.log_params(
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "reg_lambda": args.reg_lambda,
            "scale_pos_weight": args.scale_pos_weight,
        }
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
    )

    proba_test = model.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba_test))
    auc = float(roc_auc_score(y_test, proba_test))

    # IMPORTANT: primary_metric must match the string you set in sweep_job.primary_metric. :contentReference[oaicite:2]{index=2}
    mlflow.log_metric("average_precision", ap)
    mlflow.log_metric("roc_auc", auc)

    # Save model artifact
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.json"
    model.save_model(str(model_path))
    mlflow.log_artifact(str(model_path))

    mlflow.end_run()

    print(f"Done. average_precision={ap:.6f}, roc_auc={auc:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
