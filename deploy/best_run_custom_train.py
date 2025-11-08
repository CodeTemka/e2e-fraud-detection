import os
import sys
import mlflow
import pandas as pd
from mlflow.entities import ViewType

# Assume this setup module is necessary for connecting to Azure ML
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client


def get_best_runs_by_metric():
    """
    Connects to Azure ML, finds the last experiment, retrieves all runs,
    calculates the best run ID for each validation metric, and returns the result.
    
    Returns:
        dict: A dictionary mapping metric names to the best run ID.
    """
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)

    # 2. Get All Experiments and select the last one
    experiments = mlflow.search_experiments() 
    if not experiments:
        print("Error: No experiments found.")
        return {}
        
    experiment_names = [exp.name for exp in experiments]
    
    # Selecting the LAST experiment in the list
    experiment = mlflow.get_experiment_by_name(experiment_names[-1])
    experiment_id = experiment.experiment_id 
    
    # 3. Search Runs and create DataFrame
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="", run_view_type=ViewType.ALL)
    df = pd.DataFrame(runs)

    # 4. Identify Metric Columns
    score_metrics = [col for col in df.columns if col.startswith("metrics.valid_")]
    
    if not score_metrics:
        print("Warning: No validation metrics ('metrics.valid_') found in the runs.")
        return {}
        
    run_id_col = 'run_id'
    best_runs = {}

    # 5. Find Best Run for Each Metric
    for metric in score_metrics:
        # Check if the metric column contains numeric data and is not all NaN
        if pd.api.types.is_numeric_dtype(df[metric]) and not df[metric].empty and df[metric].any():
            best_row = df.loc[df[metric].idxmax()]
            best_runs[metric] = best_row[run_id_col]
            
            # Print the result only when executed as the main script (see below)
            # print(f"Best run for {metric}: Run ID = {best_row[run_id_col]}, Value = {best_row[metric]}")
            
    return best_runs

# --- Execution Block ---

# This variable will hold the result when the file is imported OR run as main.
# It is calculated once and stored.
custom_train_best_runs = get_best_runs_by_metric()

if __name__ == "__main__":
    print(f"Calculated Best Runs: {custom_train_best_runs}")
    
    for metric, run_id in custom_train_best_runs.items():
        # Re-fetch the run details to print the value (optional)
        best_run_details = mlflow.get_run(run_id)
        best_value = best_run_details.data.metrics.get(metric.split("metrics.")[-1])
        print(f"Best run for {metric}: Run ID = {run_id}, Value = {best_value}")


