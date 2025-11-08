import os
import sys
import mlflow
import pandas as pd
from mlflow.entities import ViewType
from typing import Set, Dict, List, Optional # Added Optional

# Assume this setup module is necessary for connecting to Azure ML
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# We need to ensure the ml_client import works correctly based on your setup
from azure_connection.ml_client_setup import ml_client


def get_combined_runs_dataframe(ml_client, experiment_prefix: str = "auto") -> pd.DataFrame:
    """
    Connects to Azure ML, finds all experiments matching the prefix,
    retrieves all runs, and returns a single combined DataFrame.
    """
    # 1. Setup MLflow Tracking
    try:
        # Assuming ml_client is configured with the necessary workspace details
        mlflow.set_tracking_uri(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)
    except Exception as e:
        print(f"Error setting MLflow tracking URI. Check your ml_client setup: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

    # 2. Get and Filter Experiments
    experiments = mlflow.search_experiments() 
    experiment_names = [exp.name for exp in experiments if exp.name.startswith(experiment_prefix)]

    if not experiment_names:
        print(f"Warning: No experiments found starting with '{experiment_prefix}'.")
        return pd.DataFrame()

    all_runs_dfs: List[pd.DataFrame] = []

    # 3. Fetch Runs and Collect DataFrames
    for name in experiment_names:
        experiment = mlflow.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id 
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id], 
            filter_string="", 
            run_view_type=ViewType.ALL
        )
        
        df_runs = pd.DataFrame(runs)
        all_runs_dfs.append(df_runs)

    # 4. Combine all DataFrames
    combined_runs_df = pd.concat(all_runs_dfs, ignore_index=True)
    return combined_runs_df


def find_best_run_id_per_metric(df: pd.DataFrame) -> Dict[str, str]:
    """
    Analyzes the combined runs DataFrame to find the run_id that achieved the 
    maximum value for a set of specified metrics.
    
    Returns:
        Dict[str, str]: A dictionary mapping the metric name to its best run ID.
    """
    # Define metrics of interest
    metric_col_names = [
        col for col in df.columns 
        if col.startswith("metrics.AUC") or 
           col.startswith("metrics.recall") or 
           col.startswith("metrics.average") or 
           col.startswith("metrics.f1")
    ]
    
    if df.empty or not metric_col_names:
        print("Warning: DataFrame is empty or no target metrics were found.")
        return {}

    best_runs_per_metric: Dict[str, str] = {}
    
    # Find the best run ID for each metric
    for metric in metric_col_names:
        # Check if the column is present and not entirely empty/NaN
        if metric in df.columns and not df[metric].isnull().all():
            # idxmax() gives the index of the first occurrence of the maximum value
            best_row = df.loc[df[metric].idxmax()]
            best_runs_per_metric[metric] = best_row['run_id']
    
    # *** REVISION: Return the dictionary directly ***
    return best_runs_per_metric


# --- Execution Block ---

# 1. Get the combined DataFrame
combined_df = get_combined_runs_dataframe(ml_client, experiment_prefix="auto")

# 2. Find the dictionary of best run IDs (Metric: Run ID)
# *** REVISION: The type hint is now Dict[str, str] ***
auto_ml_best_runs: Dict[str, str] = find_best_run_id_per_metric(combined_df)

# This block ensures printing/debugging logic runs only when executed directly
if __name__ == "__main__":
    if not combined_df.empty:
        print(f"Combined data from {len(combined_df)} runs.")
        print("-" * 30)
        
    if auto_ml_best_runs:
        print(f"Found {len(auto_ml_best_runs)} best run IDs, one for each metric:")
        # *** REVISION: Iterate over metric and run_id pairs ***
        for metric, run_id in auto_ml_best_runs.items():
            # You might want to re-fetch the value for printing clarity
            try:
                best_value = combined_df[combined_df['run_id'] == run_id][metric].iloc[0]
                print(f"- {metric}: Run ID = {run_id} (Value: {best_value:.4f})")
            except Exception:
                print(f"- {metric}: Run ID = {run_id} (Value: N/A)")
    else:
        print("No best runs were identified.")
