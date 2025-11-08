import os
import sys
import mlflow
import pandas as pd
from mlflow.entities import ViewType
from typing import Set, Dict, List, Any # Using Any for the nested metric value

# Assume this setup module is necessary for connecting to Azure ML
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
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
        # Note: This line might require error handling based on your Azure SDK version
        # If ml_client.workspaces.get(...).mlflow_tracking_uri fails, use a fallback
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
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
        if not experiment:
             print(f"Warning: Could not retrieve experiment {name}.")
             continue
        experiment_id = experiment.experiment_id 
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id], 
            filter_string="", 
            run_view_type=ViewType.ALL
        )
        
        df_runs = pd.DataFrame(runs)
        all_runs_dfs.append(df_runs)

    # 4. Combine all DataFrames
    if not all_runs_dfs:
        return pd.DataFrame()
    combined_runs_df = pd.concat(all_runs_dfs, ignore_index=True)
    return combined_runs_df


def find_best_run_id_per_metric(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes the combined runs DataFrame to find the run_id and the value
    that achieved the maximum for a set of specified metrics.
    
    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping the metric name 
        to a nested dictionary containing {'run_id': str, 'metric_value': float}.
    """
    # Define metrics of interest
    metric_col_names = [
        col for col in df.columns 
        if col.startswith("metrics.AUC") or 
           col.startswith("metrics.recall") or 
           col.startswith("metrics.average") or 
           col.startswith("metrics.f1") or
           col.startswith("metrics.norm")
    ]
    
    if df.empty or not metric_col_names:
        print("Warning: DataFrame is empty or no target metrics were found.")
        return {}

    # Updated: Type hint reflects the nested structure
    best_runs_per_metric: Dict[str, Dict[str, Any]] = {}
    
    # Find the best run ID and value for each metric
    for metric in metric_col_names:
        # Check if the column is present and not entirely empty/NaN
        if metric in df.columns and not df[metric].isnull().all():
            # idxmax() gives the index of the first occurrence of the maximum value
            best_row = df.loc[df[metric].idxmax()]
            
            # Create the nested dictionary using the string literal 'run_id'
            best_runs_per_metric[metric] = {
                'run_id': best_row['run_id'],        # FIX: Use string literal 'run_id'
                'metric_value': best_row[metric]     # The best value for this metric
            }
    
    return best_runs_per_metric


# --- Execution Block ---

# 1. Get the combined DataFrame
combined_df = get_combined_runs_dataframe(ml_client, experiment_prefix="auto")

# 2. Find the dictionary of best run IDs and values (Nested Dict)
auto_ml_best_runs: Dict[str, Dict[str, Any]] = find_best_run_id_per_metric(combined_df)


if __name__ == "__main__":
    if not combined_df.empty:
        print(f"Combined data from {len(combined_df)} runs.")
        print("-" * 30)
        
    if auto_ml_best_runs:
        print(f"Found {len(auto_ml_best_runs)} best run IDs, one for each metric:")
        # Iterate over metric and the nested dictionary (data)
        for metric, data in auto_ml_best_runs.items(): 
            # Access the data dictionary keys directly
            run_id = data.get('run_id', 'N/A')
            best_value = data.get('metric_value', 'N/A')
            
            # Format the value if it's a float, otherwise print N/A
            if isinstance(best_value, (float, int)):
                 print(f"- {metric}: Run ID = {run_id} (Value: {best_value:.4f})")
            else:
                 print(f"- {metric}: Run ID = {run_id} (Value: {best_value})")
    else:
        print("No best runs were identified.")