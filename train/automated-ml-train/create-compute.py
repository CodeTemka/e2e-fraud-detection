from azure.ai.ml.entities import AmlCompute
from azure_connection.ml_client_setup import ml_client

cluster_name = "automated-ml-cluster"

try:
    # Try to get the cluster
    existing_cluster = ml_client.compute.get(cluster_name)
    print(f"Compute cluster '{cluster_name}' already exists.")
except Exception as e:
    # If it doesn't exist, create it
    print(f"Compute cluster '{cluster_name}' not found. Creating new cluster...")
    train_cluster = AmlCompute(
        name=cluster_name,
        type="amlcompute",
        size='Standard_E8s_v3',
        location='eastasia',
        min_instances=0,
        max_instances=3,
        idle_time_before_scale_down=100
    )
    ml_client.compute.begin_create_or_update(train_cluster)
    print(f"Compute cluster '{cluster_name}' creation started.")