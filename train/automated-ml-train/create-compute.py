from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from dotenv import load_dotenv
import os

load_dotenv()

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

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
        location='easasia',
        min_instances=0,
        max_instances=3,
        idle_time_before_scale_down=100
    )
    ml_client.compute.begin_create_or_update(train_cluster)
    print(f"Compute cluster '{cluster_name}' creation started.")