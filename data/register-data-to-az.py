from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure_connection.ml_client_setup import ml_client

data_name = "creditcard-data"
version = "1"

data_asset = Data(
    name=data_name,
    path="./creditcard.csv", 
    description="Credit Card Fraud Detection Dataset",
    type=AssetTypes.URI_FILE,
    version=version,
)

try:
    registered_data = ml_client.data.get(name=data_name, version=version)
    print(f"Data asset '{data_name}' version '{version}' already exists. Skipping registration.")
except Exception as e:
    registered_data = ml_client.data.create_or_update(data_asset)
    print(f"Registered data asset '{data_name}' version '{version}'.")
