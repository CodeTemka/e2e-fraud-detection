import mltable
import os
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

training_data = ml_client.data.get(name='creditcard-data', version='1')
csv_path = training_data.path

train_table = mltable.from_delimited_files(paths=[{"file": csv_path}], delimiter=',', header='all_files_same_headers')
train_table.save('mltable')

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

mltable_asset = Data(
    name="mltable_creditcard_data",
    description="MLTable version of credit card data",
    path="mltable",
    type=AssetTypes.MLTABLE
)

ml_client.data.create_or_update(mltable_asset)
