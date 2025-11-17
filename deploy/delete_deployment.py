#NOTE: I am deleting online deployment to save my cost on azure subscription
#TIP: Managed online endpoints can be configured to scale down to zero instances during periods of inactivity to save cost, 
# but this feature (known as Auto Scaling to Zero or Cold Start) is only available for Azure Machine Learning Serverless inference deployments, 
# not all Managed Online Deployments.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client

endpoint_name = 'fraud-detection-1df0'

ml_client.online_endpoints.begin_delete(name=endpoint_name)