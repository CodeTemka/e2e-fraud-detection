#NOTE: I am deleting online deployment to save my cost on azure subscription

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from azure_connection.ml_client_setup import ml_client

endpoint_name = 'fraud-detection-1df0'

ml_client.online_endpoints.begin_delete(name=endpoint_name)