# Create resource group

#!/bin/bash
resource_group_name ='e2e-fraud-detection'
location ='eastasia'

az login --service-principal 

az group show --name "$resource_group_name >/dev/null 2>&1 || \
az group create --name "$resource_group_name" --location "$location"