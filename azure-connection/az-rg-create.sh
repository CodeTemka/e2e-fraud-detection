# Create resource group

#!/bin/bash
resource_group_name ='e2e-fraud-detection'
location ='eastasia'

# This command let you login by prompting microsoft login page
az login 

az group show --name "$resource_group_name >/dev/null 2>&1 || \
az group create --name "$resource_group_name" --location "$location"