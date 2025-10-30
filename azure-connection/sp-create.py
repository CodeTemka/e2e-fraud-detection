# By creating service principal, you can use it to authenticate applications and services to access Azure resources securely.
# In comparison to az login, service principal is more suitable for automated workflows and scenarios where you need to manage access without user interaction.
# First login to your azure account using az login command in terminal before creating service principal.

import os
from dotenv import load_dotenv
import subprocess
import json

# Load environment variables
load_dotenv()

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = "e2e-fraud-detection"
sp_name = 'e2e-fraud-detection-sp'
role = 'Contributor'
output_file = 'sp-credentials.json'

# in case of Windows, specify the full path to az.cmd
az_path = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"


# Function to check if SP exists
def sp_exists(sp_name):
    cmd = [
        az_path, "ad", "sp", "list",
        "--display-name", sp_name,
        "--query", "[].appId",
        "-o", "tsv"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    app_id = result.stdout.strip()
    return bool(app_id)


# Create the service principal if it doesn't exist
if sp_exists(sp_name):
    print(f"Service principal '{sp_name}' already exists. Skipping creation.")
else:
    print(f"Creating service principal '{sp_name}'...")
    cmd = [
        az_path, "ad", "sp", "create-for-rbac",
        "--name", sp_name,
        "--role", role,
        "--scopes", f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}",
        "--sdk-auth"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Save SP credentials to JSON
    with open(output_file, "w") as f:
        f.write(result.stdout)

    print(f"Service principal credentials saved to '{output_file}'")