import base64
import json
import sys
from pathlib import Path

import requests
from nacl import encoding, public

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fraud_detection.config import get_settings  # noqa: E402
from fraud_detection.utils.logging import get_logger  # noqa: E402

settings = get_settings()
settings.require_azure()
logger = get_logger(__name__)

def require_setting(value: str | None, env_var: str) -> str:
    """Return the required setting or raise a clear error."""

    if not value:
        raise ValueError(f"Missing required environment variable: {env_var}")
    return value


def optional_setting(value: str | None, env_var: str) -> str | None:
    """Return the optional setting, logging a hint when absent."""

    if value:
        return value

    logger.warning("Optional environment variable %s is not set; skipping.", env_var)
    return None


GITHUB_TOKEN = require_setting(settings.github_token, "GITHUB_TOKEN")
OWNER = require_setting(settings.github_owner, "GITHUB_OWNER")
REPO = require_setting(settings.github_repo, "GITHUB_REPO")
SP_CREDENTIALS = Path(__file__).resolve().parent.parent / "json" / "sp_credentials.json"

if not SP_CREDENTIALS.exists():
    raise FileNotFoundError(
        f"Service principal credentials not found at {SP_CREDENTIALS}. "
        "Run scripts/az_prep.py to create the service principal and workspace, "
        "then rerun this script."
    )

with open(SP_CREDENTIALS, encoding="utf-8") as f:
    sp_credentials = json.load(f)

url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/public-key"

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "e2e-fraud-detection-secret-uploader",
}

response = requests.get(url, headers=headers)
if response.status_code == 401:
    raise PermissionError(
        "Failed to authenticate with GitHub API."
        " Verify that GITHUB_TOKEN has 'repo' and 'actions' scopes."
    )
if response.status_code != 200:
    raise Exception(f"Failed to get public key: {response.text}")

public_key_id = response.json()["key_id"]
public_key_str = response.json()["key"]

public_key = public.PublicKey(public_key_str.encode("utf-8"), encoding.Base64Encoder())
sealed_box = public.SealedBox(public_key)

workspace_location = require_setting(settings.location, "LOCATION")
instance_type = settings.instance_type or "Standard_E4s_v3"
instance_count = str(settings.instance_count or 1)

required_secrets: dict[str, str] = {
    "TENANTID": require_setting(sp_credentials.get("tenantId"), "TENANTID"),
    "CLIENTID": require_setting(sp_credentials.get("clientId"), "CLIENTID"),
    "CLIENTSECRET": require_setting(
        sp_credentials.get("clientSecret"), "CLIENTSECRET"
    ),
    "SUBSCRIPTIONID": require_setting(
        sp_credentials.get("subscriptionId"), "SUBSCRIPTIONID"
    ),
    "RESOURCE_GROUP": require_setting(settings.resource_group, "RESOURCE_GROUP"),
    "WORKSPACE_NAME": require_setting(settings.workspace_name, "WORKSPACE_NAME"),
    "LOCATION": workspace_location,
    "AML_COMPUTE": require_setting(settings.default_compute, "AML_COMPUTE"),
    "AML_DATASET": require_setting(settings.default_dataset, "AML_DATASET"),
    "INSTANCE_TYPE": instance_type,
    "INSTANCE_COUNT": instance_count,
}

optional_secrets = {
    "KAGGLE_USERNAME": optional_setting(settings.kaggle_username, "KAGGLE_USERNAME"),
    "KAGGLE_KEY": optional_setting(settings.kaggle_key, "KAGGLE_KEY"),
}

secrets: dict[str, str] = {}
secrets.update(required_secrets)
for name, value in optional_secrets.items():
    if value is not None:
        secrets[name] = value


def secret_exists(secret_name: str) -> bool:
    check_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/{secret_name}"
    resp = requests.get(check_url, headers=headers)
    return resp.status_code == 200


# Upload or update each secret
for name, value in secrets.items():
    existed = secret_exists(name)

    # Encrypt secret
    encrypted = sealed_box.encrypt(value.encode("utf-8"))
    encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")

    # Upload secret
    put_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/{name}"
    payload = {"encrypted_value": encrypted_b64, "key_id": public_key_id}
    put_resp = requests.put(put_url, headers=headers, json=payload)

    if put_resp.status_code in (201, 204):
        action = "updated" if existed else "created"
        logger.info("Secret '%s' %s successfully.", name, action)
    else:
        logger.error(
            "Failed to upload %s: %s - %s", name, put_resp.status_code, put_resp.text
        )
