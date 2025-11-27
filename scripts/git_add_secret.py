import base64
import requests
from nacl import encoding, public
import json
from pathlib import Path

from fraud_detection.config import get_settings

settings = get_settings()

GITHUB_TOKEN = settings.github_token
OWNER = settings.github_owner
REPO = settings.github_repo
SECRETS = Path('json') / 'sp_credentials.json'

with open(SECRETS, 'r') as f:
    secrets = json.load(f)

url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/public-key"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

response = requests.get(url, headers=headers)
if response.status_code != 200:
    raise Exception(f"Failed to get public key: {response.text}")

public_key_id = response.json()["key_id"]
public_key_str = response.json()["key"]

public_key = public.PublicKey(public_key_str.encode("utf-8"), encoding.Base64Encoder())
sealed_box = public.SealedBox(public_key)

def secret_exists(secret_name):
    check_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/{secret_name}"
    resp = requests.get(check_url, headers=headers)
    return resp.status_code == 200

# Upload or update each secret
for name, value in secrets.items():
    # Encrypt secret
    encrypted = sealed_box.encrypt(value.encode("utf-8"))
    encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")

    # Upload secret
    put_url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/secrets/{name}"
    payload = {"encrypted_value": encrypted_b64, "key_id": public_key_id}
    put_resp = requests.put(put_url, headers=headers, json=payload)

    if put_resp.status_code in (201, 204):
        if secret_exists(name):
            print(f"Secret '{name}' updated successfully.")
        else:
            print(f"Secret '{name}' created successfully.")
    else:
        print(f"Failed to upload {name}: {put_resp.status_code} - {put_resp.text}")