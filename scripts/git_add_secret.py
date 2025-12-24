from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests
from nacl import encoding, public

from fraud_detection.config import Settings, get_settings


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",  # most compatible
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "e2e-fraud-detection-secret-uploader",
    }


def _get_public_key(owner: str, repo: str, token: str) -> tuple[str, str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key"
    r = requests.get(url, headers=_headers(token), timeout=30)

    if r.status_code in (401, 403):
        raise PermissionError("GitHub token is not authorized to manage secrets for this repo.")
    if r.status_code != 200:
        raise RuntimeError(f"Failed to get public key: {r.status_code} - {r.text}")

    data = r.json()
    return data["key_id"], data["key"]  # key is base64


def _encrypt(public_key_b64: str, plaintext: str) -> str:
    pk = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed = public.SealedBox(pk)
    encrypted_bytes = sealed.encrypt(plaintext.encode("utf-8"))
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def set_github_secret(
    name: str,
    value: str | dict[str, Any],
    *,
    settings: Settings | None = None,
) -> None:
    cfg = (settings or get_settings()).require_github()

    token = cfg.github_token or ""
    owner = cfg.github_owner or ""
    repo = cfg.github_repo or ""

    if isinstance(value, dict):
        value = json.dumps(value)

    key_id, public_key = _get_public_key(owner, repo, token)
    encrypted_value = _encrypt(public_key, value)

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{name}"
    payload = {"encrypted_value": encrypted_value, "key_id": key_id}
    r = requests.put(url, headers=_headers(token), json=payload, timeout=30)

    if r.status_code not in (201, 204):
        raise RuntimeError(f"Failed to upload secret '{name}': {r.status_code} - {r.text}")

    print(f"OK: secret '{name}' uploaded.")


def update_azure_credentials_secret(sp_path: Path = Path("sp_credentials.json")) -> None:
    sp = json.loads(sp_path.read_text(encoding="utf-8"))

    # supports common Azure outputs
    client_id = sp.get("clientId") or sp.get("appId")
    tenant_id = sp.get("tenantId")
    client_secret = sp.get("clientSecret") or sp.get("password")
    subscription_id = sp.get("subscriptionId")

    missing = [k for k, v in {
        "clientId/appId": client_id,
        "tenantId": tenant_id,
        "clientSecret/password": client_secret,
        "subscriptionId": subscription_id,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing fields in {sp_path}: {', '.join(missing)}")

    azure_credentials = {
        "clientId": client_id,
        "clientSecret": client_secret,
        "tenantId": tenant_id,
        "subscriptionId": subscription_id,
    }

    set_github_secret("AZURE_CREDENTIALS", azure_credentials)


if __name__ == "__main__":
    update_azure_credentials_secret()
