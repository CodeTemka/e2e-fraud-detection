"""Helpers for creating or updating GitHub Actions secrets.

The CD pipeline relies on two secrets that can change frequently:

``MODEL_NAME``
    Stable Azure ML model name promoted for deployment.
``ENDPOINT_NAME``
    Target managed online endpoint name used during rollout.

This module exposes a lightweight helper that encrypts and uploads those
values (or any other secrets) using the GitHub REST API.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import requests
from nacl import encoding, public

from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def require_setting(value: str | None, env_var: str) -> str:
    """Return the required setting or raise a clear error."""

    if not value:
        raise ValueError(f"Missing required environment variable: {env_var}")
    return value


@dataclass
class GitHubSecretManager:
    """Encrypt and upload GitHub Actions secrets for a repository."""

    token: str
    owner: str
    repo: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "GitHubSecretManager":
        cfg = (settings or get_settings()).require_github()
        return cls(
            token=require_setting(cfg.github_token, "GITHUB_TOKEN"),
            owner=require_setting(cfg.github_owner, "GITHUB_OWNER"),
            repo=require_setting(cfg.github_repo, "GITHUB_REPO"),
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "e2e-fraud-detection-secret-uploader",
        }

    def _get_public_key(self) -> tuple[str, str]:
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/secrets/public-key"
        response = requests.get(url, headers=self._headers, timeout=30)

        if response.status_code == 401:
            raise PermissionError(
                "Failed to authenticate with GitHub API. Verify that GITHUB_TOKEN has 'repo' and 'actions' scopes."
            )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get public key: {response.text}")

        payload = response.json()
        return payload["key_id"], payload["key"]

    def _encrypt(self, public_key: str, value: str) -> str:
        public_key_obj = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
        sealed_box = public.SealedBox(public_key_obj)
        encrypted = sealed_box.encrypt(value.encode("utf-8"))
        return base64.b64encode(encrypted).decode("utf-8")

    def secret_exists(self, secret_name: str) -> bool:
        check_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/secrets/{secret_name}"
        resp = requests.get(check_url, headers=self._headers, timeout=30)
        return resp.status_code == 200

    def set_secret(self, name: str, value: str | dict[str, Any]) -> None:
        """Encrypt and upload a secret value (string or JSON-serializable mapping)."""

        if isinstance(value, dict):
            value = json.dumps(value)

        existed = self.secret_exists(name)
        key_id, public_key = self._get_public_key()
        encrypted_b64 = self._encrypt(public_key, value)

        put_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/secrets/{name}"
        payload = {"encrypted_value": encrypted_b64, "key_id": key_id}

        put_resp = requests.put(put_url, headers=self._headers, json=payload, timeout=30)

        if put_resp.status_code not in (201, 204):
            raise RuntimeError(
                f"Failed to upload secret '{name}': {put_resp.status_code} - {put_resp.text}"
            )

        action = "updated" if existed else "created"
        logger.info("Secret '%s' %s successfully.", name, action)


def update_model_and_endpoint_secrets(
    *,
    manager: GitHubSecretManager,
    model_name: str,
    endpoint_name: str,
) -> None:
    """Update GitHub secrets required by the CD deployment workflow."""

    manager.set_secret("MODEL_NAME", model_name)
    manager.set_secret("ENDPOINT_NAME", endpoint_name)


__all__ = ["GitHubSecretManager", "update_model_and_endpoint_secrets", "require_setting"]
