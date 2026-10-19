"""Model selection and host-held cloud credentials shared by CLI and TUI."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from functools import lru_cache
from pathlib import Path

import httpx
from pydantic import SecretStr


_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:-]{0,199}\Z")
CLOUD_PROVIDERS = {"openai", "gemini"}
CLOUD_KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}


@lru_cache(maxsize=32)
def _ollama_thinking_supported(base_url: str, model: str) -> bool:
    """Probe model metadata once; offline/unrecognized models keep their default."""
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/api/show", json={"model": model}, timeout=2,
        )
        response.raise_for_status()
        return "thinking" in response.json().get("capabilities", [])
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return False


def ollama_reasoning(base_url: str, model: str) -> bool | None:
    """Keep supported reasoning separate from final text without changing models."""
    setting = os.getenv("WALLACE_REASONING", "auto").strip().lower()
    if setting in {"true", "1", "on"}:
        return True
    if setting in {"false", "0", "off"}:
        return False
    if setting == "auto" and _ollama_thinking_supported(base_url, model):
        return True
    return None


class MissingCredentialError(ValueError):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"{provider} needs an active API key")


class CloudCredentials:
    """Hold provider keys in host memory, never in model-facing messages."""

    def __init__(self) -> None:
        self._secrets: dict[str, SecretStr] = {}

    def set(self, provider: str, key: str) -> None:
        if provider not in CLOUD_PROVIDERS or not key or any(char.isspace() for char in key):
            raise ValueError("Enter a nonempty API key without whitespace")
        self._secrets[provider] = SecretStr(key)

    def get(self, provider: str) -> SecretStr | None:
        return self._secrets.get(provider)

    def clear(self, provider: str | None = None) -> None:
        if provider is None:
            self._secrets.clear()
        else:
            self._secrets.pop(provider, None)

    def redact(self, message: str) -> str:
        for secret in self._secrets.values():
            value = secret.get_secret_value()
            if value:
                message = message.replace(value, "[REDACTED API KEY]")
        return message


def use_active_key(credentials: CloudCredentials, provider: str, root: Path) -> bool:
    """Load a user-enabled environment or keyring key into host memory."""
    from wallace_keys import active_key

    secret = active_key(root, provider)
    if secret is None:
        credentials.clear(provider)
        return False
    credentials.set(provider, secret.get_secret_value())
    return True


def environment_key(provider: str) -> SecretStr | None:
    """Read a configured key without exposing its value in status messages."""
    name = CLOUD_KEY_VARIABLES.get(provider)
    value = os.getenv(name) if name else None
    if not value or any(char.isspace() for char in value):
        return None
    return SecretStr(value)


def use_environment_key(credentials: CloudCredentials, provider: str) -> bool:
    """Use a configured key for the selected provider, if one exists."""
    secret = environment_key(provider)
    if secret is None:
        return False
    credentials.set(provider, secret.get_secret_value())
    return True


def valid_model_name(name: object) -> bool:
    """Accept Ollama model identifiers, not whitespace or terminal controls."""
    return isinstance(name, str) and _MODEL_NAME.fullmatch(name) is not None


def parse_model_selection(selection: str) -> tuple[str, str]:
    """Distinguish Ollama names from openai:<id> and gemini:<id>."""
    if not valid_model_name(selection):
        raise ValueError("invalid model name")
    provider, separator, model_id = selection.partition(":")
    if provider.lower() in CLOUD_PROVIDERS and separator:
        if not valid_model_name(model_id):
            raise ValueError(f"{provider.lower()} requires a model ID after ':'")
        if model_id.lower().startswith(("sk-", "sk_", "aiza")):
            raise ValueError("That looks like an API key; use the masked /key dialog")
        return provider.lower(), model_id
    return "ollama", selection


def load_model_choice(project_root: Path, default_model: str) -> str:
    """A saved in-app choice takes precedence over the environment default."""
    try:
        data = json.loads(
            (project_root / ".wallace" / "model.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default_model
    choice = data.get("model") if isinstance(data, dict) else None
    try:
        parse_model_selection(choice)
    except (TypeError, ValueError):
        return default_model
    return choice


def save_model_choice(project_root: Path, model_name: str) -> None:
    """Atomically save a selected model without touching the user's .env."""
    parse_model_selection(model_name)
    path = project_root / ".wallace" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".model.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"model": model_name}, temporary)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def list_installed_models(base_url: str) -> list[str]:
    """Query the configured Ollama server for locally available models."""
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Ollama returned an invalid model list")
    names = {
        name
        for entry in entries
        if isinstance(entry, dict)
        for name in [entry.get("name") or entry.get("model")]
        if valid_model_name(name)
    }
    return sorted(names, key=str.casefold)


async def list_cloud_models(provider: str, api_key: SecretStr) -> list[str]:
    """List account-visible model IDs without placing a key in a request URL."""
    if provider not in CLOUD_PROVIDERS:
        raise ValueError("unknown cloud provider")
    if provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"}
    else:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": api_key.get_secret_value()}

    names: set[str] = set()
    page_token: str | None = None
    seen_tokens: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(20):
                params = {"pageSize": 1000, **({"pageToken": page_token} if page_token else {})} if provider == "gemini" else None
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("invalid model list response")
                entries = data.get("data" if provider == "openai" else "models")
                if not isinstance(entries, list):
                    raise ValueError("invalid model list response")
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if provider == "openai":
                        name = entry.get("id")
                    else:
                        methods = entry.get("supportedGenerationMethods")
                        if not isinstance(methods, list) or "generateContent" not in methods:
                            continue
                        name = entry.get("name")
                        if not isinstance(name, str) or not name.startswith("models/"):
                            continue
                        name = name.removeprefix("models/")
                    if valid_model_name(name):
                        names.add(name)
                if provider == "openai":
                    break
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                if not isinstance(page_token, str) or page_token in seen_tokens:
                    raise ValueError("invalid model list pagination")
                seen_tokens.add(page_token)
            else:
                raise ValueError("model list exceeded pagination limit")
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f"{provider} model list request failed (HTTP {error.response.status_code})") from None
    except httpx.RequestError:
        raise RuntimeError(f"Could not reach {provider} model list") from None
    return sorted(names, key=str.casefold)
