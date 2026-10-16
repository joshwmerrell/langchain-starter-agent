"""Host-only key metadata; values live in the OS keyring or user environment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from wallace_models import CLOUD_PROVIDERS, environment_key

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}\Z")
_MAX_BYTES = 20_000


@dataclass(frozen=True)
class KeyEntry:
    name: str
    provider: str
    source: str
    enabled: bool
    available: bool


def _service(root: Path) -> str:
    digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]
    return f"wallace-{digest}"


def _path(root: Path) -> Path:
    return root.resolve() / ".wallace" / "key_registry.json"


def _read(root: Path) -> dict:
    path = _path(root)
    if path.is_symlink():
        raise ValueError("API key registry cannot be a symlink")
    try:
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("API key registry is too large")
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"keys": [], "environment_enabled": {}}
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("API key registry is invalid") from error
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise ValueError("API key registry is invalid")
    if not isinstance(data.get("environment_enabled", {}), dict):
        raise ValueError("API key registry is invalid")
    for item in data["keys"]:
        if not isinstance(item, dict) or not _NAME.fullmatch(str(item.get("name", ""))):
            raise ValueError("API key registry is invalid")
        if item.get("provider") not in CLOUD_PROVIDERS or not isinstance(item.get("enabled"), bool):
            raise ValueError("API key registry is invalid")
        if "key" in item or "secret" in item or "value" in item:
            raise ValueError("API key registry must not contain secret values")
    return data


def _save(root: Path, data: dict) -> None:
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = json.dumps(data, indent=2) + "\n"
    if len(content.encode()) > _MAX_BYTES:
        raise ValueError("API key registry is too large")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".key_registry.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ring():
    try:
        import keyring
    except ImportError:
        raise RuntimeError("OS keyring support is not installed") from None
    return keyring


def list_keys(root: Path) -> list[KeyEntry]:
    data = _read(root)
    entries = []
    for provider in sorted(CLOUD_PROVIDERS):
        entries.append(KeyEntry(
            f"environment-{provider}", provider, "environment",
            data.get("environment_enabled", {}).get(provider, True),
            environment_key(provider) is not None,
        ))
    entries.extend(
        KeyEntry(item["name"], item["provider"], "keyring", item["enabled"], True)
        for item in data["keys"]
    )
    return entries


def add_key(root: Path, name: str, provider: str, secret: str) -> None:
    if provider not in CLOUD_PROVIDERS:
        raise ValueError("provider must be openai or gemini")
    if not _NAME.fullmatch(name) or name.lower().startswith(("sk-", "aiza", "environment-")):
        raise ValueError("key name must be short and alphanumeric; never put a key in the name")
    if not secret or any(char.isspace() for char in secret):
        raise ValueError("enter a nonempty API key without whitespace")
    data = _read(root)
    if any(entry.name.casefold() == name.casefold() for entry in list_keys(root)):
        raise ValueError("API key name already exists")
    try:
        _ring().set_password(_service(root), name, secret)
    except Exception:
        raise RuntimeError("OS keyring could not save the API key") from None
    data["keys"].append({"name": name, "provider": provider, "enabled": True})
    for item in data["keys"]:
        if item["provider"] == provider and item["name"] != name:
            item["enabled"] = False
    data.setdefault("environment_enabled", {})[provider] = False
    try:
        _save(root, data)
    except Exception:
        try:
            _ring().delete_password(_service(root), name)
        except Exception:
            pass
        raise


def set_key_active(root: Path, name: str, active: bool) -> None:
    data = _read(root)
    entry = next((item for item in list_keys(root) if item.name.casefold() == name.casefold()), None)
    if entry is None:
        raise ValueError("unknown API key name")
    if active and not entry.available:
        raise ValueError("API key is not available")
    if active and entry.source == "keyring":
        try:
            if not _ring().get_password(_service(root), entry.name):
                raise ValueError("API key is missing from the OS keyring")
        except ValueError:
            raise
        except Exception:
            raise RuntimeError("OS keyring could not read the API key") from None
    for item in data["keys"]:
        if item["provider"] != entry.provider:
            continue
        if active:
            item["enabled"] = item["name"] == entry.name
        elif item["name"] == entry.name:
            item["enabled"] = False
    environment_enabled = data.setdefault("environment_enabled", {})
    if active:
        environment_enabled[entry.provider] = entry.source == "environment"
    elif entry.source == "environment":
        environment_enabled[entry.provider] = False
    _save(root, data)


def remove_key(root: Path, name: str) -> None:
    data = _read(root)
    entry = next((item for item in list_keys(root) if item.name.casefold() == name.casefold()), None)
    if entry is None:
        raise ValueError("unknown API key name")
    if entry.source == "environment":
        raise ValueError("environment keys must be removed from .env or the environment by the user")
    try:
        _ring().delete_password(_service(root), entry.name)
    except Exception:
        raise RuntimeError("OS keyring could not remove the API key") from None
    data["keys"] = [item for item in data["keys"] if item["name"] != entry.name]
    _save(root, data)


def active_key(root: Path, provider: str) -> SecretStr | None:
    if provider not in CLOUD_PROVIDERS:
        return None
    for entry in list_keys(root):
        if entry.provider != provider or not entry.enabled or not entry.available:
            continue
        if entry.source == "environment":
            return environment_key(provider)
        try:
            value = _ring().get_password(_service(root), entry.name)
        except Exception:
            raise RuntimeError("OS keyring could not read the API key") from None
        return SecretStr(value) if value else None
    return None


def parse_key_command(args: list[str]) -> tuple[str, str, str]:
    if not args or args == ["list"]:
        return "list", "", ""
    operation = args[0].lower()
    if operation == "add" and len(args) == 3:
        return "add", args[1], args[2].lower()
    if operation in {"on", "off", "remove"} and len(args) == 2:
        return operation, args[1], ""
    raise ValueError("use /keys, /key add <name> <openai|gemini>, or /key on|off|remove <name>")
