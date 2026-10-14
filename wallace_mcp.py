"""Host-managed MCP connections and persisted user choices.

The model gets a read-only listing tool, not these mutation functions. Project
file tools exclude .wallace; approved shell commands are not an OS sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


CONFIG_PATH = Path(".wallace/mcp_servers.json")
TUI_CHANGE_MARKER = Path(".wallace/mcp_servers.tui-sha256")
BUILTIN_NAME = "langchain-docs"
BUILTIN_URL = "https://docs.langchain.com/mcp"
MAX_CONFIG_BYTES = 16_384
MAX_SERVERS = 8
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{0,39}\Z")
_connection_status: dict[str, str] = {}


@dataclass(frozen=True)
class MCPServer:
    name: str
    url: str
    keywords: tuple[str, ...]
    enabled: bool = True
    builtin: bool = False


def _configuration_path(root: Path) -> Path:
    path = root / CONFIG_PATH
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("MCP configuration must stay inside the workspace")
    return path


def _validate_url(name: str, url: object) -> None:
    if not isinstance(url, str) or not url or any(char.isspace() or ord(char) < 32 for char in url):
        raise ValueError(f"MCP server {name} needs a valid URL")
    try:
        parsed = urlsplit(url)
        valid_port = parsed.port is None or 0 < parsed.port <= 65535
    except ValueError:
        valid_port = False
        parsed = urlsplit("")
    if (
        not parsed.hostname
        or not valid_port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.scheme not in {"http", "https"}
        or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ValueError(f"MCP server {name} needs HTTPS or loopback HTTP without credentials in the URL")


def _validate_config(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("servers"), list):
        raise ValueError("MCP configuration needs a servers list")
    if not isinstance(value.get("langchain_docs_enabled", True), bool):
        raise ValueError("langchain_docs_enabled must be a boolean")
    if len(value["servers"]) > MAX_SERVERS:
        raise ValueError(f"MCP configuration supports at most {MAX_SERVERS} servers")
    seen = {BUILTIN_NAME}
    for entry in value["servers"]:
        if not isinstance(entry, dict):
            raise ValueError("MCP server entries must be objects")
        if not isinstance(entry.get("enabled", False), bool):
            raise ValueError("MCP server enabled must be a boolean")
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("MCP server name must be short and alphanumeric")
        if name.casefold() in seen:
            raise ValueError(f"duplicate or reserved MCP server name: {name}")
        _validate_url(name, entry.get("url"))
        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list) or len(keywords) > 12 or any(
            not isinstance(word, str) or not _NAME.fullmatch(word)
            for word in keywords
        ):
            raise ValueError(f"MCP server {name} has invalid keywords")
        seen.add(name.casefold())
    return value


def read_mcp_configuration(root: Path) -> dict:
    """Read and validate all entries, including inactive ones."""
    path = _configuration_path(root)
    if not path.exists():
        return {"servers": [], "langchain_docs_enabled": True}
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("MCP configuration is too large")
    return _validate_config(json.loads(path.read_text(encoding="utf-8")))


def save_mcp_configuration(root: Path, value: dict, *, live_applied: bool = False) -> None:
    """Atomically persist a validated registry, noting if the TUI applied it."""
    _validate_config(value)
    content = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    if len(content) > MAX_CONFIG_BYTES:
        raise ValueError("MCP configuration is too large")
    path = _configuration_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mcp_servers.", delete=False) as file:
            temporary = Path(file.name)
            os.chmod(temporary, 0o600)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # The dev watcher skips only a change applied by the running TUI. CLI
    # changes still need to restart a separate development TUI process.
    marker = path.parent / TUI_CHANGE_MARKER.name
    if not live_applied:
        marker.unlink(missing_ok=True)
        return
    marker_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mcp_marker.", delete=False) as file:
            marker_temp = Path(file.name)
            os.chmod(marker_temp, 0o600)
            file.write(hashlib.sha256(content).hexdigest().encode("ascii"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(marker_temp, marker)
    except OSError:
        pass
    finally:
        if marker_temp is not None:
            marker_temp.unlink(missing_ok=True)


def tui_applied_current_config(root: Path) -> bool:
    path = _configuration_path(root)
    marker = path.parent / TUI_CHANGE_MARKER.name
    try:
        return marker.read_text(encoding="ascii").strip() == hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False


def all_mcp_servers(root: Path) -> list[MCPServer]:
    """Return the built-in endpoint and all saved user endpoints."""
    value = read_mcp_configuration(root)
    servers = [MCPServer(
        BUILTIN_NAME, BUILTIN_URL, ("langchain", "docs", "mcp"),
        value.get("langchain_docs_enabled", True), True,
    )]
    servers.extend(
        MCPServer(
            entry["name"], entry["url"],
            tuple(word.casefold() for word in entry.get("keywords", [])),
            entry.get("enabled", False), False,
        )
        for entry in value["servers"]
    )
    return servers


def load_mcp_servers(root: Path) -> list[MCPServer]:
    """Load active user endpoints (the built-in docs server is separate)."""
    return [server for server in all_mcp_servers(root) if server.enabled and not server.builtin]


def add_mcp_server(
    root: Path, name: str, url: str, keywords: tuple[str, ...] = (),
    *, live_applied: bool = False,
) -> None:
    value = read_mcp_configuration(root)
    value["servers"].append({
        "name": name, "url": url, "enabled": True, "keywords": list(keywords),
    })
    save_mcp_configuration(root, value, live_applied=live_applied)


def remove_mcp_server(root: Path, name: str, *, live_applied: bool = False) -> None:
    if name.casefold() == BUILTIN_NAME:
        raise ValueError("The built-in LangChain server cannot be removed; use /mcp off langchain-docs")
    value = read_mcp_configuration(root)
    remaining = [entry for entry in value["servers"] if entry["name"].casefold() != name.casefold()]
    if len(remaining) == len(value["servers"]):
        raise ValueError(f"unknown MCP server: {name}")
    value["servers"] = remaining
    save_mcp_configuration(root, value, live_applied=live_applied)
    _connection_status.pop(name.casefold(), None)


def set_mcp_server_active(
    root: Path, name: str, active: bool, *, live_applied: bool = False,
) -> bool:
    """Persist an active choice; return whether the choice changed."""
    value = read_mcp_configuration(root)
    if name.casefold() == BUILTIN_NAME:
        if value.get("langchain_docs_enabled", True) == active:
            return False
        value["langchain_docs_enabled"] = active
    else:
        entry = next((item for item in value["servers"] if item["name"].casefold() == name.casefold()), None)
        if entry is None:
            raise ValueError(f"unknown MCP server: {name}")
        if entry.get("enabled", False) == active:
            return False
        entry["enabled"] = active
    save_mcp_configuration(root, value, live_applied=live_applied)
    if not active:
        _connection_status.pop(name.casefold(), None)
    return True


def parse_mcp_command(args: list[str], root: Path) -> tuple[str, str, str, tuple[str, ...]]:
    """Parse user-interface arguments, including a toggle against saved state."""
    if not args or (args[0].lower() in {"list", "status"} and len(args) == 1):
        return "list", "", "", ()
    operation = args[0].lower()
    if operation == "add" and len(args) >= 3:
        keywords = tuple(word for token in args[3:] for word in token.split(",") if word)
        return operation, args[1], args[2], keywords
    if operation in {"remove", "on", "off", "toggle"} and len(args) == 2:
        if operation == "toggle":
            server = next(
                (item for item in all_mcp_servers(root) if item.name.casefold() == args[1].casefold()),
                None,
            )
            if server is None:
                raise ValueError(f"unknown MCP server: {args[1]}")
            operation = "off" if server.enabled else "on"
        return operation, args[1], "", ()
    if operation == "reload" and len(args) == 1:
        return operation, "", "", ()
    raise ValueError(
        "use /mcp [list], /mcp add <name> <URL> [keywords...], "
        "/mcp on|off|toggle|remove <name>, or /mcp reload"
    )


def apply_mcp_command(
    root: Path, operation: str, name: str, url: str = "", keywords: tuple[str, ...] = (),
    *, live_applied: bool = False,
) -> bool:
    """Apply a validated user command; return whether the saved registry changed."""
    if operation == "add":
        add_mcp_server(root, name, url, keywords, live_applied=live_applied)
        return True
    if operation == "remove":
        remove_mcp_server(root, name, live_applied=live_applied)
        return True
    if operation in {"on", "off"}:
        return set_mcp_server_active(root, name, operation == "on", live_applied=live_applied)
    if operation == "reload":
        return False
    raise ValueError(f"unknown MCP command: {operation}")


def clear_connection_statuses() -> None:
    _connection_status.clear()


def snapshot_connection_statuses() -> dict[str, str]:
    return dict(_connection_status)


def restore_connection_statuses(statuses: dict[str, str]) -> None:
    _connection_status.clear()
    _connection_status.update(statuses)


def set_connection_status(name: str, connected: bool) -> None:
    _connection_status[name.casefold()] = "connected" if connected else "not connected"


def connection_status(server: MCPServer) -> str:
    return "inactive" if not server.enabled else _connection_status.get(server.name.casefold(), "not connected")
