"""Host-side roots and launch mode; never accept a model-provided mode switch."""

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
DEV_MODE = os.environ.get("WALLACE_DEV_MODE") == "1"
WORKSPACE_ROOT = APP_ROOT / "workspace"
DEV_THREAD_ID = "wallace_dev"


def ensure_workspace() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def is_source_thread(thread_id: str, *, development: bool | None = None) -> bool:
    """Development is a launcher capability, but source access is per thread."""
    return (DEV_MODE if development is None else development) and thread_id == DEV_THREAD_ID


def thread_workspace(
    thread_id: str, *, project_root: Path | None = None, development: bool | None = None,
) -> Path:
    root = project_root or APP_ROOT
    return root if is_source_thread(thread_id, development=development) else root / "workspace"


def plan_session_id(thread_id: str, *, development: bool | None = None) -> str:
    # Thread IDs cannot contain dots, so this cannot collide with a legacy
    # development thread's plan session ID.
    return thread_id if is_source_thread(thread_id, development=development) else f"workspace.{thread_id}"
