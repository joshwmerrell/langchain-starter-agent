"""Run Wallace's TUI with safe, idle-aware source reloads.

Unlike a plain watchfiles command, this supervisor waits for Wallace to finish
the current turn before restarting the child process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from watchfiles import Change, watch
from wallace_mcp import tui_applied_current_config


ROOT = Path(__file__).resolve().parent
TUI_SCRIPT = ROOT / "tui_agent.py"
STATUS_PATH = ROOT / ".wallace" / "reload_status"
MCP_CONFIG_PATH = ROOT / ".wallace" / "mcp_servers.json"
ETHICS_MANIFESTO_PATH = ROOT / "ethics_manifesto.md"
IDLE_SETTLE_SECONDS = 0.75
WATCH_SKIP_DIRS = {".git", ".wallace", ".venv", "venv", "node_modules", "__pycache__", "workspace"}


def _write_status(status: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=STATUS_PATH.parent,
            prefix=f".{STATUS_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(status)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, STATUS_PATH)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _is_busy() -> bool:
    try:
        return STATUS_PATH.read_text(encoding="utf-8").strip() != "idle"
    except OSError:
        return True


def _start_child() -> subprocess.Popen:
    environment = os.environ.copy()
    environment["WALLACE_RELOAD_STATUS"] = str(STATUS_PATH)
    environment["WALLACE_DEV_MODE"] = "1"
    _write_status("starting")
    return subprocess.Popen(
        [sys.executable, str(TUI_SCRIPT)],
        cwd=ROOT,
        env=environment,
    )


def _stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def _python_change(change: tuple[Change, str]) -> bool:
    _, path = change
    return Path(path).suffix == ".py"


def _changed_python_paths(changes: set[tuple[Change, str]]) -> list[Path]:
    """Return changed Python paths that are inside the project root."""
    paths = []
    for change in changes:
        if not _python_change(change):
            continue
        path = Path(change[1]).resolve()
        relative_parts = path.relative_to(ROOT).parts if path.is_relative_to(ROOT) else ()
        if relative_parts and not any(
            part.lower() in WATCH_SKIP_DIRS for part in relative_parts
        ):
            paths.append(path)
    return sorted(set(paths))


def _changes_compile(paths: list[Path]) -> bool:
    """Refuse to restart into a syntax-broken source tree."""
    for path in paths:
        if not path.exists():
            continue
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=10,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            print(
                f"Python change is not reloadable yet: {path.relative_to(ROOT)}\n"
                f"{details}",
                flush=True,
            )
            return False
    return True


def _changes_smoke() -> bool:
    """Import and construct the changed application before stopping the live TUI."""
    command = (
        "import main, tools, tui_agent; "
        "names = [tool.name for tool in tools.TOOLS]; "
        "assert len(names) == len(set(names)), 'duplicate local tool names'; "
        "tui_agent.AgentTUI()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=ROOT,
            env={**os.environ, "WALLACE_DEV_MODE": "1"},
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=25,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Source change failed startup preflight: {error}", flush=True)
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-3000:]
        print(f"Source change failed startup preflight:\n{detail}", flush=True)
        return False
    return True


def _wait_for_ready(child: subprocess.Popen, timeout: float = 35.0) -> bool:
    """Confirm the restarted child reaches its normal idle state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            print(f"Reloaded Wallace exited with code {child.returncode}; watcher remains active for a fix.", flush=True)
            return False
        if not _is_busy():
            return True
        time.sleep(0.2)
    print("Reloaded Wallace has not reported ready yet; watcher remains active.", flush=True)
    return False


def _config_change(changes: set[tuple[Change, str]]) -> bool:
    changed = any(Path(path).resolve() == MCP_CONFIG_PATH for _, path in changes)
    return changed and not tui_applied_current_config(ROOT)


def _manifesto_change(changes: set[tuple[Change, str]]) -> bool:
    return any(Path(path).resolve() == ETHICS_MANIFESTO_PATH for _, path in changes)


def _manifesto_valid() -> bool:
    from main import ethics_manifesto_prompt

    try:
        ethics_manifesto_prompt(ROOT)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Ethics manifesto is not reloadable yet: {error}", flush=True)
        return False
    return True


def _wait_for_idle(child: subprocess.Popen) -> bool:
    """Wait for a stable idle state before stopping the child."""
    idle_since: float | None = None
    while child.poll() is None:
        if not _is_busy():
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= IDLE_SETTLE_SECONDS:
                return True
        else:
            idle_since = None
        time.sleep(0.2)
    return False


def main() -> int:
    child = _start_child()
    crash_reported = False
    try:
        for changes in watch(
            ROOT,
            debounce=800,
            step=100,
            yield_on_timeout=True,
        ):
            if child.poll() is not None and not crash_reported:
                print(
                    f"Wallace exited with code {child.returncode}; the watcher will retry after a source fix.",
                    flush=True,
                )
                crash_reported = True
            config_changed = _config_change(changes)
            manifesto_changed = _manifesto_change(changes)
            if not config_changed and not manifesto_changed and not any(_python_change(change) for change in changes):
                continue

            python_paths = _changed_python_paths(changes)
            if not python_paths and not config_changed and not manifesto_changed:
                continue

            if child.poll() is None and _is_busy():
                print(
                    "Wallace is busy; source changes queued until the current "
                    "turn finishes.",
                    flush=True,
                )
                if not _wait_for_idle(child):
                    print("Wallace exited while a reload was queued; checking the new source.", flush=True)

            if not _changes_compile(python_paths):
                continue
            if not _changes_smoke():
                continue
            if manifesto_changed and not _manifesto_valid():
                continue

            if config_changed:
                from wallace_mcp import load_mcp_servers

                try:
                    load_mcp_servers(ROOT)
                except (OSError, ValueError) as error:
                    print(f"MCP configuration is not reloadable yet: {error}", flush=True)
                    continue

            print("Source change detected; reloading Wallace.", flush=True)
            if child.poll() is None:
                _stop_child(child)
            child = _start_child()
            crash_reported = False
            _wait_for_ready(child)
    except KeyboardInterrupt:
        _stop_child(child)
        return 0
    finally:
        if child.poll() is None:
            _stop_child(child)
        try:
            STATUS_PATH.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
