"""Run Wallace's TUI with safe, idle-aware source reloads.

Unlike a plain watchfiles command, this supervisor waits for Wallace to finish
the current turn before restarting the child process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from watchfiles import Change, watch


ROOT = Path(__file__).resolve().parent
TUI_SCRIPT = ROOT / "tui_agent.py"
STATUS_PATH = ROOT / ".wallace" / "reload_status"


def _write_status(status: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(status, encoding="utf-8")


def _is_busy() -> bool:
    try:
        return STATUS_PATH.read_text(encoding="utf-8").strip() != "idle"
    except OSError:
        return True


def _start_child() -> subprocess.Popen:
    environment = os.environ.copy()
    environment["WALLACE_RELOAD_STATUS"] = str(STATUS_PATH)
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


def main() -> int:
    child = _start_child()
    try:
        for changes in watch(
            ROOT,
            debounce=800,
            step=100,
            yield_on_timeout=True,
        ):
            if child.poll() is not None:
                return child.returncode or 0
            if not any(_python_change(change) for change in changes):
                continue

            if _is_busy():
                print(
                    "Wallace is busy; source changes queued until the current "
                    "turn finishes.",
                    flush=True,
                )
                while _is_busy() and child.poll() is None:
                    time.sleep(0.2)
                if child.poll() is not None:
                    return child.returncode or 0

            print("Source change detected; reloading Wallace.", flush=True)
            _stop_child(child)
            child = _start_child()
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
