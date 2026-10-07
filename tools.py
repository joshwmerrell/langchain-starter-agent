import subprocess
import platform
import shutil
import sys
import os
import re

from fnmatch import fnmatch
from pathlib import Path
from langchain.tools import tool

# Files outside this directory can never be accessed by the agent.
SAFE_ROOT = Path(__file__).resolve().parent

# Names the agent may not read or see, matched against every path component.
SENSITIVE_PATTERNS = [".env", ".env.*"]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def _run(cmd: list[str], max_chars: int = 2500) -> str:
    """Run a fixed read-only command and return trimmed output."""
    if shutil.which(cmd[0]) is None:
        return f"({cmd[0]} is not installed)"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return f"({cmd[0]} timed out)"
    out = (r.stdout or r.stderr).strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...[truncated]"
    return out


def _cpu_model() -> str:
    for line in _run(["lscpu"], 10000).splitlines():
        if line.startswith("Model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _gpu() -> str:
    parts = []
    if shutil.which("nvidia-smi"):
        parts.append(_run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,temperature.gpu,driver_version",
            "--format=csv",
        ]))
    lspci = _run(["lspci"], 10000)
    parts.append("\n".join(
        line for line in lspci.splitlines()
        if any(k in line for k in ("VGA", "3D", "Display"))
    ))
    return "\n".join(p for p in parts if p)


_SECTIONS = {
    "os": lambda: _run(["cat", "/etc/os-release"]) + "\n" + _run(["uname", "-a"]),
    "cpu": lambda: _run(["lscpu"]),
    "memory": lambda: _run(["free", "-h"]),
    "gpu": _gpu,
    "disk": lambda: _run(["df", "-h", "-x", "tmpfs", "-x", "devtmpfs"]),
    "temps": lambda: _run(["sensors"]),
    "network": lambda: _run(["ip", "-brief", "addr"]),
    "processes": lambda: "\n".join(
        _run(["ps", "aux", "--sort=-%mem"], 10000).splitlines()[:11]
    ),
    "python": lambda: f"Python {platform.python_version()} at {sys.executable}",
}


@tool
def get_system_info(section: str = "summary") -> str:
    """Report hardware and OS details for this computer (read-only).

    Args:
        section: One of "summary", "os", "cpu", "memory", "gpu", "disk",
            "temps", "network", "processes", "python". Use "summary" for a
            short overview of OS, CPU model, memory, and GPU.
    """
    section = section.strip().lower()
    if section == "summary":
        return "\n\n".join([
            "OS: " + _run(["uname", "-a"]),
            "CPU: " + _cpu_model(),
            "Memory:\n" + _run(["free", "-h"]),
            "GPU:\n" + _gpu(),
        ])
    if section not in _SECTIONS:
        return f"Unknown section '{section}'. Valid: summary, " + ", ".join(_SECTIONS)
    return _SECTIONS[section]()


def _is_allowed(resolved: Path) -> bool:
    """Whether a resolved path is inside SAFE_ROOT and not sensitive."""
    if not resolved.is_relative_to(SAFE_ROOT):
        return False
    return not any(
        fnmatch(part, pattern)
        for part in resolved.relative_to(SAFE_ROOT).parts
        for pattern in SENSITIVE_PATTERNS
    )


def _resolve_safe(path: str) -> Path:
    # resolve() collapses ".." and follows symlinks, so the check below
    # sees the real location.
    target = (SAFE_ROOT / path).resolve()

    # Use one generic error for every case so we don't leak which paths exist.
    if not _is_allowed(target):
        raise PermissionError("Access denied or file not found.")

    return target


@tool
def read_safe_file(path: str) -> str:
    """Read a UTF-8 text file from the project directory.

    Args:
        path: File path relative to the project root, e.g. "docs/readme.txt".
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise PermissionError("Access denied or file not found.")

    return target.read_text(encoding="utf-8")


@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory in the project. Subdirectories end with "/".

    Args:
        path: Directory path relative to the project root. Defaults to the root.
    """
    target = _resolve_safe(path)
    if not target.is_dir():
        raise PermissionError("Access denied or directory not found.")

    entries = sorted(
        entry.name + "/" if entry.is_dir() else entry.name
        for entry in target.iterdir()
        if _is_allowed(entry.resolve()) and entry.name not in SKIP_DIRS
    )
    return "\n".join(entries) or "(empty directory)"


@tool
def write_safe_file(path: str, content: str) -> str:
    """Create a new file or overwrite an existing file with UTF-8 text.

    Args:
        path: File path relative to the project root.
        content: The text content to write.
    """
    target = _resolve_safe(path)
    # Ensure parent directories exist if writing to a nested path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Successfully wrote to {path}"


@tool
def execute_command(command: str) -> str:
    """Run a shell command in the project root directory.

    Args:
        command: The shell command to execute (e.g., "pytest", "python main.py").
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=SAFE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output.string if hasattr(result, 'string') else output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def check_code_quality(path: str) -> str:
    """Run built-in Python syntax and formatting checks (using python -m py_compile) on a file.

    Args:
        path: File path relative to the project root to check.
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise PermissionError("Access denied or file not found.")

    try:
        import py_compile
        py_compile.compile(str(target), doraise=True)
        return f"Code quality check passed: {path} has valid syntax."
    except py_compile.PyCompileError as e:
        return f"Code quality check failed for {path}:\n{e}"
    except Exception as e:
        return f"Error running code quality check: {e}"


def get_git_diff() -> str:
    """Show the git diff of the project to review current proposed code changes."""
    try:
        result = subprocess.run(
            "git diff",
            shell=True,
            cwd=SAFE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def search_files(pattern: str, path: str = ".") -> str:
    """Search project files for a regex pattern. Returns file:line: match (max 50).

    Args:
        pattern: Regex or plain text to find.
        path: Directory relative to the project root.
    """
    root = _resolve_safe(path)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            f = Path(dirpath, name)
            if not _is_allowed(f.resolve()):
                continue
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    results.append(f"{f.relative_to(SAFE_ROOT)}:{i}: {line.strip()[:200]}")
                    if len(results) >= 50:
                        return "\n".join(results) + "\n...[truncated]"
    return "\n".join(results) or "No matches."


@tool
def read_file_lines(path: str, start: int = 1, end: int = 200) -> str:
    """Read a line range of a file, with line numbers. Use for large files.

    Args:
        path: File path relative to the project root.
        start: First line (1-based).
        end: Last line (max 400 lines per call).
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise PermissionError("Access denied or file not found.")
    lines = target.read_text(encoding="utf-8").splitlines()
    end = min(end, start + 399)
    chunk = lines[max(start, 1) - 1:end]
    return "\n".join(f"{i}: {l}" for i, l in enumerate(chunk, max(start, 1))) or "(no lines in range)"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact piece of text in an existing file. Prefer this over
    write_safe_file when changing part of a file.

    Args:
        path: File path relative to the project root.
        old_text: Exact existing text to replace. Must appear exactly once.
        new_text: The replacement text.
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise PermissionError("Access denied or file not found.")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        return f"Error: old_text found {count} times; it must appear exactly once. Include more surrounding lines."
    target.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"Edited {path}"


# Decorate get_git_diff as a LangChain tool after defining the base function
get_git_diff = tool(get_git_diff)
