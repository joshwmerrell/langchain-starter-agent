import ast
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
        # return output.string if hasattr(result, 'string') else output.strip()
        # The following line is a fix by Google Gemini's suggestion.
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def run_tests(test_path: str = ".") -> str:
    """Run tests using pytest in the project root directory.

    Args:
        test_path: The directory or file to run tests on. Defaults to the project root.
    """
    try:
        result = subprocess.run(
            ["pytest", test_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Tests timed out after 60 seconds."
    except Exception as e:
        return f"Error running tests: {e}"


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
def get_code_symbols(path: str) -> str:
    """Extract all class and function definitions from a Python file.

    Args:
        path: File path relative to the project root.
    """
    target = _resolve_safe(path)
    if not target.is_file() or target.suffix != ".py":
        return f"Error: '{path}' is not a valid Python file."

    try:
        with open(target, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(f"Class: {node.name}")
            elif isinstance(node, ast.FunctionDef):
                # Check if it's a method inside a class
                parent = None
                for p in ast.walk(tree):
                    if isinstance(p, ast.ClassDef) and node in p.body:
                        parent = p.name
                        break
                prefix = f"{parent}." if parent else ""
                symbols.append(f"Function: {prefix}{node.name}")

        if not symbols:
            return "No classes or functions found."

        return "\n".join(sorted(set(symbols)))
    except Exception as e:
        return f"Error parsing file: {e}"


@tool
def get_project_tree(path: str = ".") -> str:
    """Return a visual tree structure of the project directory.

    Args:
        path: Directory path relative to the project root. Defaults to the root.
    """
    target = _resolve_safe(path)
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    def _build_tree(current_dir: Path, prefix: str = "") -> list[str]:
        lines = []
        # Get sorted entries, excluding skip dirs
        entries = sorted(
            [e for e in current_dir.iterdir() if _is_allowed(e.resolve()) and e.name not in SKIP_DIRS],
            key=lambda e: (e.is_file(), e.name.lower())
        )
        
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                lines.extend(_build_tree(entry, prefix + ("    " if is_last else "│   ")))
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
        return lines

    tree_lines = _build_tree(target)
    return f"{target.relative_to(SAFE_ROOT)}/\n" + "\n".join(tree_lines) if tree_lines else f"{target.relative_to(SAFE_ROOT)}/ (empty)"


# Decorate get_git_diff as a LangChain tool after defining the base function
get_git_diff = tool(get_git_diff)
