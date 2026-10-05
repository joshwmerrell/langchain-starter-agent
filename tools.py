import subprocess

from fnmatch import fnmatch
from pathlib import Path

# Files outside this directory can never be accessed by the agent.
SAFE_ROOT = Path(__file__).resolve().parent

# Names the agent may not read or see, matched against every path component.
SENSITIVE_PATTERNS = [".env", ".env.*"]


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


def read_safe_file(path: str) -> str:
    """Read a UTF-8 text file from the project directory.

    Args:
        path: File path relative to the project root, e.g. "docs/readme.txt".
    """
    target = _resolve_safe(path)
    if not target.is_file():
        raise PermissionError("Access denied or file not found.")

    return target.read_text(encoding="utf-8")


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
        if _is_allowed(entry.resolve())
    )
    return "\n".join(entries) or "(empty directory)"


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
