import ast
import difflib
import os
import platform
import re
import shutil
import subprocess
import sys

from fnmatch import fnmatch
from pathlib import Path

from langchain.tools import tool

# Files outside this directory can never be accessed by the agent.
SAFE_ROOT = Path(__file__).resolve().parent

# Names the agent may not read, list, search, or write, matched against every
# path component (case-insensitive).
SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*"]

# Never readable or writable (git internals can hold credentials and hooks).
HIDDEN_DIRS = {".git"}

# Hidden from trees and searches, and never writable.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

# Output limits keep tool results small, which matters with a local model's
# limited context window.
MAX_READ_CHARS = 12000
MAX_OUTPUT_CHARS = 4000
MAX_SEARCH_MATCHES = 50
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_TREE_ENTRIES = 300
COMMAND_TIMEOUT = 30
TEST_TIMEOUT = 60

ACCESS_DENIED = "Error: Access denied or file not found."


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _is_allowed(resolved: Path) -> bool:
    """Whether a resolved path is inside SAFE_ROOT and not sensitive."""
    if not resolved.is_relative_to(SAFE_ROOT):
        return False
    parts = resolved.relative_to(SAFE_ROOT).parts
    if any(part in HIDDEN_DIRS for part in parts):
        return False
    return not any(
        fnmatch(part.lower(), pattern)
        for part in parts
        for pattern in SENSITIVE_PATTERNS
    )


def _resolve(path: str) -> Path | None:
    """Resolve a project-relative path, or None if access isn't allowed.

    resolve() collapses ".." and follows symlinks, so the check sees the real
    location. Callers return one generic error for every failure so we don't
    leak which paths exist.
    """
    try:
        target = (SAFE_ROOT / path).resolve()
    except (OSError, ValueError):
        return None
    return target if _is_allowed(target) else None


def _resolve_writable(path: str) -> Path | None:
    """Like _resolve, but also refuses paths inside skipped dirs (.venv, etc.)."""
    target = _resolve(path)
    if target is None:
        return None
    if any(part in SKIP_DIRS for part in target.relative_to(SAFE_ROOT).parts):
        return None
    return target


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _truncate_middle(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and (more of) the end of long output; errors are usually last."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"


def _run_command(args: list[str] | str, *, shell: bool = False, timeout: int = COMMAND_TIMEOUT) -> str:
    """Run a command in the project root and return formatted, bounded output.

    stdin is closed so a command can never wait for (or steal) terminal input,
    which would otherwise interfere with the Textual UI.
    """
    try:
        r = subprocess.run(
            args,
            shell=shell,
            cwd=SAFE_ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except FileNotFoundError:
        name = args[0] if isinstance(args, list) else args
        return f"Error: '{name}' is not installed."
    except Exception as e:
        return f"Error executing command: {e}"

    parts = [f"Exit code: {r.returncode}"]
    if r.stdout.strip():
        parts.append(f"STDOUT:\n{r.stdout.strip()}")
    if r.stderr.strip():
        parts.append(f"STDERR:\n{r.stderr.strip()}")
    return _truncate_middle("\n".join(parts))


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

def _run(cmd: list[str], max_chars: int = 2500) -> str:
    """Run a fixed read-only command and return trimmed output."""
    if shutil.which(cmd[0]) is None:
        return f"({cmd[0]} is not installed)"
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
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
            "temps", "network", "processes", "python". "summary" gives a
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


# ---------------------------------------------------------------------------
# Exploring the project
# ---------------------------------------------------------------------------

@tool
def get_project_tree(path: str = ".", max_depth: int = 2) -> str:
    """Show the project's directory tree. Directories deeper than max_depth are listed without contents.

    Args:
        path: Directory relative to the project root. Defaults to the root.
        max_depth: How many levels to show (default 2; use 1 for a plain listing).
    """
    target = _resolve(path)
    if target is None or not target.is_dir():
        return ACCESS_DENIED

    max_depth = max(1, max_depth)
    budget = [MAX_TREE_ENTRIES]

    def build(current: Path, prefix: str, depth: int) -> list[str]:
        try:
            entries = sorted(
                (e for e in current.iterdir()
                 if e.name not in SKIP_DIRS and _is_allowed(e.resolve())),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
        except OSError:
            return []

        lines = []
        for i, entry in enumerate(entries):
            if budget[0] <= 0:
                lines.append(f"{prefix}└── ...[more entries omitted]")
                break
            budget[0] -= 1
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                if depth < max_depth:
                    lines.extend(build(entry, prefix + ("    " if is_last else "│   "), depth + 1))
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
        return lines

    lines = build(target, "", 1)
    heading = f"{target.relative_to(SAFE_ROOT).as_posix()}/"
    return heading + ("\n" + "\n".join(lines) if lines else " (empty)")


@tool
def read_safe_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a UTF-8 text file from the project. Large files are cut off; use start_line/end_line to read a range.

    Args:
        path: File path relative to the project root.
        start_line: First line to return, 1-based (default 1).
        end_line: Last line to return, inclusive; 0 means the end of the file.
    """
    target = _resolve(path)
    if target is None or not target.is_file():
        return ACCESS_DENIED
    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return "Error: file is not valid UTF-8 text."
    except OSError as error:
        return f"Error reading file: {error}"

    total = len(lines)
    if total == 0:
        return "(empty file)"
    start = max(start_line, 1)
    end = total if end_line <= 0 else min(end_line, total)
    if start > total:
        return f"Error: file has only {total} lines."

    chunk, size, last = [], 0, start - 1
    for number in range(start, end + 1):
        line = lines[number - 1]
        if size + len(line) > MAX_READ_CHARS and chunk:
            break
        chunk.append(line)
        size += len(line)
        last = number

    text = "".join(chunk)
    if start == 1 and last == total:
        return text  # whole file: return it exactly as is
    note = f"[lines {start}-{last} of {total}]\n"
    if last < end:
        text += f"\n[cut off to save space; continue with start_line={last + 1}]"
    return note + text


@tool
def search_in_files(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    ignore_case: bool = False,
) -> str:
    """Search project text files for a regex. Returns up to 50 matches as file:line: text.

    Args:
        pattern: Regular expression to look for.
        path: Directory or file to search, relative to the project root.
        file_glob: Only search files whose name matches, e.g. "*.py".
        ignore_case: Match case-insensitively.
    """
    target = _resolve(path)
    if target is None or not target.exists():
        return ACCESS_DENIED
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        return f"Error: invalid regex: {error}"

    def iter_files():
        if target.is_file():
            yield target
            return
        for root, dirs, files in os.walk(target):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                if fnmatch(name, file_glob):
                    yield Path(root) / name

    matches: list[str] = []
    for file in iter_files():
        resolved = file.resolve()
        if not _is_allowed(resolved):
            continue
        try:
            if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or binary
        rel = resolved.relative_to(SAFE_ROOT).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{rel}:{number}: {line.strip()[:200]}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    matches.append(
                        f"[stopped at {MAX_SEARCH_MATCHES} matches; narrow the pattern, path, or file_glob]"
                    )
                    return "\n".join(matches)
    return "\n".join(matches) or "No matches found."


def _collect_symbols(node: ast.AST, scope: str, out: list[str]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            out.append(f"{child.lineno}: class {scope}{child.name}")
            _collect_symbols(child, f"{scope}{child.name}.", out)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
            out.append(f"{child.lineno}: {kind} {scope}{child.name}")
        elif isinstance(child, ast.stmt):
            _collect_symbols(child, scope, out)  # defs under if/try/with blocks


@tool
def get_code_symbols(path: str) -> str:
    """List classes, functions, and methods in a Python file with their line numbers.

    Args:
        path: Python file path relative to the project root.
    """
    target = _resolve(path)
    if target is None or not target.is_file() or target.suffix != ".py":
        return f"Error: '{path}' is not an accessible Python file."
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"), filename=path)
    except SyntaxError as error:
        return f"Error: syntax error on line {error.lineno}: {error.msg}"
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading file: {error}"

    symbols: list[str] = []
    _collect_symbols(tree, "", symbols)
    return "\n".join(symbols) or "No classes or functions found."


@tool
def check_python_syntax(path: str) -> str:
    """Check a Python file for syntax errors without running it.

    Args:
        path: Python file path relative to the project root.
    """
    target = _resolve(path)
    if target is None or not target.is_file() or target.suffix != ".py":
        return f"Error: '{path}' is not an accessible Python file."
    try:
        ast.parse(target.read_text(encoding="utf-8"), filename=path)
    except SyntaxError as error:
        return f"Syntax error in {path}, line {error.lineno}: {error.msg}"
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading file: {error}"
    return f"OK: {path} has valid Python syntax."


# ---------------------------------------------------------------------------
# Changing files (these require the user's approval; see TOOLS_REQUIRING_APPROVAL)
# ---------------------------------------------------------------------------

@tool
def write_safe_file(path: str, content: str) -> str:
    """Create a short new file or fully rewrite an existing file.

    For small changes to an existing file, use replace_in_file instead. Avoid
    sending large file contents through this tool unless a complete rewrite is
    explicitly required.

    Args:
        path: File path relative to the project root.
        content: The complete UTF-8 text to write.
    """
    target = _resolve_writable(path)
    if target is None:
        return "Error: Access denied."
    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"{'Overwrote' if existed else 'Created'} {path} ({len(content)} characters)."


@tool
def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing one exact block of text.

    Prefer this tool for small edits to existing files. old_text must match
    exactly once, so include enough surrounding lines to be unique.

    Args:
        path: File path relative to the project root.
        old_text: The exact existing text to replace (copy it from read_safe_file).
        new_text: The text to put in its place.
    """
    target = _resolve_writable(path)
    if target is None or not target.is_file():
        return ACCESS_DENIED
    if not old_text:
        return "Error: old_text must not be empty."
    if old_text == new_text:
        return "Error: old_text and new_text are identical."
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading {path}: {error}"

    count = text.count(old_text)
    if count == 0:
        return "Error: old_text was not found. Re-read the file and copy the text exactly."
    if count > 1:
        return f"Error: old_text matches {count} places. Include more surrounding lines to make it unique."
    try:
        target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Edited {path}."


def preview_write(path: str, content: str) -> str:
    """Readable preview of what write_safe_file would change (used by approval prompts)."""
    target = _resolve_writable(path)
    if target is None:
        return f"Path: {path}\n(this path is not writable)"
    if not target.is_file():
        added = "\n".join("+ " + line for line in content.splitlines())
        return f"Path: {path}\n(new file)\n\n{added}"
    try:
        old = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"Path: {path}\n(existing file could not be read)\n\n{content}"
    diff = "\n".join(difflib.unified_diff(
        old.splitlines(), content.splitlines(), "current", "proposed", lineterm="", n=2
    ))
    return f"Path: {path}\n\n{diff or '(no changes)'}"


# ---------------------------------------------------------------------------
# Running things
# ---------------------------------------------------------------------------

@tool
def execute_command(command: str) -> str:
    """Run a shell command in the project root (30 second limit, no interactive input). Output is shortened if long.

    Args:
        command: The shell command, e.g. "python main.py" or "pip list".
    """
    return _run_command(command, shell=True)


@tool
def run_tests(test_path: str = ".") -> str:
    """Run pytest (quiet mode, short tracebacks) on the project or one test file/directory.

    Args:
        test_path: Test file or directory relative to the project root. Defaults to the whole project.
    """
    target = _resolve(test_path)
    if target is None or not target.exists():
        return ACCESS_DENIED
    rel = target.relative_to(SAFE_ROOT).as_posix()
    return _run_command(
        [sys.executable, "-m", "pytest", rel, "-q", "--tb=short"],
        timeout=TEST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Git (read-only)
# ---------------------------------------------------------------------------

@tool
def get_git_status() -> str:
    """Show the current branch and which files are modified, staged, or untracked."""
    return _run_command(["git", "status", "--short", "--branch"])


@tool
def get_git_diff(staged: bool = False) -> str:
    """Show uncommitted changes as a git diff. Secret files are excluded.

    Args:
        staged: Show staged changes instead of unstaged ones.
    """
    args = ["git", "diff"]
    if staged:
        args.append("--staged")
    # Keep sensitive files out of the diff, matching the read restrictions.
    excludes = [f":(glob,exclude)**/{pattern}" for pattern in SENSITIVE_PATTERNS]
    args += ["--", ".", *excludes]
    result = _run_command(args)
    return "(no changes)" if result == "Exit code: 0" else result


# ---------------------------------------------------------------------------
# Registry: main.py builds the agent from these, so adding a tool only needs
# an entry here.
# ---------------------------------------------------------------------------

TOOLS = [
    get_project_tree,
    read_safe_file,
    search_in_files,
    get_code_symbols,
    check_python_syntax,
    write_safe_file,
    replace_in_file,
    execute_command,
    run_tests,
    get_git_status,
    get_git_diff,
    get_system_info,
]

# The agent pauses for the user's approval before these run.
TOOLS_REQUIRING_APPROVAL = {"write_safe_file", "replace_in_file", "execute_command"}
