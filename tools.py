import ast
import difflib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib

from contextvars import ContextVar, Token
from fnmatch import fnmatch
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import StructuredTool

from dependency_analysis import DependencyGraph, analyze_change_impact
from wallace_mcp import all_mcp_servers, connection_status
from wallace_runtime import APP_ROOT, DEV_MODE, WORKSPACE_ROOT
from wallace_status import current_status_text

# Model-facing file tools default to the ordinary work area. Only the host may
# select the source root for the permanent development thread.
SAFE_ROOT = WORKSPACE_ROOT
_ACTIVE_ROOT: ContextVar[Path | None] = ContextVar("wallace_active_workspace", default=None)


def current_workspace_root() -> Path:
    return _ACTIVE_ROOT.get() or SAFE_ROOT


def set_active_workspace(root: Path) -> Token:
    resolved = root.resolve()
    if resolved == APP_ROOT.resolve() and not DEV_MODE:
        raise ValueError("source workspace is unavailable outside development mode")
    return _ACTIVE_ROOT.set(resolved)


def reset_active_workspace(token: Token) -> None:
    _ACTIVE_ROOT.reset(token)

# Names the agent may not read, list, search, or write, matched against every
# path component (case-insensitive).
SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*"]

# Never readable or writable through model-facing file tools.
PRIVATE_DIRS = {".wallace", ".aws", ".ssh", ".gnupg", ".config", ".codex", ".agents"}
HIDDEN_DIRS = PRIVATE_DIRS | {".git"}

# Hidden from trees and searches, and never writable.
SKIP_DIRS = HIDDEN_DIRS | {"node_modules", "__pycache__", ".venv", "venv"}

# Output limits keep tool results small, which matters with a local model's
# limited context window.
MAX_READ_CHARS = 12000
MAX_OUTPUT_CHARS = 4000
MAX_SEARCH_MATCHES = 50
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_TREE_ENTRIES = 300
MAX_WRITE_CHARS = 50_000
MAX_REPLACEMENT_CHARS = 25_000
MAX_LINE_EDIT_LINES = 80
COMMAND_TIMEOUT = 30
SOURCE_CHECK_TIMEOUT = 60
TEST_TIMEOUT = 180

ACCESS_DENIED = (
    "Error: Access denied or file not found. Use a project-root-relative path "
    "as shown by get_project_tree (for example, notes.txt)."
)

GIT_UNAVAILABLE = (
    "Git metadata is not available in the selected workspace. This is normal "
    "for an ordinary Wallace workspace; continue with file and directory tools."
)


def _tool_environment() -> dict[str, str]:
    """Do not pass common credential variables to model-invoked subprocesses.

    The command sandbox also hides local private paths, but the subprocess
    still runs under the user's OS identity; it is not a perfect isolation
    boundary against local services.
    """
    blocked = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")
    environment = {
        name: value for name, value in os.environ.items()
        if not name.upper().endswith(blocked)
        and not name.upper().startswith("AWS_")
        and name.upper() not in {
            "SSH_AUTH_SOCK", "GPG_AGENT_INFO", "DBUS_SESSION_BUS_ADDRESS",
            "XDG_RUNTIME_DIR", "PYTHONPATH", "WALLACE_DEV_MODE",
        }
    }
    environment["HOME"] = "/tmp"
    environment["PATH"] = str(APP_ROOT / ".venv" / "bin") + os.pathsep + "/usr/bin:/bin"
    return environment


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _is_allowed(resolved: Path) -> bool:
    """Whether a resolved path is inside SAFE_ROOT and not sensitive."""
    root = current_workspace_root()
    if not resolved.is_relative_to(root):
        return False
    parts = resolved.relative_to(root).parts
    if any(part.lower() in SKIP_DIRS for part in parts):
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
        target = (current_workspace_root() / path).resolve()
    except (OSError, ValueError):
        return None
    return target if _is_allowed(target) else None


def _resolve_writable(path: str) -> Path | None:
    """Resolve a path that may be written by an approved tool."""
    target = _resolve(path)
    return target


def _atomic_write_text(target: Path, content: str) -> None:
    """Write text beside the target and replace it atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def _sandbox_command(
    args: list[str] | str,
    *,
    shell: bool,
    cwd: Path,
    expose_git_config: bool = False,
    sandbox_root: Path | None = None,
) -> list[str]:
    """Mount only the active work area for model-invoked processes.

    A missing Bubblewrap fails closed instead of running on the host.
    """
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise RuntimeError("Bubblewrap (bwrap) is required for command and test tools")
    root = (sandbox_root or current_workspace_root()).resolve()
    working = cwd.resolve()
    if not working.is_relative_to(root):
        raise RuntimeError("command working directory is outside the active workspace")
    command = [bubblewrap, "--die-with-parent", "--unshare-pid"]
    for directory in ("/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"):
        if Path(directory).exists():
            command.extend(["--ro-bind", directory, directory])
    command.extend(["--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"])
    command.extend(["--bind", str(root), str(root)])
    virtualenv = APP_ROOT / ".venv"
    if root != APP_ROOT.resolve() and virtualenv.is_dir():
        command.extend(["--ro-bind", str(virtualenv), str(virtualenv)])
    interpreter = Path(sys.executable).resolve()
    if not interpreter.is_relative_to(Path("/usr")) and not interpreter.is_relative_to(Path("/bin")):
        distribution = interpreter.parent.parent
        if distribution.is_dir() and not distribution.is_relative_to(root):
            command.extend(["--ro-bind", str(distribution), str(distribution)])
            launcher = Path(sys.executable)
            if launcher.is_symlink():
                link = launcher.readlink()
                link = link if link.is_absolute() else launcher.parent / link
                alias_distribution = link.parent.parent
                if alias_distribution != distribution and not alias_distribution.is_relative_to(root):
                    # uv's venv Python can point through an unversioned alias
                    # before reaching the versioned installation.
                    command.extend(["--ro-bind", str(distribution), str(alias_distribution)])
    # File-tool denials must hold for shell/test tools too, including nested
    # .env and key files in a user workspace. Do not traverse virtualenvs,
    # package trees or git internals for every command.
    for current, directories, files in os.walk(root, followlinks=False):
        kept = []
        for name in directories:
            path = Path(current) / name
            if path.is_symlink():
                continue
            if name.lower() in PRIVATE_DIRS or any(
                fnmatch(name.lower(), pattern) for pattern in SENSITIVE_PATTERNS
            ):
                command.extend(["--tmpfs", str(path)])
            elif name.lower() == ".git":
                config = path / "config"
                if config.is_file() and not config.is_symlink():
                    # Ordinary command/test tools must not inspect Git's
                    # private metadata. The dedicated read-only Git tools
                    # need the repository-local config, however: masking it
                    # makes Git fail before it can report status or a diff.
                    source = config if expose_git_config else Path("/dev/null")
                    command.extend(["--ro-bind", str(source), str(config)])
            elif name.lower() not in {".venv", "venv", "node_modules", "__pycache__"}:
                kept.append(name)
        directories[:] = kept
        for name in files:
            path = Path(current) / name
            if path.is_symlink():
                continue
            if any(fnmatch(name.lower(), pattern) for pattern in SENSITIVE_PATTERNS):
                command.extend(["--ro-bind", "/dev/null", str(path)])
    command.extend(["--chdir", str(working), "--"])
    if shell:
        command.extend(["/bin/sh", "-c", str(args)])
    else:
        command.extend(str(arg) for arg in args)
    return command


def _run_command(
    args: list[str] | str, *, shell: bool = False,
    timeout: int = COMMAND_TIMEOUT, cwd: Path | None = None,
    input_text: str | None = None, expose_git_config: bool = False,
    sandbox_root: Path | None = None,
) -> str:
    """Run a command in the project root and return formatted, bounded output.

    stdin is closed unless the host supplies source text to a fixed checker,
    so commands cannot wait for (or steal) input from the Textual UI.
    """
    try:
        working = cwd or current_workspace_root()
        sandboxed = _sandbox_command(
            args, shell=shell, cwd=working, expose_git_config=expose_git_config,
            sandbox_root=sandbox_root,
        )
        r = subprocess.run(
            sandboxed,
            shell=False,
            cwd=working,
            env=_tool_environment(),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL if input_text is None else None,
            input=input_text,
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


def _git_available(root: Path | None = None) -> bool:
    """Return whether the selected workspace contains Git metadata."""
    marker = (root or current_workspace_root()) / ".git"
    return marker.is_dir() or marker.is_file()


def _git_root() -> Path | None:
    """Return the repository root used by read-only Git tools.

    Git inspection is intentionally repository-wide and read-only. This lets
    an ordinary thread inspect changes in the source tree and in workspace/
    without granting its file/edit tools access to source files.
    """
    for candidate in (APP_ROOT.resolve(), current_workspace_root().resolve()):
        if _git_available(candidate):
            return candidate
    return None


def _git_pathspec_excludes() -> list[str]:
    excludes = [f":(glob,exclude)**/{pattern}" for pattern in SENSITIVE_PATTERNS]
    excludes.extend(f":(glob,exclude)**/{directory}/**" for directory in PRIVATE_DIRS)
    return excludes


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
            env=_tool_environment(),
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
    # Command-line arguments may contain secrets; report only process names.
    "processes": lambda: "\n".join(
        _run(["ps", "-eo", "pid,comm", "--sort=-rss"], 10000).splitlines()[:11]
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
    """Show the selected workspace's directory tree.

    The selected workspace is the authoritative scope for the current thread;
    it may be the ordinary ``workspace/`` directory rather than Wallace's
    source repository. Directories deeper than max_depth are listed without
    contents.

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
                 if e.name.lower() not in SKIP_DIRS and _is_allowed(e.resolve())),
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
    heading = f"{target.relative_to(current_workspace_root()).as_posix()}/"
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
            dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS)
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
        rel = resolved.relative_to(current_workspace_root()).as_posix()
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
    if target is not None and not target.is_file() and not Path(path).suffix:
        # Local models often omit the .py suffix after seeing a test module
        # name. Accept the unambiguous sibling without weakening path rules.
        sibling = _resolve(path + ".py")
        if sibling is not None and sibling.is_file():
            target = sibling
            path = path + ".py"
    if target is None or not target.is_file() or target.suffix != ".py":
        return f"Error: '{path}' is not an accessible Python file."
    try:
        compile(target.read_text(encoding="utf-8"), path, "exec", dont_inherit=True)
    except SyntaxError as error:
        return f"Error: Python syntax error in {path}, line {error.lineno}: {error.msg}"
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading file: {error}"
    return f"OK: {path} has valid Python syntax."


@tool
def check_source_file(path: str) -> str:
    """Check a source file with an installed, non-networked parser/compiler.

    TypeScript uses Deno for Deno projects or standalone files, and tsc for
    Node projects. HTML/CSS get a basic parse (not a browser behavior test).
    External checkers must already be installed; this tool never installs packages.

    Args:
        path: Source file relative to the project root.
    """
    target = _resolve(path)
    if target is None or not target.is_file():
        return ACCESS_DENIED
    suffix = target.suffix.lower()
    if suffix == ".py":
        return check_python_syntax.invoke({"path": path})
    if suffix in {".json", ".toml"}:
        try:
            source = target.read_text(encoding="utf-8")
            json.loads(source) if suffix == ".json" else tomllib.loads(source)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            return f"Error: invalid {suffix[1:]} in {path}: {error}"
        return f"OK: {path} has valid {suffix[1:]} syntax."
    if suffix in {".html", ".css"}:
        if shutil.which("deno") is None:
            return f"Error: deno is not installed; cannot parse {path}."
        try:
            source = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return f"Error reading {path}: {error}"
        result = _run_command(
            ["deno", "fmt", "--no-config", "--no-editorconfig", "--ext", suffix[1:], "-"],
            timeout=SOURCE_CHECK_TIMEOUT, input_text=source,
        )
        if result.startswith("Exit code: 0"):
            return (
                f"OK: basic {suffix[1:]} parse completed for {path}; "
                "CSS property validity and cross-file references were not checked; "
                "browser behavior was not checked."
            )
        return f"{result}\nChecker: deno fmt for {path}"
    if suffix in {".ts", ".tsx"}:
        node_project = any(
            (parent / name).is_file()
            for parent in (target.parent, *target.parents)
            if parent == current_workspace_root() or parent.is_relative_to(current_workspace_root())
            for name in ("tsconfig.json", "package.json")
        )
        deno_project = any(
            (parent / name).is_file()
            for parent in (target.parent, *target.parents)
            if parent == current_workspace_root() or parent.is_relative_to(current_workspace_root())
            for name in ("deno.json", "deno.jsonc")
        )
        if deno_project or (not node_project and shutil.which("deno") is not None):
            command = [
                "deno", "check", "--cached-only", "--no-lock",
                "--node-modules-dir=none", str(target),
            ]
        else:
            command = ["tsc", "--noEmit", "--pretty", "false", "--skipLibCheck"]
            if suffix == ".tsx":
                command.extend(["--jsx", "react-jsx"])
            command.append(str(target))
        if shutil.which(command[0]) is None:
            return f"Error: {command[0]} is not installed; cannot check {path}."
        result = _run_command(command, timeout=SOURCE_CHECK_TIMEOUT)
        return f"{result}\nChecker: {command[0]} for {path}"
    commands = {
        ".js": ["node", "--check", str(target)],
        ".mjs": ["node", "--check", str(target)],
        ".cjs": ["node", "--check", str(target)],
        ".sh": ["bash", "-n", str(target)],
        ".go": ["gofmt", "-e", "-d", str(target)],
        ".rs": ["rustc", "--edition=2021", "--crate-type=lib", "--emit=metadata", "-o", os.devnull, str(target)],
    }
    command = commands.get(suffix)
    if command is None:
        return f"Error: unsupported source type {suffix or '(none)'}."
    if shutil.which(command[0]) is None:
        return f"Error: {command[0]} is not installed; cannot check {path}."
    result = _run_command(command, timeout=SOURCE_CHECK_TIMEOUT)
    return f"{result}\nChecker: {command[0]} for {path}"


@tool
def summarize_project_file(path: str) -> str:
    """Summarize a Python project file's public API and dependency relationships.

    Args:
        path: Python file path relative to the project root.
    """
    target = _resolve(path)
    if target is None or not target.is_file() or target.suffix != ".py":
        return f"Error: '{path}' is not an accessible Python file."
    graph = DependencyGraph(current_workspace_root()).build()
    return json.dumps(graph.summarize_module(target), indent=2)


@tool
def analyze_project_dependencies() -> str:
    """Build a project-wide Python import graph and report module relationships."""
    graph = DependencyGraph(current_workspace_root()).build()
    lines = [f"Analyzed {len(graph.module_paths)} local Python modules."]
    for module in sorted(graph.module_paths):
        imports = ", ".join(sorted(graph.modules[module])) or "none"
        external = ", ".join(sorted(graph.external_imports.get(module, set()))) or "none"
        dependents = ", ".join(graph.get_dependents(module)) or "none"
        lines.append(
            f"{module}: imports [{imports}]; external [{external}]; "
            f"dependents [{dependents}]"
        )
    if graph.declared_dependencies:
        lines.append(
            "Declared packages: " + ", ".join(sorted(graph.declared_dependencies))
        )
    return "\n".join(lines)


@tool
def analyze_git_change_impact(staged: bool = False) -> str:
    """Analyze changed Python files and report affected local modules.

    Non-Python changes are intentionally ignored because they cannot be mapped
    to the Python import graph. Untracked Python files are included for the
    unstaged view.
    """
    root = _git_root()
    if root is None:
        return GIT_UNAVAILABLE
    command = ["git", "diff", "--name-status"]
    if staged:
        command.append("--cached")
    try:
        result = subprocess.run(
            _sandbox_command(
                command, shell=False, cwd=root, expose_git_config=True, sandbox_root=root,
            ),
            cwd=root, env=_tool_environment(), capture_output=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT,
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        return f"Error: Git change-impact analysis could not run: {error}"
    if result.returncode != 0:
        return f"Error: git diff failed ({result.stderr.strip()})."

    changes = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status, path = fields[0], fields[-1]
        if not path.endswith(".py") or not _is_allowed((root / path).resolve()):
            continue
        hunk = (
            {"added": True}
            if status.startswith("A")
            else {"deleted": True}
            if status.startswith("D")
            else {"modified": True}
        )
        changes.append({"old_file": path, "new_file": path, "hunks": [hunk]})

    if not staged:
        try:
            untracked = subprocess.run(
                _sandbox_command(
                    ["git", "ls-files", "--others", "--exclude-standard"],
                    shell=False, cwd=root, expose_git_config=True, sandbox_root=root,
                ),
                cwd=root, env=_tool_environment(), capture_output=True,
                encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                timeout=COMMAND_TIMEOUT,
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            untracked = None
        if untracked is not None and untracked.returncode == 0:
            for path in untracked.stdout.splitlines():
                if path.endswith(".py") and _is_allowed((root / path).resolve()):
                    changes.append({
                        "old_file": None,
                        "new_file": path,
                        "hunks": [{"added": True}],
                    })

    if not changes:
        return "No changed Python files found."
    graph = DependencyGraph(root).build()
    impacts = analyze_change_impact({"diff": changes}, graph)
    return json.dumps(impacts, indent=2) if impacts else "No Git changes found."


# ---------------------------------------------------------------------------
# Changing files (these require the user's approval; see TOOLS_REQUIRING_APPROVAL)
# ---------------------------------------------------------------------------

def _python_write_error(target: Path, content: str) -> str | None:
    """Reject invalid Python before replacing the file on disk."""
    if target.suffix.lower() != ".py":
        return None
    relative = target.relative_to(current_workspace_root()).as_posix()
    try:
        compile(content, relative, "exec", dont_inherit=True)
    except SyntaxError as error:
        return (
            f"Error: Python syntax error in {relative}, line {error.lineno}: "
            f"{error.msg}. File was not changed."
        )
    return None


@tool
def write_safe_file(path: str, content: str) -> str:
    """Write a short new file; avoid using this for existing files.

    For changes to an existing file, use replace_in_file in small blocks.
    A whole-file overwrite sends much more JSON and should only be used when
    the user explicitly requests a complete rewrite.

    Args:
        path: File path relative to the project root.
        content: The complete UTF-8 text to write.
    """
    target = _resolve_writable(path)
    if target is None:
        return "Error: Access denied."
    if len(content) > MAX_WRITE_CHARS:
        return (
            f"Error: content is {len(content)} characters; the maximum is "
            f"{MAX_WRITE_CHARS}. Use replace_in_file for a small edit."
        )
    syntax_error = _python_write_error(target, content)
    if syntax_error is not None:
        return syntax_error
    existed = target.exists()
    try:
        _atomic_write_text(target, content)
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
    if len(new_text) > MAX_REPLACEMENT_CHARS:
        return (
            f"Error: replacement is {len(new_text)} characters; the maximum is "
            f"{MAX_REPLACEMENT_CHARS}. Use a short edit."
        )
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading {path}: {error}"

    count = text.count(old_text)
    if count == 0:
        # A ranged read includes a display-only line-number header. Models
        # sometimes copy that header together with the real anchor.
        without_header = re.sub(r"^\[lines \d+-\d+ of \d+\]\n", "", old_text, count=1)
        if without_header != old_text:
            old_text = without_header
            count = text.count(old_text)
    if count == 0:
        return (
            "Error: old_text was not found. Re-read the file and copy the text "
            "exactly, or use edit_file_lines with current line numbers."
        )
    if count > 1:
        return f"Error: old_text matches {count} places. Include more surrounding lines to make it unique."
    updated = text.replace(old_text, new_text, 1)
    syntax_error = _python_write_error(target, updated)
    if syntax_error is not None:
        return syntax_error
    try:
        _atomic_write_text(target, updated)
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Edited {path}."


@tool
def edit_file_lines(path: str, start_line: int, end_line: int, new_text: str) -> str:
    """Replace a small 1-based inclusive line range in an existing file.

    Use this when replace_in_file cannot match an exact text block. First read
    the current file with read_safe_file to obtain line numbers. To insert
    before start_line, set end_line to start_line - 1. Never use stale line
    numbers after another edit; read the file again. Python syntax is checked
    before any change is written.

    Args:
        path: File path relative to the project root.
        start_line: First line to replace, starting at 1.
        end_line: Last line to replace, inclusive; start_line - 1 inserts.
        new_text: Replacement text; keep the edit short.
    """
    target = _resolve_writable(path)
    if target is None or not target.is_file():
        return ACCESS_DENIED
    if len(new_text) > MAX_REPLACEMENT_CHARS:
        return (
            f"Error: replacement is {len(new_text)} characters; the maximum is "
            f"{MAX_REPLACEMENT_CHARS}. Use a short edit."
        )
    try:
        original = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return f"Error reading {path}: {error}"
    lines = original.splitlines(keepends=True)
    if not (1 <= start_line <= len(lines) + 1 and start_line - 1 <= end_line <= len(lines)):
        return f"Error: line range is invalid for a file with {len(lines)} lines. Re-read the file."
    if end_line - start_line + 1 > MAX_LINE_EDIT_LINES:
        return f"Error: edit spans more than {MAX_LINE_EDIT_LINES} lines. Use smaller edits."
    before = "".join(lines[:start_line - 1])
    after = "".join(lines[end_line:])
    replacement = new_text
    if end_line == start_line - 1 and before and not before.endswith("\n") and replacement:
        replacement = "\n" + replacement
    # A missing terminal newline must not accidentally join the next line.
    if replacement and not replacement.endswith("\n") and (after or (
        end_line >= start_line and lines[end_line - 1].endswith("\n")
    )):
        replacement += "\n"
    updated = before + replacement + after
    if updated == original:
        return "Error: edit makes no change. Re-read the file."
    syntax_error = _python_write_error(target, updated)
    if syntax_error is not None:
        return syntax_error
    try:
        _atomic_write_text(target, updated)
    except OSError as error:
        return f"Error writing {path}: {error}"
    return f"Edited {path} lines {start_line}-{end_line}."


def preview_line_edit(path: str, start_line: int, end_line: int, new_text: str) -> str:
    """Show the actual current lines that a pending line edit would replace."""
    target = _resolve_writable(path)
    if target is None or not target.is_file():
        return f"Path: {path}\n(this path is not writable)"
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return f"Path: {path}\n(existing file could not be read)"
    if not (1 <= start_line <= len(lines) + 1 and start_line - 1 <= end_line <= len(lines)):
        return f"Path: {path}\n(invalid line range; file has {len(lines)} lines)"
    removed = [f"- {line}" for line in lines[start_line - 1:end_line]]
    added = [f"+ {line}" for line in new_text.splitlines()]
    return f"Path: {path}\nLines: {start_line}-{end_line}\n\n" + "\n".join(removed + added)


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
def list_mcp_connections() -> str:
    """List MCP server URLs and live connection states, without changing them.

    MCP management is user-only through the TUI's /mcp commands. Do not edit
    the private MCP registry with shell commands.
    """
    try:
        servers = all_mcp_servers(APP_ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return f"Error: MCP configuration could not be read: {error}"
    return json.dumps({"servers": [
        {
            "name": server.name,
            "url": server.url,
            "active": server.enabled,
            "connection": connection_status(server),
            "built_in": server.builtin,
            "keywords": server.keywords,
        }
        for server in servers
    ]})

@tool
def execute_command(command: str) -> str:
    """Run a shell command in the project root (30 second limit, no interactive input). Output is shortened if long.

    Args:
        command: The shell command, e.g. "python main.py" or "pip list".
    """
    # `rm -rf` succeeds even when its target has already gone. Avoid reporting
    # repeated no-op removals as completed work, especially on resumed plans.
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = []
    if parts and parts[0] in {"rm", "rmdir", "unlink"} and not re.search(
        r"[;&|`$\n\r<>]", command
    ):
        paths = [part for part in parts[1:] if not part.startswith("-")]
        if len(paths) == 1 and not Path(paths[0]).is_absolute():
            target = _resolve(paths[0])
            lexical = current_workspace_root() / paths[0]
            if target is not None and not target.exists() and not lexical.is_symlink():
                return (
                    f"Error: {paths[0]} is already absent; removal was not run. "
                    "Do not repeat removal. Verify absence, then call "
                    "run_tests(test_path='.') for the remaining checks."
                )
    return _run_command(command, shell=True)


@tool
def run_tests(test_path: str = ".") -> str:
    """Run the project's tests on a file or directory.

    Uses pytest when installed and falls back to unittest discovery, so the
    tool remains useful in a fresh environment.

    Args:
        test_path: Test file or directory relative to the project root. Defaults to the whole project.
    """
    target = _resolve(test_path)
    if target is None or not target.exists():
        return ACCESS_DENIED
    root = current_workspace_root()
    rel = target.relative_to(root).as_posix()
    if _module_available("pytest"):
        command = [sys.executable, "-m", "pytest", rel, "-q", "--tb=short"]
    else:
        discover_root = rel if target.is_dir() else str(Path(rel).parent)
        if rel == "." and (root / "tests").is_dir():
            # unittest does not recurse into a tests/ directory without an
            # __init__.py, unlike pytest. Discover from there explicitly.
            discover_root = "tests"
        command = [sys.executable, "-m", "unittest", "discover", "-s", discover_root]
        if target.is_file():
            command.extend(["-p", target.name])
    return _run_command(command, timeout=TEST_TIMEOUT)


@tool
def run_language_tests(language: str, path: str = ".") -> str:
    """Run a project's Python, JavaScript/TypeScript, Go, or Rust tests.

    Uses pytest/unittest, Deno or npm test, go test, or cargo test.
    Standalone Deno test files also work without a manifest. The relevant
    runner must already be installed. Tests may execute project code, so
    approval is required.

    Args:
        language: One of python, javascript, typescript, go, rust.
        path: Project or test directory relative to the Wallace workspace.
    """
    target = _resolve(path)
    if target is None or not target.exists():
        return ACCESS_DENIED
    language = language.strip().lower()
    if language == "python":
        return run_tests.invoke({"test_path": path})
    if not target.is_dir():
        return "Error: non-Python test path must be a project directory."
    if language in {"javascript", "typescript"}:
        deno_manifest = next(
            (target / name for name in ("deno.json", "deno.jsonc") if (target / name).is_file()),
            None,
        )
        if deno_manifest is not None:
            manifest = deno_manifest
            runner = "deno"
            command = ["deno", "test", "--cached-only", "--no-lock", "--node-modules-dir=none"]
        elif (target / "package.json").is_file():
            manifest = target / "package.json"
            runner = "npm"
            command = [runner, "test"]
        elif any(next(target.glob(pattern), None) is not None for pattern in (
            "*_test.ts", "*.test.ts", "*_test.tsx", "*.test.tsx",
            "*_test.js", "*.test.js", "*_test.mjs", "*.test.mjs",
        )):
            manifest = None
            runner = "deno"
            command = [
                "deno", "test", "--cached-only", "--no-config", "--no-lock",
                "--node-modules-dir=none",
            ]
        else:
            return f"Error: no test manifest or standalone Deno tests found in {path}."
    elif language == "go":
        manifest = target / "go.mod"
        runner = "go"
        command = [runner, "test", "./..."]
    elif language == "rust":
        manifest = target / "Cargo.toml"
        runner = "cargo"
        command = [runner, "test", "--manifest-path", str(manifest)]
    else:
        return f"Error: unsupported test language {language!r}."
    if manifest is not None and (not manifest.is_file() or not _is_allowed(manifest.resolve())):
        return f"Error: {manifest.name} is missing from {path}."
    if shutil.which(runner) is None:
        return f"Error: {runner} is not installed; cannot test {path}."
    return _run_command(command, timeout=TEST_TIMEOUT, cwd=target)


@tool
def verify_tool_contract(tool_name: str, test_path: str) -> str:
    """Verify a tool definition and its passing contract-test file.

    In the source thread, verify Wallace's built-in registry. In an ordinary
    workspace, verify a local @tool definition without exposing Wallace source.
    The test must call the tool's invoke method; unrelated tests are not enough.

    Args:
        tool_name: Exact name of a registered built-in or a workspace @tool.
        test_path: Python test file relative to the workspace root.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", tool_name):
        return "Error: invalid tool name."
    target = _resolve(test_path)
    if target is None or not target.is_file() or target.suffix != ".py":
        return ACCESS_DENIED
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"), filename=test_path)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        return f"Error: could not inspect test file: {error}"
    exercised = any(
        isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and function.name.startswith("test_")
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "invoke"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == tool_name
            for node in ast.walk(function)
        )
        for function in ast.walk(tree)
    )
    if not exercised:
        return f"Error: a test_ function in {test_path} must call {tool_name}.invoke(...)."
    source_thread = current_workspace_root().resolve() == APP_ROOT.resolve()
    if source_thread:
        registry = _run_command([
            sys.executable, "-c",
            "import sys; from tools import TOOLS; sys.exit(0 if sys.argv[1] in {tool.name for tool in TOOLS} else 1)",
            tool_name,
        ])
        if not registry.startswith("Exit code: 0"):
            return f"Error: {tool_name} is not registered in tools.TOOLS, or the tools module cannot be imported. {registry}"
    else:
        defined = False
        for directory, directories, files in os.walk(current_workspace_root()):
            directories[:] = [name for name in directories if name.lower() not in SKIP_DIRS]
            for name in files:
                if not name.endswith(".py"):
                    continue
                source = Path(directory) / name
                try:
                    resolved = source.resolve()
                except (OSError, RuntimeError):
                    continue
                if not _is_allowed(resolved):
                    continue
                try:
                    if source.stat().st_size > MAX_SEARCH_FILE_BYTES:
                        continue
                    module = ast.parse(source.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, SyntaxError):
                    continue
                for node in ast.walk(module):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != tool_name:
                        continue
                    if any(
                        isinstance(decorator, ast.Name) and decorator.id == "tool"
                        or isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "tool"
                        for decorator in node.decorator_list
                    ):
                        defined = True
                        break
                if defined:
                    break
            if defined:
                break
        if not defined:
            return f"Error: {tool_name} is not defined with @tool in this workspace."
    result = run_tests.invoke({"test_path": test_path})
    if not result.startswith("Exit code: 0"):
        return f"Error: contract tests failed for {tool_name}. {result}"
    if source_thread:
        return f"OK: {tool_name} is registered; {test_path} contains a test invocation and its tests pass."
    return (
        f"OK: workspace tool {tool_name} has a @tool definition and passing contract tests. "
        "It is not installed in Wallace's built-in registry."
    )


def _module_available(name: str) -> bool:
    """Return whether a Python module can be imported by this interpreter."""
    root = current_workspace_root()
    try:
        result = subprocess.run(
            _sandbox_command([
                sys.executable, "-c",
                "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                name,
            ], shell=False, cwd=root),
            cwd=root, env=_tool_environment(), capture_output=True,
            stdin=subprocess.DEVNULL, timeout=5,
        )
        return result.returncode == 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Git (read-only)
# ---------------------------------------------------------------------------

@tool
def get_git_status() -> str:
    """Show repository-wide Git status, including every untracked file.

    Git inspection uses the repository root even when the current thread's
    file workspace is the narrower ``workspace/`` directory. If no repository
    is available, that is a scope limitation, not evidence that the workspace
    is empty.
    """
    root = _git_root()
    if root is None:
        return GIT_UNAVAILABLE
    args = [
        "git", "status", "--short", "--branch", "--untracked-files=all",
        "--", ".", *_git_pathspec_excludes(),
    ]
    return _run_command(
        args, cwd=root, expose_git_config=True, sandbox_root=root,
    )


@tool
def get_git_diff(staged: bool = False) -> str:
    """Show uncommitted changes as a git diff. Secret files are excluded.

    Args:
        staged: Show staged changes instead of unstaged ones.
    """
    root = _git_root()
    if root is None:
        return GIT_UNAVAILABLE
    args = ["git", "diff"]
    if staged:
        args.append("--staged")
    # Keep sensitive files out of the diff, matching the read restrictions.
    args += ["--", ".", *_git_pathspec_excludes()]
    result = _run_command(
        args, cwd=root, expose_git_config=True, sandbox_root=root,
    )
    return "(no changes)" if result == "Exit code: 0" else result


def _get_session_status() -> str:
    """Show this Wallace process's thread, model, trust mode, and reported token use.

    Provider account balance, remaining quota, and reset time are not
    available from generic model responses. No API keys are read or shown.
    """
    return current_status_text()


async def _get_session_status_async() -> str:
    """Run status in the active async turn without crossing thread boundaries."""
    return current_status_text()


get_session_status = StructuredTool.from_function(
    func=_get_session_status,
    coroutine=_get_session_status_async,
    name="get_session_status",
)


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
    check_source_file,
    summarize_project_file,
    analyze_project_dependencies,
    analyze_git_change_impact,
    write_safe_file,
    replace_in_file,
    edit_file_lines,
    execute_command,
    run_tests,
    run_language_tests,
    verify_tool_contract,
    get_git_status,
    get_git_diff,
    get_system_info,
    list_mcp_connections,
    get_session_status,
]

# Explicitly reviewed read-only tools may run without approval. A new local
# tool is approval-gated automatically until it is deliberately added here.
READ_ONLY_TOOLS = {
    "get_project_tree", "read_safe_file", "search_in_files",
    "get_code_symbols", "check_python_syntax", "summarize_project_file",
    "analyze_project_dependencies", "analyze_git_change_impact",
    "get_git_status", "get_git_diff", "get_system_info",
    "list_mcp_connections", "get_session_status",
}
TOOLS_REQUIRING_APPROVAL = {
    tool.name for tool in TOOLS if tool.name not in READ_ONLY_TOOLS
}
