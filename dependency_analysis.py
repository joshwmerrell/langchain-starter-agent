"""Dependency, module-summary, and change-impact analysis for Wallace."""

from __future__ import annotations

import ast
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_SEVERITY_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


def _requirement_name(value: object) -> str:
    """Return a normalized package name from a PEP 508 string or old dict form."""
    if isinstance(value, dict):
        value = value.get("name") or value.get("package") or value.get("requirement") or ""
    if not isinstance(value, str):
        return ""
    match = _REQUIREMENT_NAME.match(value)
    return match.group(1).lower().replace("_", "-") if match else ""


def _requirement_specifier(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("requirement") or value.get("specifier") or value.get("version") or ""
    if not isinstance(value, str):
        return ""
    match = _REQUIREMENT_NAME.match(value)
    return value[match.end():].strip() if match else ""


def _module_name(path: Path) -> str:
    return path.stem


class DependencyGraph:
    """Forward and reverse dependency relationships for a Python workspace."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root.resolve()
        self.modules: dict[str, set[str]] = defaultdict(set)
        self.reverse_deps: dict[str, set[str]] = defaultdict(set)

    def load_from_pyproject(self, path: Path | None = None) -> None:
        """Load declared project dependencies from pyproject.toml."""
        target = (path or self.root / "pyproject.toml").resolve()
        try:
            with target.open("rb") as file:
                data = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError):
            return

        for dependency in data.get("project", {}).get("dependencies", []):
            name = _requirement_name(dependency)
            if name:
                self.modules.setdefault(name, set())

    def extract_import_dependencies(self, path: Path) -> None:
        """Add imports from one Python file to the graph."""
        target = path.resolve()
        try:
            relative = target.relative_to(self.root)
            tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            return

        module = _module_name(target)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])

        self.modules[module] = imported
        for dependency in imported:
            self.reverse_deps[dependency].add(module)

    def build(self, *, include_nested: bool = True) -> "DependencyGraph":
        """Load pyproject dependencies and all readable Python modules."""
        self.load_from_pyproject()
        pattern = "**/*.py" if include_nested else "*.py"
        for path in self.root.glob(pattern):
            if any(
                part in {".git", ".venv", "venv", "__pycache__", ".wallace", "node_modules"}
                for part in path.relative_to(self.root).parts
            ):
                continue
            self.extract_import_dependencies(path)
        return self

    def get_dependents(self, module: str) -> list[str]:
        """Return modules that import the requested module."""
        name = Path(module).stem
        return sorted(self.reverse_deps.get(name, set()))

    def get_all_transitive_deps(self, module: str) -> list[str]:
        """Return reachable dependencies in deterministic order."""
        start = Path(module).stem
        visited: set[str] = set()
        result: list[str] = []
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for dependency in sorted(self.modules.get(current, set()), reverse=True):
                if dependency not in visited:
                    result.append(dependency)
                    stack.append(dependency)
        return sorted(set(result))

    def summarize_module(self, module_path: str | Path) -> dict[str, Any]:
        """Return public API, imports, dependents, and size for one module."""
        target = Path(module_path)
        if not target.is_absolute():
            target = self.root / target
        target = target.resolve()
        summary: dict[str, Any] = {
            "path": str(target.relative_to(self.root)) if target.is_relative_to(self.root) else str(target),
            "public_api": [],
            "imports": [],
            "dependents": [],
            "transitive_deps": [],
            "size_bytes": 0,
            "line_count": 0,
        }
        try:
            content = target.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(target))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            summary["error"] = str(error)
            return summary

        summary["size_bytes"] = len(content.encode("utf-8"))
        summary["line_count"] = content.count("\n") + (1 if content else 0)
        summary["public_api"] = sorted(
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        )
        self.extract_import_dependencies(target)
        name = _module_name(target)
        summary["imports"] = sorted(self.modules.get(name, set()))
        summary["dependents"] = self.get_dependents(name)
        summary["transitive_deps"] = self.get_all_transitive_deps(name)
        return summary


def _change_kind(hunk: dict[str, Any]) -> tuple[str, str]:
    if hunk.get("deleted") or hunk.get("removed"):
        return "removed", "high"
    if hunk.get("added"):
        return "added", "low"
    return "modified", "medium"


def analyze_change_impact(diff: dict[str, Any], dep_graph: DependencyGraph) -> list[dict[str, Any]]:
    """Analyze structured diff entries against a dependency graph."""
    entries: dict[str, dict[str, Any]] = {}
    for change in diff.get("diff", []):
        if not isinstance(change, dict):
            continue
        old_file = change.get("old_file")
        new_file = change.get("new_file")
        path = new_file or old_file
        if not path:
            continue
        module = Path(str(path).removeprefix("a/").removeprefix("b/")).stem
        entry = entries.setdefault(
            module,
            {"module": str(path), "changes": [], "severity": "unknown"},
        )
        hunks = change.get("hunks") or [{}]
        for hunk in hunks:
            kind, severity = _change_kind(hunk if isinstance(hunk, dict) else {})
            entry["changes"].append({"type": kind, "target": str(path)})
            if _SEVERITY_RANK[severity] > _SEVERITY_RANK[entry["severity"]]:
                entry["severity"] = severity

        dependents = dep_graph.get_dependents(module)
        if dependents:
            entry["dependents_affected"] = dependents
            entry["severity"] = "high"
        transitive = dep_graph.get_all_transitive_deps(module)
        if transitive:
            entry["transitive_deps"] = transitive
    return [entries[name] for name in sorted(entries)]


def compare_versions(old_version: dict[str, Any], new_version: dict[str, Any]) -> list[dict[str, str]]:
    """Compare project dependency names and specifiers between two TOML maps."""
    def requirements(data: dict[str, Any]) -> dict[str, str]:
        result = {}
        for value in data.get("project", {}).get("dependencies", []):
            name = _requirement_name(value)
            if name:
                result[name] = _requirement_specifier(value)
        return result

    old = requirements(old_version)
    new = requirements(new_version)
    changes: list[dict[str, str]] = []
    for package in sorted(new.keys() - old.keys()):
        changes.append({"action": "added", "package": package, "new_version": new[package]})
    for package in sorted(old.keys() - new.keys()):
        changes.append({"action": "removed", "package": package, "old_version": old[package]})
    for package in sorted(old.keys() & new.keys()):
        if old[package] != new[package]:
            changes.append({
                "action": "updated",
                "package": package,
                "old_version": old[package],
                "new_version": new[package],
            })
    return changes


if __name__ == "__main__":
    graph = DependencyGraph().build()
    print(f"Analyzed {len(graph.modules)} modules and packages.")
    for module in ("main.py", "tools.py", "tui_agent.py"):
        print(graph.summarize_module(module))
