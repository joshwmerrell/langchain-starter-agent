"""Safe, atomic generation and persistence of plan artifacts."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import (
    MAX_PLAN_REQUEST_CHARS,
    PlanSpec,
    PlanStep,
    SourceRef,
    utc_now,
)


@dataclass(frozen=True)
class PlanPaths:
    session_dir: Path
    plan_file: Path
    input_file: Path
    sources_dir: Path
    outputs_dir: Path


def plans_root(project_root: Path) -> Path:
    return project_root.resolve() / ".wallace" / "plans"


def _private_path(project_root: Path, target: Path) -> Path:
    root = plans_root(project_root)
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("plan path escaped private Wallace state")
    return resolved


def plan_paths(project_root: Path, plan: PlanSpec) -> PlanPaths:
    session_dir = plans_root(project_root) / "sessions" / plan.session_id
    return PlanPaths(
        session_dir=session_dir,
        plan_file=session_dir / f"plan_{plan.plan_id}.md",
        input_file=session_dir / f"input_{plan.plan_id}.md",
        sources_dir=session_dir / f"sources_{plan.plan_id}",
        outputs_dir=session_dir / f"outputs_{plan.plan_id}",
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_private(project_root: Path, path: Path, content: str) -> None:
    _private_path(project_root, path)
    _atomic_write(path, content)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _validate_sources(sources: Iterable[SourceRef]) -> list[SourceRef]:
    safe: list[SourceRef] = []
    for source in sources:
        path = Path(source.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or any(part.lower() in {".wallace", ".git", ".venv", "venv"} for part in path.parts)
            or any(part.lower().startswith(".env") for part in path.parts)
        ):
            raise ValueError(f"unsafe source reference: {source.path}")
        safe.append(source)
    return safe


def render_input_markdown(plan: PlanSpec) -> str:
    return f"# Wallace request\n\n{plan.request}\n"


def render_plan_markdown(plan: PlanSpec) -> str:
    lines = [
        f"# Wallace plan `{plan.plan_id}`",
        "",
        f"- Session: `{plan.session_id}`",
        f"- Status: **{plan.status}**",
        f"- Created: `{plan.created_at}`",
        f"- Updated: `{plan.updated_at}`",
        "",
        "## Request and intent",
        "",
        plan.request,
        "",
        f"**Intent:** {plan.intent}",
        "",
        "## Constraints",
        "",
    ]
    lines.extend(f"- {item}" for item in plan.constraints or ["None recorded."])
    lines.extend(["", "## Source references", ""])
    lines.extend(
        f"- `{source.path}`" + (f" (lines {source.start_line}-{source.end_line})" if source.end_line else "")
        for source in plan.sources
    )
    if not plan.sources:
        lines.append("- None recorded; Wallace should inspect relevant files before acting.")
    lines.extend(["", "## Steps", ""])
    for step in plan.steps:
        lines.extend([
            f"### {step.id}: {step.title}",
            "",
            f"- Status: **{step.status}**",
            f"- Description: {step.description or 'No additional description.'}",
            f"- Expected output: {step.expected_output or 'No specific output recorded.'}",
        ])
        if step.tool_name:
            lines.append(f"- Tool: `{step.tool_name}`")
        if step.sources:
            lines.append("- Sources: " + ", ".join(f"`{source.path}`" for source in step.sources))
        if step.result:
            lines.extend(["", "Result:", "", "```text", step.result, "```"])
        lines.append("")
    lines.extend(["## Expected outputs", ""])
    lines.extend(f"- {item}" for item in plan.expected_outputs or ["A concise response describing the result."])
    lines.extend(["", "## Recent tool calls", ""])
    if plan.tool_history:
        lines.extend(
            f"- `{item['name']}` {item['target']}: {item['outcome']}"
            for item in plan.tool_history[-12:]
        )
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Summary", "", plan.summary or "Pending completion.", ""])
    if plan.last_error:
        lines.extend(["## Last error", "", plan.last_error, ""])
    return "\n".join(lines)


def _index_path(project_root: Path) -> Path:
    return plans_root(project_root) / "index.json"


def _read_index(project_root: Path) -> dict:
    try:
        value = json.loads(_index_path(project_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"plans": [], "deleted": []}
    return value if isinstance(value, dict) else {"plans": [], "deleted": []}


def _write_index(project_root: Path, index: dict) -> None:
    root = plans_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_private(project_root, _index_path(project_root), json.dumps(index, indent=2) + "\n")


def _update_index(project_root: Path, plan: PlanSpec, *, archived_at: str | None = None) -> None:
    index = _read_index(project_root)
    records = [item for item in index.get("plans", []) if isinstance(item, dict)]
    record = {
        "session_id": plan.session_id,
        "plan_id": plan.plan_id,
        "status": plan.status,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "archived_at": archived_at or plan.archived_at,
        "deleted_at": None,
    }
    replaced = False
    for position, item in enumerate(records):
        if item.get("plan_id") == plan.plan_id:
            records[position] = {**item, **record}
            replaced = True
            break
    if not replaced:
        records.append(record)
    index["plans"] = records
    _write_index(project_root, index)


def save_plan(project_root: Path, plan: PlanSpec) -> PlanPaths:
    paths = plan_paths(project_root, plan)
    for path in (paths.session_dir, paths.sources_dir, paths.outputs_dir):
        _private_path(project_root, path).mkdir(parents=True, exist_ok=True)
    _write_private(project_root, paths.plan_file, render_plan_markdown(plan))
    _write_private(project_root, paths.plan_file.with_suffix(".json"), json.dumps(plan.to_dict(), indent=2) + "\n")
    _write_private(project_root, paths.input_file, render_input_markdown(plan))
    _update_index(project_root, plan)
    return paths


def create_plan(
    project_root: Path,
    request: str,
    *,
    intent: str = "general task",
    constraints: Iterable[str] = (),
    sources: Iterable[SourceRef] = (),
    steps: Iterable[PlanStep] = (),
    expected_outputs: Iterable[str] = (),
    session_id: str | None = None,
) -> PlanSpec:
    request = request.strip()
    if not request:
        raise ValueError("request cannot be empty")
    if len(request) > MAX_PLAN_REQUEST_CHARS:
        raise ValueError("request is too large for a durable plan")
    session_id = session_id or _new_id("session")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError("invalid session identifier")
    safe_sources = _validate_sources(list(sources))
    plan = PlanSpec(
        plan_id=_new_id("plan"),
        session_id=session_id,
        request=request,
        intent=intent,
        constraints=list(constraints),
        sources=safe_sources,
        steps=list(steps),
        expected_outputs=list(expected_outputs),
    )
    if not plan.steps:
        plan.steps.append(PlanStep(id="step-01", title="Work through the request", description="Use the available tools and model reasoning to complete the request."))
    save_plan(project_root, plan)
    return plan


def load_plan(project_root: Path, session_id: str, plan_id: str) -> PlanSpec:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id + plan_id):
        raise ValueError("invalid plan identifier")
    path = plans_root(project_root) / "sessions" / session_id / f"plan_{plan_id}.json"
    _private_path(project_root, path)
    return PlanSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_plans(project_root: Path) -> list[PlanSpec]:
    result: list[PlanSpec] = []
    root = plans_root(project_root) / "sessions"
    if not root.exists():
        return result
    for path in sorted(root.glob("*/plan_*.json")):
        try:
            result.append(PlanSpec.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            continue
    return result


def active_plan(project_root: Path, session_id: str | None = None) -> PlanSpec | None:
    plans = [plan for plan in list_plans(project_root) if plan.status == "active"]
    if session_id:
        plans = [plan for plan in plans if plan.session_id == session_id]
    return max(plans, key=lambda plan: plan.updated_at, default=None)
