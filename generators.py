"""
Session and plan generation utilities for the Wallace planning layer.

This module handles creating session directories, allocating stable IDs,
and writing the initial input and plan markdown files atomically.
"""

import os
import tempfile
import uuid
from fnmatch import fnmatch
from pathlib import Path
from datetime import datetime

from models import PlanSpec, Session, SourceRef, StepStatus


# ---------------------------------------------------------------------------
# Safe path resolution
# ---------------------------------------------------------------------------

SAFE_ROOT = Path(__file__).resolve().parent
HIDDEN_DIRS = {".git", ".wallace"}
SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key"]
MAX_WRITE_CHARS = 50_000


def _is_allowed(resolved: Path) -> bool:
    """Whether a resolved path is inside SAFE_ROOT and not sensitive."""
    if not resolved.is_relative_to(SAFE_ROOT):
        return False
    parts = resolved.relative_to(SAFE_ROOT)
    if any(part.lower() in HIDDEN_DIRS for part in parts):
        return False
    return not any(fnmatch(part.lower(), pattern) for part in parts for pattern in SENSITIVE_PATTERNS)


def _resolve(path: str) -> Path | None:
    """Resolve a project-relative path, or None if access isn't allowed."""
    try:
        target = (SAFE_ROOT / path).resolve()
    except (OSError, ValueError):
        return None
    return target if _is_allowed(target) else None


def _atomic_write_text(target: Path, content: str) -> None:
    """Write text beside the target and replace it atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = os.stat(target).st_mode & 0o7777 if target.exists() else None
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
# Session management
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages session lifecycle, ID allocation, and storage paths."""

    def __init__(self, base_dir: Path):  # e.g., "/project/.wallace/plans/sessions/"
        self.base_dir = base_dir
        self._sessions: dict[str, Session] = {}
        self._next_session_counter = 0

    @property
    def active_sessions(self) -> list[Session]:
        """Return all sessions that are not archived."""
        return [s for s in self._sessions.values() if not getattr(s, 'archived', False)]

    def allocate_session_id(self) -> str:
        """Generate a unique session ID and create the session directory."""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = Session(id=session_id)
        return session_id

    def get_or_create_session(self, name: str | None = None) -> Session:
        """Get an existing session by name, or create a new one."""
        for session in self.active_sessions:
            if session.name == name:
                return session
        new_id = self.allocate_session_id()
        session = self._sessions[new_id]
        session.name = name
        return session

    def archive_session(self, session_id: str):
        """Archive a session so it's no longer considered active."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.archived = True
            del self._sessions[session_id]


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


def generate_plan_files(session_manager: SessionManager, plan_spec: PlanSpec) -> str:
    """Create session/turn directories and write initial plan files atomically.

    Returns the path to the newly created plan file.
    """
    # Determine which session to store this plan in
    session = session_manager.get_or_create_session(name=plan_spec.metadata.get('session_name'))
    plan_id = session.add_plan(plan_spec)
    plan_dir = session_manager.base_dir / session.id / plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    # Build the plan markdown content
    plan_content = _build_plan_markdown(plan_spec)

    # Write atomically
    plan_file = plan_dir / f"plan_{plan_id}.md"
    _atomic_write_text(plan_file, plan_content)

    return str(plan_file.relative_to(SAFE_ROOT))


def generate_input_files(session_manager: SessionManager, plan_spec: PlanSpec) -> list[str]:
    """Create input_<turn>.md files for each source reference atomically.

    Only generates files for sources that don't already have an input file.
    Returns list of paths to newly created input files.
    """
    session = session_manager.get_or_create_session(name=plan_spec.metadata.get('session_name'))
    plan = session.plans.get(plan_spec.metadata.get('plan_id'))

    if plan is None:
        return []

    newly_created: list[str] = []

    for source_key, source_ref in plan.sources.items():
        # Check if input file already exists
        plan_dir = session_manager.base_dir / session.id / plan_spec.metadata["plan_id"]
        input_file = plan_dir / f"input_{source_key}.md"
        if input_file.exists():
            continue

        # Generate input file content
        input_content = _build_input_markdown(source_ref, plan_spec)

        # Write atomically
        _atomic_write_text(input_file, input_content)
        newly_created.append(str(input_file.relative_to(SAFE_ROOT)))

    return newly_created


def generate_outputs_directories(session_manager: SessionManager, plan_spec: PlanSpec) -> dict[str, str]:
    """Create outputs_<turn>/ directories for each output location.

    Returns mapping from output key to directory path.
    """
    session = session_manager.get_or_create_session(name=plan_spec.metadata.get('session_name'))
    plan = session.plans.get(plan_spec.metadata.get('plan_id'))

    if plan is None:
        return {}

    plan_dir = session_manager.base_dir / session.id / plan_spec.metadata["plan_id"]
    outputs_dir = plan_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    for output_key, output_location in plan.expected_outputs.items():
        rel_path = output_location.strip('/')
        full_path = outputs_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths[output_key] = str(full_path.relative_to(SAFE_ROOT))

    return output_paths


# ---------------------------------------------------------------------------
# Markdown templating helpers
# ---------------------------------------------------------------------------


def _format_tool_call(call: dict) -> str:
    """Format a tool call as readable Python-like syntax."""
    args = ", ".join(f"{k}={repr(v)}" for k, v in (call.get("args") or {}).items())
    return f"tools.execute('{call.get('name')}', args={args})"


def _format_source_ref(ref: SourceRef) -> str:
    """Format a source reference as a markdown list item."""
    lines = [f"- **Source:** [`{ref.path}`](project://{ref.path})"]
    if ref.lines != (1, 0):
        lines.append(f"  Lines: `{ref.range_str()}`")
    if ref.snippet:
        lines.append(f"  Snippet:\n```\n{ref.snippet}\n```")
    return "\n".join(lines)


def _build_plan_markdown(spec: PlanSpec) -> str:
    """Build the complete plan_<turn>.md content from a PlanSpec."""

    header = f"# Plan {spec.metadata.get('plan_id', 'unknown')}"
    
    body_lines = [header]

    # 1. Request and intent
    body_lines.append("")
    body_lines.append(f"## Intent")
    body_lines.append(spec.intent)

    # 2. Constraints
    if spec.constraints:
        body_lines.append("")
        body_lines.append("## Constraints")
        for key, value in sorted(spec.constraints.items()):
            body_lines.append(f"- **{key}:** {value}")

    # 3. Source references
    if spec.sources:
        body_lines.append("")
        body_lines.append("## Sources")
        for key, ref in sorted(spec.sources.items()):
            body_lines.append(_format_source_ref(ref))

    # 4. Ordered steps
    if spec.steps:
        body_lines.append("")
        body_lines.append("## Steps")
        for i, step in enumerate(spec.steps, 1):
            step_num = step.metadata.get('step_number', i)
            body_lines.append("")
            body_lines.append(f"### Step {step_num}: {step.title}")
            if step.description:
                body_lines.append(step.description)
            if step.sources:
                body_lines.append("")
                for key, ref in sorted(step.sources.items()):
                    body_lines.append(_format_source_ref(ref))
            if step.expected_output:
                body_lines.append("")
                body_lines.append("#### Expected Output")
                body_lines.append(step.expected_output)

    # 5. Status summary
    body_lines.append("")
    body_lines.append("## Current Status")
    for step in spec.steps:
        status_label = {
            StepStatus.PENDING: "⏳ Pending",
            StepStatus.RUNNING: "🔄 Running",
            StepStatus.COMPLETE: "✅ Complete",
            StepStatus.FAILED: "❌ Failed",
            StepStatus.SKIPPED: "⏭️ Skipped",
        }.get(step.status, "? Unknown?")
        step_info = f"- **{step.id}:** {status_label}"
        if step.result:
            step_info += f" — *{step.result}*"
        body_lines.append(step_info)

    # 6. Tool calls and bounded results
    if spec.metadata.get('tool_calls'):
        body_lines.append("")
        body_lines.append("## Tool Calls & Results")
        for tool_call in spec.metadata['tool_calls']:
            formatted_call = _format_tool_call(tool_call)
            result = spec.metadata.get('tool_results', {})
            result_entry = f"- **{formatted_call}**: {result.get(formatted_call, 'No result recorded yet') or '*Pending*'}"
            body_lines.append(result_entry)

    last_updated = spec.last_updated or spec.created_at
    footer_lines = [
        "",
        f"| Timestamp | Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        "|----------|-------------------------------------------------------------|",
        f"| Created   | {spec.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
    ]
    body_lines.extend(footer_lines)

    return "\n".join(body_lines)


def _build_input_markdown(source_ref: SourceRef, parent_spec: PlanSpec) -> str:
    """Build input_<turn>.md content for a single source reference."""

    lines = [
        f"# Input: {source_ref.path}",
        "",
        f"**Referenced by:** Step {source_ref.path.split('/')[-1].split('_')[0] if '_' in source_ref.path else source_ref.path}",
        "",
        f"Lines: `{source_ref.range_str()}`",
    ]

    if source_ref.snippet:
        lines.append("")
        lines.append("Content:")
        lines.append("```")
        lines.append(source_ref.snippet)
        lines.append("```")

    # Add relevant excerpt from parent spec if available
    if parent_spec.metadata.get('context_for_source'):
        lines.append("")
        lines.append("Relevant context:")
        lines.append(parent_spec.metadata['context_for_source'])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def build_progress_tracker(plan_spec: PlanSpec) -> dict:
    """Build a progress tracker dictionary for the plan."""
    tracker = {
        "total_steps": len(plan_spec.steps),
        "pending_steps": sum(1 for s in plan_spec.steps if s.is_pending()),
        "running_steps": sum(1 for s in plan_spec.steps if s.status == StepStatus.RUNNING),
        "complete_steps": sum(1 for s in plan_spec.steps if s.status == StepStatus.COMPLETE),
        "failed_steps": sum(1 for s in plan_spec.steps if s.status == StepStatus.FAILED),
        "skipped_steps": sum(1 for s in plan_spec.steps if s.status == StepStatus.SKIPPED),
        "completion_percentage": round(
            (
                sum(
                    1
                    for s in plan_spec.steps
                    if s.status in (StepStatus.COMPLETE, StepStatus.SKIPPED)
                )
                / len(plan_spec.steps)
            )
            * 100
            if plan_spec.steps
            else 0,
            1,
        ),
    }
    return tracker
