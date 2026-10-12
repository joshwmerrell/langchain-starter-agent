"""
Core data models for the Wallace planning layer.

This module defines the data structures used throughout the planning system:
- PlanSpec: The structured representation of a user request
- Step: Individual executable units within a plan
- Session: A container for multiple related plans
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class StepStatus:  # noqa: N806
    """Possible states for a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SourceRef:
    """Reference to a source file with optional content snippet."""

    path: str
    lines: tuple[int, int] = (1, 0)  # (start_line, end_line); end_line=0 means EOF
    snippet: Optional[str] = None
    
    def range_str(self) -> str:
        """Return human-readable line range."""
        if self.lines == (1, 0):
            return "whole file"
        start, end = self.lines
        return f"lines {start}-{end}".lstrip("-")


@dataclass
class Step:
    """A single executable unit within a plan."""

    id: str
    title: str
    description: Optional[str] = None
    sources: dict[str, SourceRef] = field(default_factory=dict)
    expected_output: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def mark_running(self):
        """Mark step as currently executing."""
        self.status = StepStatus.RUNNING
        self.updated_at = datetime.utcnow()

    def mark_complete(self, result: Optional[str] = None):
        """Mark step as successfully finished."""
        self.status = StepStatus.COMPLETE
        self.result = result
        self.updated_at = datetime.utcnow()

    def mark_failed(self, reason: str):
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.result = f"Failed: {reason}"
        self.updated_at = datetime.utcnow()

    def mark_skipped(self, reason: str):
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.result = f"Skipped: {reason}"
        self.updated_at = datetime.utcnow()

    def is_pending(self) -> bool:
        """Check if step is pending."""
        return self.status == StepStatus.PENDING

    def is_done(self) -> bool:
        """Check if step is either complete or failed."""
        return self.status in (StepStatus.COMPLETE, StepStatus.FAILED)


@dataclass
class PlanSpec:
    """Structured representation of a user request ready for execution."""

    intent: str
    constraints: dict = field(default_factory=dict)
    sources: dict[str, SourceRef] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    expected_outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: Optional[datetime] = None

    def mark_step_as_started(self, step_id: str):
        """Convenience method to mark a specific step as running."""
        for step in self.steps:
            if step.id == step_id:
                step.mark_running()
                break

    def mark_step_completed(self, step_id: str, result: Optional[str] = None):
        """Convenience method to mark a specific step as complete."""
        for step in self.steps:
            if step.id == step_id:
                step.mark_complete(result)
                break

    def mark_step_failed(self, step_id: str, reason: str):
        """Convenience method to mark a specific step as failed."""
        for step in self.steps:
            if step.id == step_id:
                step.mark_failed(reason)
                break

    def get_next_pending_step(self) -> Optional[Step]:
        """Return the next step that hasn't been started yet."""
        for step in self.steps:
            if step.is_pending():
                return step
        return None

    def get_all_pending_steps(self) -> list[Step]:
        """Return all steps that haven't been started yet."""
        return [step for step in self.steps if step.is_pending()]

    def get_step_by_id(self, step_id: str) -> Optional[Step]:
        """Find a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def update_last_updated(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.utcnow()


@dataclass
class Session:
    """Container for a single planning session."""

    id: str
    name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    input_markdown: dict = field(default_factory=dict)
    plans: dict[str, PlanSpec] = field(default_factory=dict)
    archive_entries: list[dict] = field(default_factory=list)
    archived: bool = False

    def add_plan(self, plan_spec: PlanSpec) -> str:
        """Add a new plan to the session."""
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        plan_spec.metadata.setdefault("plan_id", plan_id)
        self.plans[plan_id] = plan_spec
        self.archive_entries.append({
            "plan_id": plan_id,
            "created_at": datetime.utcnow(),
            "source_request": plan_spec.intent,
        })
        return plan_id

    def get_plan(self, plan_id: str) -> Optional[PlanSpec]:
        """Retrieve a plan by its ID."""
        return self.plans.get(plan_id)

    def get_active_plans(self) -> list[PlanSpec]:
        """Return all plans that aren't marked as archived."""
        return [p for p in self.plans.values() if not getattr(p, 'archived', False)]


# Convenience constants
DEFAULT_SESSION_ID_PREFIX = "session_"
DEFAULT_PLAN_ID_PREFIX = "plan_"
DEFAULT_STEP_ID_PREFIX = "step_"
