"""Data models for Wallace's durable planning layer.

The models intentionally contain only JSON-friendly state.  This keeps plans
portable between the CLI, TUI, and a future voice/workspace frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StepStatus = Literal["pending", "running", "complete", "failed"]
PlanStatus = Literal["active", "complete", "failed"]

MAX_PLAN_REQUEST_CHARS = 12_000
MAX_PLAN_RESULT_CHARS = 4_000
MAX_PLAN_SUMMARY_CHARS = 2_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def bounded(text: object, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated at {limit} characters]"


@dataclass
class SourceRef:
    path: str
    start_line: int = 1
    end_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "SourceRef":
        if not isinstance(value, dict):
            raise ValueError("source reference must be an object")
        return cls(
            path=str(value.get("path", "")),
            start_line=max(1, int(value.get("start_line", 1))),
            end_line=(
                None
                if value.get("end_line") in (None, "")
                else max(1, int(value["end_line"]))
            ),
        )


@dataclass
class PlanStep:
    id: str
    title: str
    description: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    expected_output: str = ""
    status: StepStatus = "pending"
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [source.to_dict() for source in self.sources]
        data["description"] = bounded(self.description, MAX_PLAN_RESULT_CHARS)
        data["expected_output"] = bounded(self.expected_output, MAX_PLAN_RESULT_CHARS)
        data["result"] = bounded(self.result, MAX_PLAN_RESULT_CHARS)
        return data

    @classmethod
    def from_dict(cls, value: Any) -> "PlanStep":
        if not isinstance(value, dict):
            raise ValueError("plan step must be an object")
        status = value.get("status", "pending")
        if status not in {"pending", "running", "complete", "failed"}:
            status = "failed"
        return cls(
            id=str(value.get("id", "step-unknown")),
            title=str(value.get("title", "Untitled step")),
            description=bounded(value.get("description", ""), MAX_PLAN_RESULT_CHARS),
            sources=[SourceRef.from_dict(item) for item in value.get("sources", [])],
            expected_output=bounded(value.get("expected_output", ""), MAX_PLAN_RESULT_CHARS),
            status=status,
            tool_name=(str(value["tool_name"]) if value.get("tool_name") else None),
            tool_args=dict(value.get("tool_args") or {}),
            result=bounded(value.get("result", ""), MAX_PLAN_RESULT_CHARS),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or utc_now()),
        )


@dataclass
class PlanSpec:
    plan_id: str
    session_id: str
    request: str
    intent: str
    constraints: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    status: PlanStatus = "active"
    summary: str = ""
    last_error: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    archived_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "request": bounded(self.request, MAX_PLAN_REQUEST_CHARS),
            "intent": bounded(self.intent, MAX_PLAN_RESULT_CHARS),
            "constraints": [bounded(item, MAX_PLAN_RESULT_CHARS) for item in self.constraints],
            "sources": [source.to_dict() for source in self.sources],
            "steps": [step.to_dict() for step in self.steps],
            "tool_history": self.tool_history[-12:],
            "expected_outputs": [bounded(item, MAX_PLAN_RESULT_CHARS) for item in self.expected_outputs],
            "status": self.status,
            "summary": bounded(self.summary, MAX_PLAN_SUMMARY_CHARS),
            "last_error": bounded(self.last_error, MAX_PLAN_RESULT_CHARS),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PlanSpec":
        if not isinstance(value, dict):
            raise ValueError("plan must be an object")
        status = value.get("status", "active")
        if status not in {"active", "complete", "failed"}:
            status = "failed"
        plan = cls(
            plan_id=str(value.get("plan_id", "")),
            session_id=str(value.get("session_id", "")),
            request=bounded(value.get("request", ""), MAX_PLAN_REQUEST_CHARS),
            intent=bounded(value.get("intent", ""), MAX_PLAN_RESULT_CHARS),
            constraints=[bounded(item, MAX_PLAN_RESULT_CHARS) for item in value.get("constraints", [])],
            sources=[SourceRef.from_dict(item) for item in value.get("sources", [])],
            steps=[PlanStep.from_dict(item) for item in value.get("steps", [])],
            tool_history=[
                {
                    "id": bounded(item.get("id", ""), 100),
                    "name": bounded(item.get("name", ""), 80),
                    "target": bounded(item.get("target", ""), 160),
                    "outcome": bounded(item.get("outcome", ""), 160),
                    "ok": bool(item.get("ok", False)),
                }
                for item in value.get("tool_history", [])[-12:]
                if isinstance(item, dict)
            ],
            expected_outputs=[bounded(item, MAX_PLAN_RESULT_CHARS) for item in value.get("expected_outputs", [])],
            status=status,
            summary=bounded(value.get("summary", ""), MAX_PLAN_SUMMARY_CHARS),
            last_error=bounded(value.get("last_error", ""), MAX_PLAN_RESULT_CHARS),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or utc_now()),
            archived_at=value.get("archived_at"),
        )
        if not plan.plan_id or not plan.session_id:
            raise ValueError("plan_id and session_id are required")
        return plan

    def next_pending(self) -> PlanStep | None:
        return next((step for step in self.steps if step.status == "pending"), None)

    def refresh_status(self) -> None:
        if any(step.status == "failed" for step in self.steps):
            self.status = "failed"
        elif self.steps and all(step.status == "complete" for step in self.steps):
            self.status = "complete"
        else:
            self.status = "active"
        self.updated_at = utc_now()

    def record_progress(
        self,
        activity: list[dict[str, Any]],
        *,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record observed work, without treating a polite preamble as completion."""
        known_ids = {item.get("id") for item in self.tool_history if item.get("id")}
        observed_history = list(self.tool_history)
        for item in activity:
            call_id = str(item.get("id", ""))
            if call_id and call_id in known_ids:
                continue
            if call_id:
                known_ids.add(call_id)
            entry = {
                "id": bounded(call_id, 100),
                "name": bounded(item.get("name", ""), 80),
                "target": bounded(item.get("target", ""), 160),
                "outcome": bounded(item.get("outcome", ""), 160),
                "ok": bool(item.get("ok", False)),
            }
            observed_history.append(entry)
        self.tool_history = observed_history[-12:]

        successful = [item for item in observed_history if item["ok"]]
        observed = {str(item["name"]) for item in successful}
        inspection = {
            "get_project_tree", "read_safe_file", "search_in_files",
            "get_code_symbols", "summarize_project_file",
            "analyze_project_dependencies", "analyze_git_change_impact",
            "get_git_status", "get_git_diff",
        }
        edits = {"write_safe_file", "replace_in_file"}
        validation = {"check_python_syntax", "run_tests"}

        def finish(step: PlanStep, evidence: str) -> None:
            if step.status != "complete":
                step.status = "complete"
                step.result = bounded(evidence, MAX_PLAN_RESULT_CHARS)
                step.updated_at = utc_now()

        if self.intent in {"implement", "fix", "refactor"} and len(self.steps) >= 4:
            if observed & inspection:
                finish(self.steps[0], "Project context inspected with tools.")
            edit_events = [
                (index, item) for index, item in enumerate(observed_history)
                if item["ok"] and item["name"] in edits
            ]
            latest_edit = edit_events[-1][0] if edit_events else -1
            validations_after_edit = [
                item for index, item in enumerate(observed_history)
                if item["ok"] and item["name"] in validation and index > latest_edit
            ] if latest_edit >= 0 else []
            validated_after_edit = bool(validations_after_edit)
            feature_steps = self.steps[1:-2]
            validation_step = self.steps[-2]
            if latest_edit >= 0 and not validated_after_edit and validation_step.status == "complete":
                validation_step.status = "pending"
                validation_step.result = "Revalidate after the latest edit."
                validation_step.updated_at = utc_now()
            if response and not error and validated_after_edit and feature_steps:
                edit = edit_events[-1][1]
                edit_id = str(edit["id"] or f"{latest_edit}:{edit['name']}:{edit['target']}")
                consumed = {
                    str(step.tool_args.get("_evidence_edit_id"))
                    for step in feature_steps if step.status == "complete"
                }
                if edit_id not in consumed:
                    pending_feature = next((step for step in feature_steps if step.status != "complete"), None)
                    if pending_feature is not None:
                        finish(pending_feature, "Edited and validated: " + str(edit["target"]))
                        pending_feature.tool_args["_evidence_edit_id"] = edit_id
            if validated_after_edit and all(step.status == "complete" for step in feature_steps):
                finish(validation_step, "Validation tool completed: " + ", ".join(sorted({item["name"] for item in validations_after_edit})))
            if response and not error and all(step.status == "complete" for step in self.steps[:-1]):
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif self.intent in {"review", "research", "test"} and len(self.steps) >= 2:
            if observed & inspection or observed & validation:
                finish(self.steps[0], "Project context inspected with tools.")
            if response and not error and self.steps[0].status == "complete":
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif response and not error and self.steps:
            finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))

        if response:
            self.summary = bounded(response, MAX_PLAN_SUMMARY_CHARS)
            self.last_error = ""
        if error:
            self.last_error = bounded(error, MAX_PLAN_RESULT_CHARS)
        self.refresh_status()

    def complete_from_response(self, response: str) -> None:
        """Compatibility shim: a response alone is not proof of tool work."""
        self.record_progress([], response=response)


@dataclass
class IndexRecord:
    session_id: str
    plan_id: str
    status: str
    created_at: str
    updated_at: str
    archived_at: str | None = None
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
