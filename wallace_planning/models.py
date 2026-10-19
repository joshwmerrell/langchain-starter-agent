"""Data models for Wallace's durable planning layer.

The models intentionally contain only JSON-friendly state.  This keeps plans
portable between the CLI, TUI, and a future voice/workspace frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import re
import shlex

StepStatus = Literal["pending", "running", "complete", "failed"]
PlanStatus = Literal["active", "complete", "failed"]

MAX_PLAN_REQUEST_CHARS = 12_000
MAX_PLAN_RESULT_CHARS = 4_000
MAX_PLAN_SUMMARY_CHARS = 2_000
MAX_TRACKED_TARGETS = 80
MAX_UNRESOLVED_FAILURES = 20
TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".rst", ".log", ".csv", ".tsv"}


def _tool_work(request: str) -> bool:
    """Only requests to build or repair a tool need a contract gate."""
    action = r"(?:build|add|create|implement|fix|repair|debug|troubleshoot|update)"
    return bool(re.search(
        rf"\b{action}\b.{{0,80}}\btools?\b|\btools?\b.{{0,80}}\b{action}\b",
        request, re.IGNORECASE,
    ))


def _named_tool(request: str) -> str | None:
    match = re.search(
        r"\b(?:tool\s+(?:named|called)\s+|(?:fix|repair|debug|troubleshoot)\s+(?:the\s+)?)"
        r"`?([a-z][a-z0-9]*_[a-z0-9_]+)`?",
        request, re.IGNORECASE,
    )
    return match.group(1) if match else None


def _validation_applies(item: dict[str, Any], target: str) -> bool:
    """A successful check counts only when it covers the edited file."""
    name = item["name"]
    checked = Path(str(item.get("target", "")))
    edited = Path(target)
    suffix = edited.suffix.lower()
    if name == "read_safe_file":
        return suffix in TEXT_ARTIFACT_SUFFIXES and checked == edited
    if name == "check_source_file":
        return checked == edited
    if name == "check_python_syntax":
        return suffix == ".py" and checked == edited
    if name == "run_tests":
        # A scoped test is not evidence about unrelated application modules.
        # Additional coverage mappings must come from the host, never model prose.
        covered = item.get("validation_targets", [])
        return suffix == ".py" and (
            str(checked) in {".", "tests"} or checked == edited
            or target in covered
        )
    if name == "run_language_tests":
        languages = {
            ".py": "python",
            ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".go": "go", ".rs": "rust",
        }
        expected = languages.get(suffix)
        language = str(item.get("language", "")).lower()
        if language == "python":
            return expected == "python" and _validation_applies({**item, "name": "run_tests"}, target)
        return bool(expected and language == expected and (
            checked == Path(".") or edited.is_relative_to(checked)
        ))
    if name == "verify_tool_contract":
        return edited.name == "tools.py"
    return False


def _unscoped_mutation(item: dict[str, Any]) -> bool:
    """Conservatively identify work whose changed files the host did not record."""
    if item.get("changed_paths"):
        return False
    if item.get("name") == "execute_command" and _deletion_target(item) is not None:
        return False
    if item.get("mutation") is not None:
        return bool(item["mutation"]) and not item.get("changed_paths")
    if item.get("name") != "execute_command":
        return False
    command = str(item.get("target", ""))
    if re.search(r"(?:^|\s)(?:sed\s+-i|tee|touch|mkdir|cp|mv|rm|rmdir|unlink|install|patch)\b|(?<![<=>])>{1,2}(?![=])|\bgit\s+(?:apply|checkout|restore|reset|clean|merge|commit|add)\b", command):
        return True
    try:
        words = shlex.split(command)
    except ValueError:
        return True
    if any(character in command for character in ";|&") or any(flag in words for flag in ("-delete", "-exec", "-execdir")):
        return True
    if words and words[0] == "git":
        return len(words) < 2 or words[1] not in {"status", "diff", "log", "show", "ls-files", "rev-parse"}
    if not words:
        return False
    if words[0] in {"pytest", "ruff", "shellcheck"}:
        return False
    if words[0] in {"python", "python3"} and words[1:3] in (["-m", "unittest"], ["-m", "pytest"]):
        return False
    if words[:2] == ["deno", "check"] or (words[0] == "tsc" and "--noEmit" in words):
        return False
    # Arbitrary scripts and shell pipelines can write files. The host can mark
    # a command read-only or supply changed_paths when it knows the actual scope.
    return words[0] not in {
        "pwd", "ls", "cat", "head", "tail", "rg", "grep", "find", "file",
        "stat", "wc", "test", "which", "echo", "printf",
    }


def _activity_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Keep bounded host evidence through serialization and incremental updates."""
    metadata: dict[str, Any] = {}
    if isinstance(item.get("mutation"), bool):
        metadata["mutation"] = item["mutation"]
    for key in ("changed_paths", "validation_targets", "resolved_mutation_ids"):
        values = item.get(key)
        if isinstance(values, list):
            metadata[key] = [bounded(value, 160) for value in values[:MAX_TRACKED_TARGETS] if isinstance(value, str)]
    return metadata


def _target_validated(target: str, validations: list[str]) -> bool:
    checks = set(validations)
    if Path(target).name == "tools.py":
        # The contract checker imports the tool registry in a fresh process
        # and runs an invocation test, so it subsumes a syntax-only check.
        return "verify_tool_contract" in checks
    suffix = Path(target).suffix.lower()
    if suffix in TEXT_ARTIFACT_SUFFIXES:
        valid = "read_safe_file" in checks
    elif suffix == ".py":
        valid = bool(checks & {"check_source_file", "check_python_syntax", "run_tests"})
    elif suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".html", ".css", ".sh"}:
        valid = "check_source_file" in checks
    elif suffix in {".go", ".rs"}:
        valid = bool(checks & {"check_source_file", "run_language_tests"})
    else:
        valid = bool(checks & {"check_source_file", "run_tests", "run_language_tests"})
    return valid


def _blocking_failure(item: dict[str, Any]) -> bool:
    if item["name"] == "check_python_syntax" and Path(str(item["target"])).suffix != ".py":
        # A malformed tool path is not evidence that source code failed a check.
        return False
    if item["name"] in {
        "write_safe_file", "replace_in_file", "edit_file_lines", "check_python_syntax",
        "check_source_file", "run_tests", "run_language_tests", "verify_tool_contract",
        "host_path_absence_check",
    }:
        return True
    return item["name"] == "execute_command" and bool(re.search(
        r"\b(?:check|test|build|compile|lint|fmt|tsc|deno)\b",
        str(item["target"]), re.IGNORECASE,
    ))


def _safe_command_path(raw: str) -> str | None:
    path = Path(raw)
    if (
        path.is_absolute() or ".." in path.parts or str(path) in {"", "."}
        or re.search(r"[;&|*?\[\]{}$`\n\r]", raw)
    ):
        return None
    return path.as_posix()


def _deletion_target(item: dict[str, Any]) -> str | None:
    """Extract one literal relative target from a simple removal command."""
    if item.get("name") != "execute_command":
        return None
    try:
        parts = shlex.split(str(item.get("target", "")))
    except ValueError:
        return None
    if parts[:2] == ["git", "rm"]:
        parts = parts[2:]
    elif parts and parts[0] in {"rm", "rmdir", "unlink"}:
        parts = parts[1:]
    else:
        return None
    paths = [part for part in parts if not part.startswith("-")]
    return _safe_command_path(paths[0]) if len(paths) == 1 else None


def _absence_target(item: dict[str, Any]) -> str | None:
    if item.get("name") != "execute_command":
        return None
    try:
        parts = shlex.split(str(item.get("target", "")))
    except ValueError:
        return None
    if len(parts) != 4 or parts[:2] != ["test", "!"] or parts[2] not in {"-e", "-d", "-f"}:
        return None
    return _safe_command_path(parts[3])


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
    seen_tool_call_ids: list[str] = field(default_factory=list)
    edit_evidence: dict[str, Any] = field(default_factory=dict)
    unresolved_failures: list[dict[str, str]] = field(default_factory=list)
    failure_overflow: bool = False
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
            "seen_tool_call_ids": self.seen_tool_call_ids[-128:],
            "edit_evidence": self.edit_evidence,
            "unresolved_failures": self.unresolved_failures[-MAX_UNRESOLVED_FAILURES:],
            "failure_overflow": self.failure_overflow,
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
        raw_evidence = value.get("edit_evidence")
        if not isinstance(raw_evidence, dict):
            raw_evidence = {}
        raw_validations = raw_evidence.get("validations", [])
        if not isinstance(raw_validations, list):
            raw_validations = []
        raw_targets = raw_evidence.get("targets")
        if not isinstance(raw_targets, dict):
            raw_targets = {}
        targets = {
            bounded(target, 160): {
                "id": bounded(data.get("id", ""), 100),
                "validations": [
                    bounded(name, 80) for name in data.get("validations", [])[:10]
                    if isinstance(name, str)
                ],
            }
            for target, data in list(raw_targets.items())[:MAX_TRACKED_TARGETS]
            if isinstance(target, str) and isinstance(data, dict)
            and isinstance(data.get("validations", []), list)
        }
        # Old plans recorded only the last edit, without the check target.
        # Keep that edit pending rather than trusting an unrelated check.
        if not targets and raw_evidence.get("target"):
            targets[bounded(raw_evidence["target"], 160)] = {
                "id": bounded(raw_evidence.get("id", ""), 100),
                "validations": [],
            }
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
                    "diagnostic": bounded(item.get("diagnostic", ""), 120),
                    "language": bounded(item.get("language", ""), 20),
                    "test_path": bounded(item.get("test_path", ""), 160),
                    "ok": bool(item.get("ok", False)),
                    **_activity_metadata(item),
                }
                for item in value.get("tool_history", [])[-12:]
                if isinstance(item, dict)
            ],
            seen_tool_call_ids=[
                bounded(item, 100)
                for item in value.get("seen_tool_call_ids", [])[-128:]
                if isinstance(item, str) and item
            ] if isinstance(value.get("seen_tool_call_ids", []), list) else [],
            edit_evidence={
                "id": bounded(raw_evidence.get("id", ""), 100),
                "target": bounded(raw_evidence.get("target", ""), 160),
                "requires_contract": bool(raw_evidence.get("requires_contract", False)),
                "validations": [
                    bounded(name, 80)
                    for name in raw_validations[:10]
                    if isinstance(name, str)
                ],
                "targets": targets,
                "target_overflow": bool(raw_evidence.get("target_overflow"))
                or len(raw_targets) > MAX_TRACKED_TARGETS,
                "contract": {
                    "tool_name": bounded(raw_evidence.get("contract", {}).get("tool_name", ""), 80),
                    "test_path": bounded(raw_evidence.get("contract", {}).get("test_path", ""), 160),
                } if isinstance(raw_evidence.get("contract"), dict) else {},
                "unscoped_mutations": [
                    {key: bounded(item.get(key, ""), 160) for key in ("id", "name", "target")}
                    for item in raw_evidence.get("unscoped_mutations", [])[:MAX_TRACKED_TARGETS]
                    if isinstance(item, dict)
                ] if isinstance(raw_evidence.get("unscoped_mutations", []), list) else [],
            } if raw_evidence else {},
            unresolved_failures=[
                {
                    "name": bounded(item.get("name", ""), 80),
                    "target": bounded(item.get("target", ""), 160),
                    "outcome": bounded(item.get("outcome", ""), 80),
                    "diagnostic": bounded(item.get("diagnostic", ""), 120),
                }
                for item in value.get("unresolved_failures", [])[-MAX_UNRESOLVED_FAILURES:]
                if isinstance(item, dict)
            ] if isinstance(value.get("unresolved_failures", []), list) else [],
            failure_overflow=bool(value.get("failure_overflow"))
            or (isinstance(value.get("unresolved_failures"), list)
                and len(value["unresolved_failures"]) > MAX_UNRESOLVED_FAILURES),
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

    def incomplete_handoff(self) -> str | None:
        """Give the host a factual handoff instead of an unverified success claim."""
        targets = self.edit_evidence.get("targets", {})
        unscoped = self.edit_evidence.get("unscoped_mutations", [])
        if (self.status == "complete" and not unscoped) or (
            self.intent not in {"implement", "fix", "refactor", "delete"}
            and not (self.intent == "test" and len(self.steps) >= 3)
            and not targets and not self.unresolved_failures
            and not self.failure_overflow
            and not self.edit_evidence.get("target_overflow")
            and not unscoped
        ):
            return None
        unchecked = self.unchecked_targets()
        details = []
        if unscoped:
            details.append("A command or external tool may have changed files without a recorded change scope; inspect its effects before claiming verification.")
        if self.unresolved_failures:
            failed = self.unresolved_failures[-1]
            detail = failed.get("diagnostic") or failed.get("outcome") or "failed"
            details.append(f"Unresolved {failed['name']} failure: {detail}.")
        if self.failure_overflow or self.edit_evidence.get("target_overflow"):
            details.append("Too many unresolved checks or edited files to verify in this plan.")
        if unchecked:
            details.append("Files needing matching validation: " + ", ".join(unchecked[:5]) + ".")
        if targets and _tool_work(self.request) and not self._contract_verified():
            details.append("A matching tool contract test has not passed after the latest edit.")
        if self.intent == "test" and len(self.steps) >= 3 and self.steps[-2].status != "complete":
            details.append("Required code checks or tests have not passed.")
        if self.intent == "delete" and len(self.steps) >= 4:
            if self.steps[1].status != "complete":
                details.append("No successful deletion command was observed.")
            if self.steps[2].status != "complete":
                details.append("Target absence, Git status, and full project tests have not all been verified after deletion.")
        if not targets and self.intent in {"implement", "fix", "refactor"}:
            details.append("No successful file edit was verified.")
        if not details:
            details.append("A required plan step is still pending.")
        return (
            "Wallace has not verified this request as complete. "
            + " ".join(details)
            + " The plan is saved for a follow-up; the model's completion claim was not accepted."
        )

    def unchecked_targets(self) -> list[str]:
        """Return edited files without a relevant, passing post-edit check."""
        return [
            target for target, record in self.edit_evidence.get("targets", {}).items()
            if not _target_validated(target, record.get("validations", []))
        ]

    def verification_note(self) -> str | None:
        """Distinguish passing source checks from observed behavioral tests."""
        if self.intent == "delete":
            return None
        targets = self.edit_evidence.get("targets", {})
        if not targets:
            return None
        unchecked = self.unchecked_targets()
        if unchecked:
            return "Matching post-edit checks are still missing for " + ", ".join(unchecked[:5]) + "."
        source_only = [
            target for target, record in targets.items()
            if not set(record.get("validations", [])) & {
                "run_tests", "run_language_tests", "verify_tool_contract",
            }
        ]
        if source_only and all(Path(target).suffix.lower() in TEXT_ARTIFACT_SUFFIXES for target in source_only):
            return "Saved text read back: " + ", ".join(source_only[:5]) + "."
        if source_only:
            return (
                "Source/syntax checks only for " + ", ".join(source_only[:5])
                + "; functional behavior was not tested for these files."
            )
        return "Passing tests were observed within their recorded scope; they do not prove every requested behavior."

    def _contract_verified(self) -> bool:
        contract = self.edit_evidence.get("contract", {})
        if not isinstance(contract, dict) or not contract.get("tool_name"):
            return False
        requested = _named_tool(self.request)
        return requested is None or requested == contract["tool_name"]

    def execution_assessment(self) -> str:
        """State what the host actually observed, without inventing a root cause."""
        failed = self.unresolved_failures[-1] if self.unresolved_failures else None
        if failed is None:
            resolved: set[tuple[str, str]] = set()
            for item in reversed(self.tool_history):
                key = (str(item.get("name", "")), str(item.get("target", "")))
                if item.get("ok"):
                    resolved.add(key)
                elif key not in resolved:
                    failed = item
                    break
        if failed is not None:
            name = failed["name"]
            if name in {"run_tests", "run_language_tests", "verify_tool_contract"}:
                return (
                    f"Verification failed in {name}. Inspect the full result, reproduce the "
                    "specific case, then decide whether the tool, test, or setup is faulty."
                )
            return (
                f"Invocation of {name} failed. Check arguments, workspace scope, "
                "approvals, and dependencies before calling the tool implementation faulty."
            )
        if self.edit_evidence.get("targets") and _tool_work(self.request) and not self._contract_verified():
            return "Agent execution gap: edited tool code without a passing matching contract check."
        if self.intent in {"implement", "fix", "refactor"} and not self.edit_evidence.get("targets"):
            return "No successful file edit has been observed yet."
        return "No failure cause is established by the recorded tool results."

    def record_progress(
        self,
        activity: list[dict[str, Any]],
        *,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record observed work, without treating a polite preamble as completion."""
        edits = {"write_safe_file", "replace_in_file", "edit_file_lines"}
        validation = {
            "check_python_syntax", "check_source_file", "run_tests",
            "run_language_tests", "verify_tool_contract", "host_path_absence_check",
        }
        evidence = dict(self.edit_evidence)
        unscoped = list(evidence.get("unscoped_mutations", []))
        targets = {
            path: {"id": record.get("id", ""), "validations": list(record.get("validations", []))}
            for path, record in evidence.get("targets", {}).items()
        }
        failures = [
            failed for failed in self.unresolved_failures
            if not (
                failed.get("name") == "check_python_syntax"
                and Path(str(failed.get("target", ""))).suffix != ".py"
            )
        ]

        def observe_evidence(item: dict[str, Any], index: int) -> None:
            nonlocal evidence, unscoped
            # A failing command can still have written files before it failed.
            if not item["ok"] and not item.get("changed_paths") and not item.get("mutation"):
                return
            resolved = set(item.get("resolved_mutation_ids", [])) if item["ok"] else set()
            unscoped = [event for event in unscoped if event.get("id") not in resolved]
            if _unscoped_mutation(item):
                event = {key: str(item.get(key, "")) for key in ("id", "name", "target")}
                if event not in unscoped:
                    unscoped.append(event)
                if len(unscoped) > MAX_TRACKED_TARGETS:
                    evidence["target_overflow"] = True
                unscoped = unscoped[-MAX_TRACKED_TARGETS:]
            changed = list(item.get("changed_paths", []))
            if item["ok"] and item["name"] in edits:
                changed.append(str(item["target"]))
            for target in dict.fromkeys(changed):
                targets[target] = {
                    "id": str(item["id"] or f"{index}:{item['name']}:{target}"),
                    "validations": [],
                }
                if len(targets) > MAX_TRACKED_TARGETS:
                    targets.pop(next(iter(targets)))
                    evidence["target_overflow"] = True
                evidence = {
                    "id": targets[target]["id"],
                    "target": target,
                    "requires_contract": bool(evidence.get("requires_contract"))
                    or Path(target).name == "tools.py",
                    "validations": [],
                    "target_overflow": bool(evidence.get("target_overflow")),
                    "contract": {},
                }
            if item["ok"] and (item["name"] in validation or item["name"] == "read_safe_file"):
                if item["name"] == "verify_tool_contract":
                    evidence["contract"] = {
                        "tool_name": str(item["target"]),
                        "test_path": str(item.get("test_path", "")),
                    }
                for target, record in targets.items():
                    if _validation_applies(item, target):
                        checked = record["validations"]
                        if item["name"] not in checked:
                            checked.append(item["name"])
                if evidence.get("target") in targets:
                    evidence["validations"] = list(targets[evidence["target"]]["validations"])

        def observe_failure(item: dict[str, Any]) -> None:
            nonlocal failures
            name, target = str(item["name"]), str(item["target"])
            if item["ok"]:
                if name in edits:
                    failures = [
                        failed for failed in failures
                        if not (failed["target"] == target and failed["name"] in edits)
                    ]
                elif name in validation:
                    failures = [
                        failed for failed in failures
                        if not (
                            failed["name"] in validation
                            and (
                                failed["name"] == name
                                or (name == "check_source_file" and failed["name"] == "check_python_syntax")
                            )
                            and (
                                failed["target"] == target
                                or (
                                    name == "run_tests"
                                    and target.rstrip("/") in {".", "tests"}
                                    and failed["name"] == "run_tests"
                                    and failed["target"].rstrip("/") in {"", ".", "tests"}
                                )
                            )
                        )
                    ]
                elif name == "execute_command":
                    failures = [
                        failed for failed in failures
                        if not (failed["name"] == name and failed["target"] == target)
                    ]
            elif _blocking_failure(item):
                failures = [
                    failed for failed in failures
                    if not (failed["name"] == name and failed["target"] == target)
                ]
                failures.append({
                    "name": name,
                    "target": target,
                    "outcome": str(item["outcome"]),
                    "diagnostic": str(item.get("diagnostic", "")),
                })
                if len(failures) > MAX_UNRESOLVED_FAILURES:
                    self.failure_overflow = True
                failures = failures[-MAX_UNRESOLVED_FAILURES:]

        # Older plans may have a journal but no durable edit evidence yet.
        if not targets:
            for index, item in enumerate(self.tool_history):
                observe_evidence(item, index)
                observe_failure(item)
        if not self.seen_tool_call_ids:
            self.seen_tool_call_ids = [
                str(item["id"]) for item in self.tool_history if item.get("id")
            ]
        known_ids = set(self.seen_tool_call_ids)
        known_ids.update(item.get("id") for item in self.tool_history if item.get("id"))
        if evidence.get("id"):
            known_ids.add(evidence["id"])
        observed_history = list(self.tool_history)
        for item in activity:
            call_id = str(item.get("id", ""))
            if call_id and call_id in known_ids:
                continue
            if call_id:
                known_ids.add(call_id)
                self.seen_tool_call_ids.append(bounded(call_id, 100))
            entry = {
                "id": bounded(call_id, 100),
                "name": bounded(item.get("name", ""), 80),
                "target": bounded(item.get("target", ""), 160),
                "outcome": bounded(item.get("outcome", ""), 160),
                "diagnostic": bounded(item.get("diagnostic", ""), 120),
                "language": bounded(item.get("language", ""), 20),
                "test_path": bounded(item.get("test_path", ""), 160),
                "ok": bool(item.get("ok", False)),
                **_activity_metadata(item),
            }
            observed_history.append(entry)
            observe_evidence(entry, len(observed_history) - 1)
            observe_failure(entry)
        self.tool_history = observed_history[-12:]
        self.seen_tool_call_ids = self.seen_tool_call_ids[-128:]
        if evidence:
            evidence["targets"] = targets
            if evidence.get("target") in targets:
                evidence["validations"] = list(targets[evidence["target"]]["validations"])
        if unscoped:
            evidence["unscoped_mutations"] = unscoped
        else:
            evidence.pop("unscoped_mutations", None)
        self.edit_evidence = evidence
        self.unresolved_failures = failures

        successful = [item for item in observed_history if item["ok"]]
        observed = {str(item["name"]) for item in successful}
        inspection = {
            "get_project_tree", "read_safe_file", "search_in_files",
            "get_code_symbols", "summarize_project_file",
            "analyze_project_dependencies", "analyze_git_change_impact",
            "get_git_status", "get_git_diff",
        }

        def finish(step: PlanStep, evidence: str) -> None:
            if step.status != "complete":
                step.status = "complete"
                step.result = bounded(evidence, MAX_PLAN_RESULT_CHARS)
                step.updated_at = utc_now()

        if self.intent == "delete" and len(self.steps) >= 4:
            if observed & inspection:
                finish(self.steps[0], "Deletion scope inspected with tools.")
            deletion_indexes = [
                index for index, item in enumerate(observed_history)
                if item["ok"] and _deletion_target(item) is not None
            ]
            if deletion_indexes:
                self.steps[1].tool_args["_deletion_target"] = _deletion_target(
                    observed_history[deletion_indexes[-1]]
                )
                finish(self.steps[1], "Deletion command completed successfully.")
            # The bounded journal may no longer contain the deletion command;
            # in that case the durable completed step is the evidence anchor.
            after_delete = (
                observed_history[deletion_indexes[-1] + 1:]
                if deletion_indexes else observed_history if self.steps[1].status == "complete" else []
            )
            deleted_target = self.steps[1].tool_args.get("_deletion_target")
            absence_checked = any(
                item["ok"] and deleted_target is not None
                and _absence_target(item) == deleted_target
                for item in after_delete
            )
            git_checked = any(item["ok"] and item["name"] == "get_git_status" for item in after_delete)
            suite_checked = any(
                item["ok"] and item["name"] == "run_tests"
                and str(item["target"]).rstrip("/") in {".", "tests"}
                for item in after_delete
            )
            verified = (
                self.steps[1].status == "complete" and absence_checked and git_checked
                and suite_checked and not failures and not self.failure_overflow and not unscoped
            )
            if verified:
                finish(self.steps[2], "Target absence, Git status, and full project tests verified.")
            elif self.steps[2].status == "complete" and (
                any(item["ok"] and _deletion_target(item) is not None for item in activity)
                or any(not item.get("ok", False) and _blocking_failure(item) for item in activity)
            ):
                self.steps[2].status = "pending"
                self.steps[2].result = "Reverify after the latest deletion."
                self.steps[-1].status = "pending"
                self.steps[-1].result = "Completion was not verified."
            if response and not error and all(step.status == "complete" for step in self.steps[:-1]):
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif self.intent in {"implement", "fix", "refactor"} and len(self.steps) >= 4:
            if observed & inspection:
                finish(self.steps[0], "Project context inspected with tools.")
            validation_names = {
                name for record in targets.values() for name in record["validations"]
            }
            validated_after_edit = bool(targets) and all(
                _target_validated(target, record["validations"])
                for target, record in targets.items()
            ) and (not _tool_work(self.request) or self._contract_verified()) and not failures and not self.failure_overflow and not evidence.get("target_overflow") and not unscoped
            feature_steps = self.steps[1:-2]
            validation_step = self.steps[-2]
            if not validated_after_edit and validation_step.status == "complete":
                validation_step.status = "pending"
                validation_step.result = "Revalidate changed files and resolve failed checks."
                validation_step.updated_at = utc_now()
            if not validated_after_edit and self.steps[-1].status == "complete":
                self.steps[-1].status = "pending"
                self.steps[-1].result = "Completion was not verified."
                self.steps[-1].updated_at = utc_now()
            # These stages describe observed actions, not proof of individual
            # requested features. Preserve legacy plans without inventing a
            # mapping from an arbitrary edit ID to one feature's correctness.
            if len(feature_steps) > 1:
                work_step = PlanStep(
                    "step-02", "Apply requested changes",
                    "Record changed files; the original request remains the scope.",
                    sources=self.sources,
                )
                self.steps = [self.steps[0], work_step, *self.steps[-2:]]
                for number, step in enumerate(self.steps, 1):
                    step.id = f"step-{number:02d}"
                feature_steps = [work_step]
                validation_step = self.steps[-2]
            if targets and feature_steps:
                finish(feature_steps[0], "Successful file edits recorded: " + ", ".join(targets)[:500])
            if validated_after_edit and all(step.status == "complete" for step in feature_steps):
                finish(validation_step, "Passing checks observed: " + ", ".join(sorted(validation_names)) + ". Check scope does not establish every requested behavior.")
            if response and not error and validated_after_edit and all(step.status == "complete" for step in self.steps[:-1]):
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif self.intent == "test" and len(self.steps) >= 3:
            if observed & inspection or observed & validation:
                finish(self.steps[0], "Project context inspected with tools.")
            checks_passed = (
                self.steps[-2].status == "complete" or bool(observed & validation)
            ) and not failures and not self.failure_overflow and not evidence.get("target_overflow") and not unscoped and all(
                _target_validated(target, record["validations"])
                for target, record in targets.items()
            )
            if checks_passed:
                finish(self.steps[-2], "Validation tools completed: " + ", ".join(sorted(observed & validation)))
            else:
                for step in self.steps[-2:]:
                    if step.status == "complete":
                        step.status = "pending"
                        step.result = "Resolve failed checks and verify the result."
                        step.updated_at = utc_now()
            if response and not error and checks_passed and self.steps[0].status == "complete":
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif self.intent in {"review", "research", "test"} and len(self.steps) >= 2:
            if observed & inspection or observed & validation:
                finish(self.steps[0], "Project context inspected with tools.")
            edits_validated = not failures and not self.failure_overflow and not evidence.get("target_overflow") and not unscoped and all(
                _target_validated(target, record["validations"])
                for target, record in targets.items()
            )
            if not edits_validated and self.steps[-1].status == "complete":
                self.steps[-1].status = "pending"
                self.steps[-1].result = "Resolve failed checks and validate changed files."
                self.steps[-1].updated_at = utc_now()
            if response and not error and self.steps[0].status == "complete" and edits_validated:
                finish(self.steps[-1], "Final response provided.\n" + bounded(response, 1_000))
        elif self.steps:
            # A short or ambiguous instruction may still cause file edits.
            # Do not let its generic, one-step plan certify unchecked work.
            validated_after_edit = bool(targets) and all(
                _target_validated(target, record["validations"])
                for target, record in targets.items()
            ) and not failures and not self.failure_overflow and not evidence.get("target_overflow") and not unscoped
            if (targets and not validated_after_edit) or failures or self.failure_overflow or evidence.get("target_overflow") or unscoped:
                self.steps[-1].status = "pending"
                self.steps[-1].result = "Resolve failed tools and validate edited files."
                self.steps[-1].updated_at = utc_now()
            elif response and not error:
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
