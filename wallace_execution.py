"""Shared tool evidence and bounded model execution for the CLI and TUI."""

from __future__ import annotations

import inspect
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.agents.middleware import wrap_tool_call
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from wallace_planning.quality import response_quality_error

def _content_text(content: object) -> str | None:
    """Extract displayable text from common LangChain content shapes."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip() or None
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        nested = content.get("content")
        if nested is not None:
            return _content_text(nested)
    text = getattr(content, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def assistant_response_text(messages: Sequence[BaseMessage], *, after_instruction: bool = False) -> str | None:
    """Return a final assistant message, never preliminary text or tool output."""
    message_list = list(messages)
    # A model can narrate its intent before calling tools. That narration is
    # not a completion, even if no later AI text arrives.
    last_tool = max(
        (index for index, msg in enumerate(message_list)
         if isinstance(msg, ToolMessage)
         or (after_instruction and getattr(msg, "type", None) == "human")),
        default=-1,
    )
    for index in range(len(message_list) - 1, last_tool, -1):
        msg = message_list[index]
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                return None
            return _content_text(msg.content)

    return None


def no_text_status(plan, activity: list[dict[str, object]]) -> str:
    """Factual host-side handoff when the model never supplies final text."""
    parts = [
        "The model produced no final text after one recovery attempt; "
        "the request is unfinished."
    ]
    if plan is not None:
        pending = plan.next_pending()
        if pending is not None:
            parts.append(f"Next plan step: {pending.title}.")
    if activity:
        last = activity[-1]
        parts.append(
            f"Last observed tool: {last['name']} ({last['outcome']})."
        )
    if not any(
        item.get("changed_paths") or (
            item.get("ok") and item.get("name") in {"replace_in_file", "edit_file_lines", "write_safe_file"}
        )
        for item in activity
    ):
        parts.append("No file edit was observed in this turn.")
    if plan is not None:
        parts.append("The plan and observed tool results are saved for continuation.")
    return " ".join(parts)


class RepetitiveResponseError(RuntimeError):
    """The model completed a call but its text was clearly unusable."""


# Backwards-compatible alias for the old private name.
_assistant_response_text = assistant_response_text


def _safe_tool_diagnostic(name: str, output: str) -> str:
    """Keep only a useful error category, never returned source or command output."""
    if name == "run_tests":
        if "NO TESTS RAN" in output or "Ran 0 tests" in output:
            return "no tests were discovered; check the test directory"
        if "Command timed out" in output:
            return "test run timed out"
        summary = re.search(r"FAILED \((?:failures|errors)=\d+[^)]*\)", output)
        if summary:
            return summary.group(0)
    if name == "replace_in_file":
        if "old_text was not found" in output:
            return "replacement anchor not found; re-read or use edit_file_lines"
        if re.search(r"old_text matches \d+ places", output):
            return "replacement anchor is not unique"
        if "old_text and new_text are identical" in output:
            return "replacement makes no change"
    if name == "edit_file_lines":
        if "line range is invalid" in output:
            return "line range is stale or invalid; re-read the file"
        if "edit spans more than" in output:
            return "line edit is too large; use smaller ranges"
        if "edit makes no change" in output:
            return "line edit makes no change"
    if "not installed" in output:
        return "required checker is not installed"
    if re.search(r"(?:SyntaxError|syntax error|parse error)", output, re.IGNORECASE):
        line = re.search(r"\bline\s+(\d+)\b|:(\d+):\d+", output, re.IGNORECASE)
        return "syntax error" + (f" on line {line.group(1) or line.group(2)}" if line else "")
    type_error = re.search(r"\bTS\d{4,5}\b", output, re.IGNORECASE)
    if type_error is not None:
        return f"TypeScript type-check failed ({type_error.group().upper()}); inspect checker output"
    if "Access denied" in output:
        return "access denied"
    return ""


def tool_result_failed(output: str, status: str | None = None) -> bool:
    """Recognize failed shell/checker results even when ToolMessage is not an error."""
    exit_match = re.match(r"Exit code:\s*(-?\d+)", output)
    return (
        status == "error"
        or output.startswith(("Error", "Access denied"))
        or (exit_match is not None and exit_match.group(1) != "0")
    )


def model_tool_activity(messages: Sequence[BaseMessage]) -> list[dict[str, object]]:
    """Summarize executed tools without persisting file contents or edit bodies."""
    calls = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                if call.get("id"):
                    calls[call["id"]] = call

    activity = []
    seen = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(message.tool_call_id or "")
        if call_id and call_id in seen:
            continue
        seen.add(call_id)
        call = calls.get(call_id, {})
        name = str(message.name or call.get("name") or "tool")
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if name == "execute_command":
            target = str(args.get("command", ""))[:160]
        elif "path" in args:
            target = str(args["path"])[:160]
        elif "test_path" in args:
            target = str(args["test_path"])[:160]
        elif name == "run_tests":
            target = "."
        elif "tool_name" in args:
            target = str(args["tool_name"])[:160]
        else:
            target = ""
        output = _content_text(message.content) or ""
        if (
            name == "check_python_syntax" and target
            and not Path(target).suffix and output.startswith("OK: ")
        ):
            target += ".py"
        exit_match = re.match(r"Exit code:\s*(-?\d+)", output)
        failed = tool_result_failed(output, getattr(message, "status", None))
        # Never persist returned file text or command output in the plan.
        outcome = (
            f"Exit code: {exit_match.group(1)}"
            if exit_match is not None
            else "failed" if failed else "ok"
        )
        observed = {
            "id": call_id,
            "name": name,
            "target": target,
            "outcome": outcome,
            "diagnostic": _safe_tool_diagnostic(name, output) if failed else "",
            "language": str(args.get("language", ""))[:20] if name == "run_language_tests" else "",
            "test_path": str(args.get("test_path", ""))[:160] if name == "verify_tool_contract" else "",
            "ok": not failed,
        }
        metadata = message.additional_kwargs.get("wallace_evidence", {})
        for key in ("mutation", "changed_paths", "validation_targets", "resolved_mutation_ids"):
            if key in metadata:
                observed[key] = metadata[key]
        activity.append(observed)
    return activity


def verified_plan_response(plan, response: str, activity: list[dict[str, object]]) -> str:
    """Prevent an implementation handoff from contradicting observed tool results."""
    plan.record_progress(activity, response=response)
    handoff = plan.incomplete_handoff()
    if handoff is None:
        note = plan.verification_note() if hasattr(plan, "verification_note") else ""
        targets = plan.edit_evidence.get("targets", {})
        checks = {check for record in targets.values() for check in record.get("validations", [])}
        if targets and not checks & {"run_tests", "run_language_tests", "verify_tool_contract"} and re.search(
            r"\b(?:tests?|test suite)\s+(?:have\s+)?(?:passed|pass\b|are passing)|\bfully tested\b",
            response, re.I,
        ):
            response = "Changed: " + ", ".join(targets) + ". Observed checks: " + ", ".join(sorted(checks)) + "."
        return response if not note or note in response else f"{response}\n\n{note}"
    plan.summary = handoff
    plan.last_error = handoff
    return handoff


def pending_work_prompt(plan, messages: Sequence, response: str) -> str | None:
    """Turn observed missing work into a concrete continuation, never a success claim."""
    if plan is None or plan.intent not in {"implement", "fix", "refactor", "delete", "test"}:
        return None
    plan.record_progress(model_tool_activity(messages), response=response)
    if plan.status == "complete":
        return None
    pending = plan.next_pending()
    if pending is None:
        return None
    needed = plan.unchecked_targets()
    detail = " Matching checks needed: " + ", ".join(needed[:5]) + "." if needed else ""
    if plan.intent in {"implement", "fix", "refactor"} and not plan.edit_evidence.get("targets"):
        detail += " No file edit has occurred. Apply the requested change now with a file-edit tool; announcing it is not an edit."
    if plan.unresolved_failures:
        failed = plan.unresolved_failures[-1]
        detail += f" Unresolved result: {failed['name']} — {failed.get('diagnostic') or failed.get('outcome')} ."
    return (
        f"The requested work is unfinished. Next: {pending.title}.{detail} "
        "Use the completed tool results above; do not repeat successful work. "
        "Take the next authorized action and check it. If a missing capability or input "
        "prevents progress, state the concrete blocker and remaining work."
    )


def _repeated_call(call: dict, messages: Sequence) -> bool:
    """Stop an unchanged call after three consecutive attempts, before execution."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    attempts = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for prior in reversed(message.tool_calls or []):
            if prior.get("id") == call.get("id"):
                continue
            if prior.get("id") not in results:
                continue
            if (prior.get("name"), prior.get("args")) != (call.get("name"), call.get("args")):
                return False
            attempts += 1
            if attempts >= 3:
                return True
    return False


def _workspace_state() -> dict | None:
    """Bounded host observation of visible files, never a model-provided scope."""
    from tools import current_workspace_root, SKIP_DIRS, _is_allowed

    root = current_workspace_root()
    state = {}
    deadline = time.monotonic() + 0.5
    try:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS
                       and not (Path(directory) / name).is_symlink()
                       and _is_allowed(Path(directory) / name)]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink() or not _is_allowed(path):
                    continue
                info = path.stat()
                state[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns)
                if len(state) > 5_000 or time.monotonic() > deadline:
                    return None
    except OSError:
        return None
    return state


@wrap_tool_call
async def guard_tool_call(request, handler):
    """Give argument mistakes back to the model; stop unchanged tool loops."""
    call = request.tool_call
    if _repeated_call(call, request.state.get("messages", [])):
        return ToolMessage(
            content="Error: repeated identical tool call stopped. Use the prior result, change the approach, or report the blocker.",
            tool_call_id=call["id"], name=call["name"], status="error",
        )
    try:
        schema = getattr(request.tool, "tool_call_schema", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema.model_validate(call.get("args", {}))
        before = _workspace_state() if call["name"] == "execute_command" else None
        result = await handler(request)
        if isinstance(result, ToolMessage):
            if result.status == "error" and str(result.content).startswith("Error invoking tool"):
                return result.model_copy(update={"content": (
                    f"Error: invalid arguments for {call['name']}. Review the tool schema and correct the arguments."
                )})
            evidence = {}
            if call["name"] == "execute_command" and before is not None:
                after = _workspace_state()
                if after is not None:
                    changed = sorted(path for path in before.keys() | after.keys()
                                     if before.get(path) != after.get(path))
                    evidence = {"mutation": bool(changed), "changed_paths": changed[:80]}
                    if len(changed) > 80:
                        evidence = {"mutation": True}
            metadata = getattr(request.tool, "metadata", None) or {}
            mcp = metadata.get("wallace_mcp", {})
            if mcp:
                read_only = mcp.get("server") == "langchain-docs" or metadata.get("wallace_read_only")
                evidence["mutation"] = not bool(read_only)
            if evidence:
                result = result.model_copy(update={"additional_kwargs": {
                    **result.additional_kwargs, "wallace_evidence": evidence,
                }})
        return result
    except ValidationError as error:
        fields = "; ".join(
            f"{'.'.join(map(str, item['loc']))}: {item['type']}"
            for item in error.errors(include_input=False, include_url=False)[:5]
        )
        return ToolMessage(
            content=f"Error: invalid arguments for {call['name']}: {fields}. Correct the arguments using the tool schema.",
            tool_call_id=call["id"], name=call["name"], status="error",
        )


@dataclass
class TurnResult:
    messages: list
    response: str | None
    config: dict


class TurnRunner:
    """One approval/recovery loop shared by both interfaces.

    UI callbacks display events; permission decisions still belong to the host.
    Checkpoints receive only observed results, before the next model request.
    """

    def __init__(
        self, agent, config: dict, *, approve: Callable,
        malformed_retry: Callable, malformed_error: Callable,
        malformed_prompt: str, empty_prompt: str,
        on_message: Callable | None = None, on_warning: Callable | None = None,
        checkpoint: Callable | None = None, usage=None, model_name: str = "unknown",
        continue_work: Callable | None = None,
        stream: bool = True,
    ):
        self.agent, self.config, self.approve = agent, config, approve
        self.malformed_retry, self.malformed_error = malformed_retry, malformed_error
        self.malformed_prompt, self.empty_prompt = malformed_prompt, empty_prompt
        self.on_message, self.on_warning, self.checkpoint = on_message, on_warning, checkpoint
        self.usage, self.model_name, self.stream = usage, model_name, stream
        self.continue_work = continue_work
        self.rejected = False
        self.messages: list = []
        self._seen: set = set()

    async def _callback(self, callback, *args):
        if callback is not None:
            value = callback(*args)
            if inspect.isawaitable(value):
                return await value
            return value

    async def _accept(self, messages, *, display=True):
        if not isinstance(messages, (list, tuple)):
            messages = [messages]
        has_result = False
        for message in messages:
            if not isinstance(message, BaseMessage):
                continue
            identity = getattr(message, "id", None)
            if isinstance(message, ToolMessage):
                identity = identity or f"tool:{message.tool_call_id}"
            identity = identity or id(message)
            if identity in self._seen:
                continue
            self._seen.add(identity)
            self.messages.append(message)
            if self.usage is not None:
                self.usage.record_message(message, self.model_name)
            if display:
                await self._callback(self.on_message, message)
            has_result |= isinstance(message, ToolMessage)
        if has_result:
            await self._callback(self.checkpoint, self.messages)

    async def _snapshot(self):
        try:
            state = await self.agent.aget_state(self.config)
            messages = state.values.get("messages", [])
        except Exception:
            return
        # Snapshots can contain model tool calls omitted from a streamed update.
        # Their call IDs are needed to associate edits with their actual paths.
        prior = list(self.messages)
        await self._accept(messages, display=False)
        # Place checkpoint-only call messages before their streamed results.
        merged, seen = [], set()
        for message in [*messages, *prior]:
            if not isinstance(message, BaseMessage):
                continue
            identity = getattr(message, "id", None)
            if isinstance(message, ToolMessage):
                identity = identity or f"tool:{message.tool_call_id}"
            identity = identity or id(message)
            if identity not in seen:
                seen.add(identity)
                merged.append(message)
        self.messages = merged
        await self._callback(self.checkpoint, self.messages)

    async def _once(self, payload):
        interrupts = []
        if self.stream:
            async for update in self.agent.astream(payload, self.config, stream_mode="updates"):
                for key, output in update.items():
                    if key == "__interrupt__":
                        interrupts.extend(output)
                    elif isinstance(output, dict) and output.get("messages") is not None:
                        await self._accept(output["messages"])
        else:
            result = await self.agent.ainvoke(payload, self.config)
            interrupts.extend(result.get("__interrupt__") or [])
            await self._accept(result.get("messages") or [])
        return interrupts

    async def run(
        self, prompt: str, *, seed_messages=(), recover_final: bool = True,
        current_request: str = "", previous_assistant: str | None = None,
        clean_prompt: str | None = None, new_config: Callable | None = None,
    ) -> TurnResult:
        payload = {"messages": [*seed_messages, {"role": "user", "content": prompt}]}
        malformed_retried = empty_retried = quality_retried = False
        continuations = 0
        while True:
            try:
                interrupts = await self._once(payload)
            except Exception as error:
                await self._snapshot()
                if malformed_retried or not self.malformed_error(error):
                    raise
                malformed_retried = True
                await self._callback(self.on_warning, "The model returned malformed tool-call JSON; recovering from the saved turn state.")
                payload = await self.malformed_retry(self.agent, self.config, prompt)
                continue
            requests = []
            for interrupt in interrupts:
                value = getattr(interrupt, "value", interrupt)
                if isinstance(value, dict):
                    requests.extend(value.get("action_requests", []))
            if requests:
                decisions = []
                for request in requests:
                    approved = await self._callback(self.approve, request)
                    self.rejected |= not bool(approved)
                    decisions.append({"type": "approve" if approved else "reject"})
                payload = Command(resume={"decisions": decisions})
                continue
            if self.stream:
                await self._snapshot()
            response = assistant_response_text(self.messages, after_instruction=True)
            if recover_final and response is None and not empty_retried:
                empty_retried = True
                await self._callback(self.on_warning, "The model returned no final text; continuing the unfinished turn once.")
                payload = {"messages": [{"role": "user", "content": self.empty_prompt}]}
                continue
            issue = response_quality_error(
                response, previous_assistant=previous_assistant, current_request=current_request,
            ) if response is not None else None
            if issue and clean_prompt and new_config and not quality_retried and not model_tool_activity(self.messages):
                quality_retried = True
                await self._callback(self.on_warning, "The model returned an unusable answer; retrying once without prior replies.")
                self.config = new_config()
                self.messages, self._seen = [], set()
                payload = {"messages": [{"role": "user", "content": clean_prompt}]}
                continue
            if recover_final and issue:
                raise RepetitiveResponseError(issue)
            if response is not None and not self.rejected and continuations < 2:
                followup = await self._callback(self.continue_work, self.messages, response)
                if followup:
                    continuations += 1
                    await self._callback(self.on_warning, "Continuing the next unverified step from observed results.")
                    payload = {"messages": [{"role": "user", "content": followup}]}
                    continue
            return TurnResult(self.messages, response, self.config)


