"""Provider-neutral, process-local model usage for Wallace's status view.

This module keeps only model names, response IDs, and numeric usage. It never
reads credentials or asks a provider for account billing information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _reported_usage(message: object) -> tuple[int, int, int] | None:
    metadata = getattr(message, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        parsed = _usage_fields(metadata)
        if parsed is not None:
            return parsed
    response_metadata = getattr(message, "response_metadata", None)
    metadata = response_metadata.get("token_usage") if isinstance(response_metadata, Mapping) else None
    return _usage_fields(metadata) if isinstance(metadata, Mapping) else None


def _usage_fields(metadata: Mapping) -> tuple[int, int, int] | None:
    input_tokens = _nonnegative_int(metadata.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _nonnegative_int(metadata.get("prompt_tokens"))
    output_tokens = _nonnegative_int(metadata.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _nonnegative_int(metadata.get("completion_tokens"))
    total_tokens = _nonnegative_int(metadata.get("total_tokens"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    return input_tokens, output_tokens, total_tokens if total_tokens is not None else input_tokens + output_tokens


@dataclass
class UsageTotals:
    responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, previous: tuple[int, int, int] | None, current: tuple[int, int, int]) -> None:
        if previous is None:
            self.responses += 1
            previous = (0, 0, 0)
        self.input_tokens += max(0, current[0] - previous[0])
        self.output_tokens += max(0, current[1] - previous[1])
        self.total_tokens += max(0, current[2] - previous[2])


@dataclass
class SessionUsage:
    """Count provider-reported response tokens since this process started."""

    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    by_model: dict[str, UsageTotals] = field(default_factory=dict)
    _seen: dict[tuple[str, str], tuple[int, int, int]] = field(default_factory=dict, repr=False)

    def record_message(self, message: object, model_name: str) -> None:
        if getattr(message, "type", None) != "ai":
            return
        usage = _reported_usage(message)
        if usage is None:
            return
        response_id = getattr(message, "id", None)
        identity = str(response_id) if response_id else f"object:{id(message)}"
        key = (model_name, identity)
        previous = self._seen.get(key)
        if previous == usage:
            return
        self.by_model.setdefault(model_name, UsageTotals()).add(previous, usage)
        self._seen[key] = usage

    def record_messages(self, messages: Sequence[object], model_name: str) -> None:
        for message in messages:
            self.record_message(message, model_name)

    def render(self, *, thread_id: str, model_name: str, trust_mode: str,
               approval_timeout_seconds: int, workspace: Path) -> str:
        trust = (f"{trust_mode} ({approval_timeout_seconds}s auto-approve)"
                 if trust_mode == "informed-trust" else trust_mode)
        lines = [
            "Wallace session status",
            f"Started: {self.started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Thread: {thread_id}",
            f"Model: {model_name}",
            f"Trust: {trust}",
            f"Workspace: {workspace}",
        ]
        if self.by_model:
            responses = sum(item.responses for item in self.by_model.values())
            input_tokens = sum(item.input_tokens for item in self.by_model.values())
            output_tokens = sum(item.output_tokens for item in self.by_model.values())
            total_tokens = sum(item.total_tokens for item in self.by_model.values())
            lines.append(
                f"Provider-reported usage this process: {responses} responses; "
                f"{input_tokens} input, {output_tokens} output, {total_tokens} total tokens."
            )
            for name, item in sorted(self.by_model.items()):
                lines.append(f"  {name}: {item.total_tokens} tokens in {item.responses} responses")
        else:
            lines.append("Provider-reported usage this process: none received yet.")
        lines.append(
            "Remaining account quota/reset: unavailable from generic model responses; "
            "check your provider's usage dashboard. Counts may be incomplete."
        )
        return "\n".join(lines)


@dataclass
class StatusContext:
    usage: SessionUsage
    thread_id: str
    model_name: str
    trust_mode: str
    approval_timeout_seconds: int
    workspace: Path

    def render(self) -> str:
        return self.usage.render(
            thread_id=self.thread_id,
            model_name=self.model_name,
            trust_mode=self.trust_mode,
            approval_timeout_seconds=self.approval_timeout_seconds,
            workspace=self.workspace,
        )


_ACTIVE_STATUS: ContextVar[StatusContext | None] = ContextVar("wallace_active_status", default=None)


def bind_status(context: StatusContext) -> Token:
    return _ACTIVE_STATUS.set(context)


def reset_status(token: Token) -> None:
    _ACTIVE_STATUS.reset(token)


def current_status_text() -> str:
    context = _ACTIVE_STATUS.get()
    return context.render() if context is not None else "Session status is unavailable outside an active Wallace turn."
