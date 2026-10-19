"""Durable thread transcripts, bounded context, and local conversation retrieval."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar, Token
from pathlib import Path

from langchain_core.tools import tool
from wallace_config import (
    MAX_HISTORY_MESSAGES, MAX_HISTORY_CHARS, MAX_MEMORY_CHARS,
    MEMORY_RECENT_MESSAGES, MEMORY_COMPACT_AFTER_MESSAGES,
)
from wallace_runtime import APP_ROOT
from wallace_planning.threads import thread_file_lock
from wallace_planning.quality import response_quality_error

CONVERSATION_PATH = APP_ROOT / ".wallace" / "conversation.json"

class ConversationStore:
    """Retain a complete thread and compact its model-facing memory."""

    def __init__(self, path: Path = CONVERSATION_PATH, pending_path: Path | None = None) -> None:
        self.path = path
        self._pending_path = pending_path

    @staticmethod
    def _valid_messages(raw: object) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        history: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                history.append({"role": role, "content": content})
        return history

    def _read_state(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 2, "messages": [], "summary": "", "summarized_through": 0}
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read conversation history: {error}") from error
        if isinstance(raw, list):  # History written before rolling memory existed.
            raw = {"messages": raw}
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            raise ValueError("conversation history has an invalid format")
        messages = self._valid_messages(raw["messages"])
        through = raw.get("summarized_through", 0)
        invalid_pointer = (
            not isinstance(through, int)
            or isinstance(through, bool)
            or not 0 <= through <= len(messages)
            or len(messages) != len(raw["messages"])
        )
        if invalid_pointer:
            through = 0
        summary = raw.get("summary", "")
        return {
            "version": 2,
            "messages": messages,
            "summary": summary if isinstance(summary, str) and not invalid_pointer else "",
            "summarized_through": through,
        }

    def load_all(self) -> list[dict[str, str]]:
        """Return the full retained transcript, including compacted turns."""
        return self._read_state()["messages"]

    def load(self) -> list[dict[str, str]]:
        """Return only the recent turns used by the visible TUI log."""
        try:
            return self.trim(self.load_all())
        except ValueError:
            return []

    @staticmethod
    def trim(history: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        """Bound history by complete messages and approximate character size."""
        bounded = [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ]
        bounded = bounded[-MAX_HISTORY_MESSAGES:]
        total = sum(len(item["content"]) for item in bounded)
        while bounded and total > MAX_HISTORY_CHARS:
            total -= len(bounded.pop(0)["content"])
        if bounded and bounded[0]["role"] == "assistant":
            bounded.pop(0)
        return bounded

    def save(self, history: Sequence[dict[str, str]]) -> None:
        """Replace a transcript explicitly; regular turns should use append()."""
        with thread_file_lock(self.path.with_suffix(".lock")):
            self._write_state({
                "version": 2,
                "messages": self._valid_messages(list(history)),
                "summary": "",
                "summarized_through": 0,
            })

    def append(self, role: str, content: str) -> None:
        """Append one text message without discarding older turns or memory."""
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("conversation message must have a nonempty user or assistant role")
        with thread_file_lock(self.path.with_suffix(".lock")):
            state = self._read_state()
            state["messages"].append({"role": role, "content": content})
            self._write_state(state)

    def _write_state(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix="conversation-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(state, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _memory_line(message: dict[str, str], limit: int) -> str:
        label = "User" if message["role"] == "user" else "Previous Wallace reply (unverified)"
        if message["role"] == "assistant":
            limit = min(limit, 220)
        content = message["content"].strip().replace("\x00", "")
        if len(content) > limit:
            marker = " … [middle omitted] … "
            retained = max(2, limit - len(marker))
            head = max(1, retained * 3 // 4)
            content = content[:head] + marker + content[-(retained - head):]
        return f"{label}: {content}"

    @staticmethod
    def _safe_memory_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        """Keep transcript positions but withhold unusable assistant prose from prompts."""
        safe: list[dict[str, str]] = []
        previous_assistant = None
        current_request = ""
        for message in messages:
            if message["role"] == "user":
                current_request = message["content"]
                safe.append(message)
                continue
            content = message["content"]
            if response_quality_error(
                content,
                previous_assistant=previous_assistant,
                current_request=current_request,
            ):
                safe.append({
                    "role": "assistant",
                    "content": "[Unusable earlier Wallace reply omitted; no result verified.]",
                })
            else:
                safe.append(message)
            previous_assistant = content
        return safe

    async def memory_context(
        self,
        current_prompt: str,
        summarizer: Callable[[str, list[dict[str, str]], int], Awaitable[str]],
        *,
        max_chars: int = MAX_MEMORY_CHARS,
    ) -> tuple[str, str | None]:
        """Summarize older turns and return bounded context for one model turn.

        The current request is already supplied separately, so exclude its
        just-persisted user message. Failed compaction never deletes raw turns.
        """
        max_chars = max(400, max_chars)
        state = self._read_state()
        messages = state["messages"]
        prior = messages[:-1] if messages and messages[-1] == {"role": "user", "content": current_prompt} else messages
        if not prior:
            return "", None
        safe_prior = self._safe_memory_messages(prior)

        summary_budget = max(160, MAX_MEMORY_CHARS * 2 // 5)
        displayed_summary_budget = min(summary_budget, max(160, max_chars * 2 // 5))
        recent_budget = max_chars - displayed_summary_budget - 150
        through = min(state["summarized_through"], len(prior))
        pending = safe_prior[through:]
        pending_chars = sum(len(item["content"]) + 48 for item in pending)
        should_compact = (
            len(pending) > MEMORY_COMPACT_AFTER_MESSAGES
            or pending_chars > recent_budget
        )
        cutoff = through
        if should_compact:
            cutoff = max(through, len(prior) - MEMORY_RECENT_MESSAGES)
            while cutoff < len(prior) - 2 and sum(
                min(len(item["content"]), 1_200) + 48 for item in safe_prior[cutoff:]
            ) > recent_budget:
                cutoff += 1

        warning = None
        # One bounded summarization per turn avoids a chain of hidden model calls.
        compaction_calls = 0
        while through < cutoff and compaction_calls < 1:
            batch = []
            batch_chars = 0
            for message in safe_prior[through:cutoff]:
                excerpt = self._memory_line(message, 1_200)
                if batch and (batch_chars + len(excerpt) > MAX_MEMORY_CHARS * 2 or len(batch) >= 40):
                    break
                batch.append(message)
                batch_chars += len(excerpt)
            compaction_calls += 1
            try:
                updated = (await summarizer(state["summary"], batch, summary_budget)).strip()
                if not updated:
                    raise ValueError("memory summarizer returned no text")
            except Exception as error:
                warning = (
                    "Could not summarize older thread turns; they remain on disk "
                    f"({type(error).__name__})."
                )
                break
            if len(updated) > summary_budget:
                marker = "\n…\n"
                head = (summary_budget - len(marker)) // 2
                updated = updated[:head] + marker + updated[-(summary_budget - len(marker) - head):]
            previous_through = through
            with thread_file_lock(self.path.with_suffix(".lock")):
                current = self._read_state()
                if (current["messages"][:len(messages)] != messages
                        or current["summarized_through"] != previous_through):
                    warning = "Thread changed during memory compaction; new turns remain saved."
                    state = current
                    through = min(current["summarized_through"], len(prior))
                    break
                state["summary"] = updated
                through += len(batch)
                current["summary"] = updated
                current["summarized_through"] = through
                self._write_state(current)

        heading = (
            "Conversation memory for this thread (reference only; current request wins). "
            "Older summary status may be stale; the recent exchange supersedes it. "
            "Prior assistant claims are unverified.\n"
        )
        summary = state["summary"]
        if len(summary) > displayed_summary_budget:
            marker = "\n…\n"
            head = (displayed_summary_budget - len(marker)) // 2
            summary = summary[:head] + marker + summary[-(displayed_summary_budget - len(marker) - head):]
        body = f"Older thread summary:\n{summary}\n" if summary else ""
        recent_heading = "Recent exchange (oldest first):\n"
        missing_note = "\nSome older turns are temporarily unavailable in model context." if warning else ""
        available = max(0, max_chars - len(heading) - len(body) - len(recent_heading) - len(missing_note))
        selected: list[str] = []
        for message in reversed(safe_prior[through:]):
            excerpt_limit = (
                min(600, max(60, available // 3))
                if message["role"] == "assistant"
                else min(1_200, max(60, available))
            )
            line = self._memory_line(message, excerpt_limit)
            if len(line) + 1 > available and selected:
                break
            if len(line) + 1 > available:
                line = line[:available]
            selected.append(line)
            available -= len(line) + 1
            if available <= 0:
                break
        if selected:
            body += recent_heading + "\n".join(reversed(selected))
        body += missing_note
        return (heading + body)[:max_chars], warning

    def clear(self) -> None:
        """Remove persisted conversational history, if it exists."""
        try:
            with thread_file_lock(self.path.with_suffix(".lock")):
                self.path.unlink(missing_ok=True)
                self.pending_path.unlink(missing_ok=True)
        except OSError:
            pass

    @property
    def pending_path(self) -> Path:
        """Path for the user request currently being processed."""
        return self._pending_path or self.path.with_name("pending_turn.json")

    def save_pending(self, prompt: str) -> None:
        """Persist the active user request before any tools are called."""
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.pending_path.parent,
                prefix="pending-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump({"prompt": prompt}, temporary, ensure_ascii=False)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.pending_path)
            try:
                os.chmod(self.pending_path, 0o600)
            except OSError:
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_pending(self) -> str | None:
        """Return the request that was active if Wallace was interrupted."""
        try:
            raw = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        prompt = raw.get("prompt") if isinstance(raw, dict) else None
        return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None

    def clear_pending(self) -> None:
        """Mark the active request complete."""
        try:
            self.pending_path.unlink(missing_ok=True)
        except OSError:
            pass


_active_conversation: ContextVar[ConversationStore | None] = ContextVar(
    "wallace_conversation", default=None
)


def bind_conversation(store: ConversationStore) -> Token:
    return _active_conversation.set(store)


def reset_conversation(token: Token) -> None:
    _active_conversation.reset(token)


@tool
def search_conversation(query: str, limit: int = 5) -> str:
    """Find older details in the current thread. Earlier assistant claims are unverified."""
    store = _active_conversation.get()
    if store is None:
        return "Error: no conversation is bound to this turn."
    query = query.strip()
    if not query or len(query) > 200:
        return "Error: use a search query of 1–200 characters."
    limit = max(1, min(limit, 5))
    words = set(re.findall(r"\w+", query.casefold()))
    if not words:
        return "Error: include at least one word to search for."
    try:
        messages = store._safe_memory_messages(store.load_all())
    except (OSError, ValueError):
        return "Error: conversation history is unavailable."
    matches = []
    for index, message in enumerate(messages):
        content = message["content"]
        folded = content.casefold()
        score = len(words & set(re.findall(r"\w+", folded)))
        if query.casefold() in folded:
            score += len(words)
        if score:
            matches.append((score, index, message))
    chosen = sorted(sorted(matches, reverse=True)[:limit], key=lambda item: item[1])
    if not chosen:
        return "No matching earlier messages in this thread."
    excerpts = []
    for _, index, message in chosen:
        content = message["content"]
        positions = [content.casefold().find(word) for word in words]
        first = min((position for position in positions if position >= 0), default=0)
        start = max(0, first - 120)
        excerpt = content[start:start + 600]
        if start:
            excerpt = "…" + excerpt
        if start + 600 < len(content):
            excerpt += "…"
        label = "User" if message["role"] == "user" else "Wallace (unverified)"
        excerpts.append(f"Message {index + 1} — {label}: {excerpt}")
    return "Current thread only; verify earlier assistant claims.\n" + "\n\n".join(excerpts)

