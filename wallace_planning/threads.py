"""Durable, isolated conversation threads for Wallace."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


THREAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ThreadRecord:
    thread_id: str
    title: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ThreadRecord":
        if not isinstance(value, dict):
            raise ValueError("thread record must be an object")
        thread_id = str(value.get("thread_id", ""))
        if not THREAD_ID_RE.fullmatch(thread_id):
            raise ValueError("invalid thread identifier")
        return cls(
            thread_id=thread_id,
            title=str(value.get("title") or thread_id),
            created_at=str(value.get("created_at") or _now()),
            updated_at=str(value.get("updated_at") or _now()),
        )


class ConversationThreadManager:
    """Create, select, rename, and persist independent conversation threads."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.root = self.project_root / ".wallace" / "conversations"
        self.index_path = self.root / "index.json"
        self.active_path = self.root / ".active-thread.json"
        self.legacy_path = self.project_root / ".wallace" / "conversation.json"
        self.legacy_pending_path = self.project_root / ".wallace" / "pending_turn.json"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, stat.S_IRWXU)
        except OSError:
            pass

    def _read_index(self) -> list[ThreadRecord]:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        values = raw.get("threads", []) if isinstance(raw, dict) else []
        records: list[ThreadRecord] = []
        for value in values:
            try:
                records.append(ThreadRecord.from_dict(value))
            except (TypeError, ValueError):
                continue
        return records

    def _write_index(self, records: list[ThreadRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            self.index_path,
            json.dumps({"threads": [record.to_dict() for record in records]}, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
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

    @staticmethod
    def validate_id(thread_id: str) -> str:
        thread_id = thread_id.strip()
        if not THREAD_ID_RE.fullmatch(thread_id):
            raise ValueError("thread IDs may contain letters, numbers, '-' and '_'")
        return thread_id

    def _find(self, thread_id: str) -> ThreadRecord | None:
        return next((record for record in self._read_index() if record.thread_id == thread_id), None)

    def resolve(self, reference: str) -> ThreadRecord:
        """Resolve an existing thread by ID or exact title without creating one."""
        reference = reference.strip()
        if not reference:
            raise ValueError("thread ID or title cannot be empty")
        records = self._read_index()
        by_id = next((record for record in records if record.thread_id == reference), None)
        if by_id is not None:
            return by_id
        matches = [record for record in records if record.title.casefold() == reference.casefold()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple threads have the title: {reference}")
        raise ValueError(f"unknown thread ID or title: {reference}")

    def ensure(self, thread_id: str = "main", title: str | None = None) -> ThreadRecord:
        thread_id = self.validate_id(thread_id)
        records = self._read_index()
        existing = next((record for record in records if record.thread_id == thread_id), None)
        if existing is not None:
            return existing
        record = ThreadRecord(thread_id, title or ("Main" if thread_id == "main" else thread_id), _now(), _now())
        records.append(record)
        self._write_index(records)
        self._migrate_legacy_to_main(record)
        return record

    def _migrate_legacy_to_main(self, record: ThreadRecord) -> None:
        if record.thread_id != "main":
            return
        target = self.conversation_path("main")
        if target.exists() or not self.legacy_path.exists():
            return
        try:
            raw = self.legacy_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, list) or (
                isinstance(parsed, dict) and isinstance(parsed.get("messages"), list)
            ):
                self._atomic_write(target, raw)
            if self.legacy_pending_path.exists():
                self._atomic_write(self.pending_path("main"), self.legacy_pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

    def list_threads(self) -> list[ThreadRecord]:
        return sorted(self._read_index(), key=lambda record: record.updated_at, reverse=True)

    def active_thread_id(self) -> str:
        """Return the last selected, still-existing thread, or a safe fallback."""
        records = self.list_threads()
        fallback = next((record.thread_id for record in records if record.thread_id == "main"), None)
        fallback = fallback or (records[0].thread_id if records else "main")
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback
        selected = data.get("thread_id") if isinstance(data, dict) else None
        return selected if isinstance(selected, str) and any(
            record.thread_id == selected for record in records
        ) else fallback

    def select_thread(self, thread_id: str) -> None:
        """Remember an existing thread across normal starts and hot reloads."""
        thread_id = self.validate_id(thread_id)
        if self._find(thread_id) is None:
            raise ValueError(f"unknown thread: {thread_id}")
        self._atomic_write(self.active_path, json.dumps({"thread_id": thread_id}) + "\n")

    def create(self, title: str | None = None, thread_id: str | None = None) -> ThreadRecord:
        records = self._read_index()
        if thread_id is None:
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        thread_id = self.validate_id(thread_id)
        if any(record.thread_id == thread_id for record in records):
            raise ValueError(f"thread already exists: {thread_id}")
        record = ThreadRecord(thread_id, (title or thread_id).strip()[:120], _now(), _now())
        records.append(record)
        self._write_index(records)
        return record

    def rename(self, thread_id: str, title: str) -> ThreadRecord:
        thread_id = self.validate_id(thread_id)
        title = title.strip()
        if not title:
            raise ValueError("thread title cannot be empty")
        records = self._read_index()
        for record in records:
            if record.thread_id == thread_id:
                record.title = title[:120]
                record.updated_at = _now()
                self._write_index(records)
                return record
        raise ValueError(f"unknown thread: {thread_id}")

    def remove(self, reference: str) -> ThreadRecord:
        """Remove a thread by ID or title and its persisted conversation files.

        The last remaining thread is kept so Wallace always has a valid
        conversation target.
        """
        thread_id = self.resolve(reference).thread_id
        records = self._read_index()
        removed = next((record for record in records if record.thread_id == thread_id), None)
        if removed is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if len(records) == 1:
            raise ValueError("cannot remove the last conversation thread")

        removing_active = self.active_thread_id() == thread_id
        remaining = [record for record in records if record.thread_id != thread_id]
        self._write_index(remaining)
        if removing_active:
            self.select_thread(next(
                (record.thread_id for record in remaining if record.thread_id == "main"),
                remaining[0].thread_id,
            ))
        self.conversation_path(thread_id).unlink(missing_ok=True)
        self.pending_path(thread_id).unlink(missing_ok=True)
        return removed

    def touch(self, thread_id: str) -> None:
        thread_id = self.validate_id(thread_id)
        records = self._read_index()
        for record in records:
            if record.thread_id == thread_id:
                record.updated_at = _now()
                self._write_index(records)
                return

    def conversation_path(self, thread_id: str) -> Path:
        return self.root / f"{self.validate_id(thread_id)}.json"

    def pending_path(self, thread_id: str) -> Path:
        return self.root / f"{self.validate_id(thread_id)}.pending.json"

    def store(self, thread_id: str):
        """Return a ConversationStore without importing main at module load time."""
        self.ensure(thread_id)
        from main import ConversationStore

        return ConversationStore(self.conversation_path(thread_id), self.pending_path(thread_id))

