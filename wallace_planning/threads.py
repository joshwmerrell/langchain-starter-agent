"""Durable, isolated conversation threads for Wallace."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fcntl


THREAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TRUST_MODES = ("no-trust", "informed-trust", "yes-trust")
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 5
MAX_APPROVAL_TIMEOUT_SECONDS = 300
DEV_THREAD_ID = "wallace_dev"


@contextmanager
def thread_file_lock(path: Path):
    """Serialize read/modify/write transactions across Wallace processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_trust_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in TRUST_MODES:
        raise ValueError("trust mode must be no-trust, informed-trust, or yes-trust")
    return mode


def validate_approval_timeout(value: int | str) -> int:
    """Keep unattended auto-approval delays explicit and within a useful range."""
    if isinstance(value, bool):
        raise ValueError("informed-trust timeout must be 1-300 whole seconds")
    try:
        seconds = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("informed-trust timeout must be 1-300 whole seconds") from error
    if str(value).strip() != str(seconds) or not 1 <= seconds <= MAX_APPROVAL_TIMEOUT_SECONDS:
        raise ValueError("informed-trust timeout must be 1-300 whole seconds")
    return seconds


def parse_new_thread_arguments(arguments: list[str]) -> tuple[str | None, str]:
    """Parse a title and an optional --trust MODE without swallowing typos."""
    title_parts: list[str] = []
    trust_mode = "no-trust"
    found_mode = False
    index = 0
    while index < len(arguments):
        part = arguments[index]
        if part == "--trust":
            if found_mode or index + 1 >= len(arguments):
                raise ValueError("use /new [--trust MODE] [title]")
            trust_mode = validate_trust_mode(arguments[index + 1])
            found_mode = True
            index += 2
        elif part.startswith("--trust="):
            if found_mode:
                raise ValueError("specify --trust only once")
            trust_mode = validate_trust_mode(part.partition("=")[2])
            found_mode = True
            index += 1
        elif part.startswith("--"):
            raise ValueError(f"unknown /new option: {part}")
        else:
            title_parts.append(part)
            index += 1
    return " ".join(title_parts) or None, trust_mode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ThreadRecord:
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    trust_mode: str = "no-trust"
    approval_timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS

    def to_dict(self) -> dict[str, str | int]:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "trust_mode": self.trust_mode,
            "approval_timeout_seconds": self.approval_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ThreadRecord":
        if not isinstance(value, dict):
            raise ValueError("thread record must be an object")
        thread_id = str(value.get("thread_id", ""))
        if not THREAD_ID_RE.fullmatch(thread_id):
            raise ValueError("invalid thread identifier")
        saved_mode = value.get("trust_mode", "no-trust")
        trust_mode = saved_mode if saved_mode in TRUST_MODES else "no-trust"
        try:
            approval_timeout_seconds = validate_approval_timeout(
                value.get("approval_timeout_seconds", DEFAULT_APPROVAL_TIMEOUT_SECONDS)
            )
        except ValueError:
            approval_timeout_seconds = DEFAULT_APPROVAL_TIMEOUT_SECONDS
        return cls(
            thread_id=thread_id,
            title=str(value.get("title") or thread_id),
            created_at=str(value.get("created_at") or _now()),
            updated_at=str(value.get("updated_at") or _now()),
            trust_mode=trust_mode,
            approval_timeout_seconds=approval_timeout_seconds,
        )


class ConversationThreadManager:
    """Create, select, rename, and persist independent conversation threads."""

    def __init__(self, project_root: Path, *, scope: str = "legacy") -> None:
        if scope not in {"legacy", "workspace", "development"}:
            raise ValueError("unknown conversation scope")
        self.scope = scope
        self.project_root = project_root.resolve()
        self.dev_root = self.project_root / ".wallace" / "conversations"
        self.root = self.project_root / ".wallace" / (
            "conversations" if scope == "legacy" else "workspace_conversations"
        )
        self.index_path = self.root / "index.json"
        self.active_path = (
            self.dev_root / ".active-dev-thread.json" if scope == "development"
            else self.root / ".active-thread.json"
        )
        self.legacy_path = self.project_root / ".wallace" / "conversation.json"
        self.legacy_pending_path = self.project_root / ".wallace" / "pending_turn.json"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, stat.S_IRWXU)
        except OSError:
            pass
        if scope != "legacy":
            self._import_legacy_threads()

    def _read_index(self) -> list[ThreadRecord]:
        records = self._read_index_at(self.index_path)
        if self.scope != "legacy":
            # Older builds could write the reserved development ID into the
            # ordinary index. It must never be selectable from that store.
            records = [record for record in records if record.thread_id != DEV_THREAD_ID]
        return records

    @staticmethod
    def _read_index_at(path: Path) -> list[ThreadRecord]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
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
        self._write_index_at(self.index_path, records)

    @classmethod
    def _write_index_at(cls, path: Path, records: list[ThreadRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._atomic_write(
            path,
            json.dumps({"threads": [record.to_dict() for record in records]}, indent=2) + "\n",
        )

    def _import_legacy_threads(self) -> None:
        """Copy ordinary old dev threads into the common store without deleting originals."""
        source_index = self.dev_root / "index.json"
        if not source_index.exists():
            return
        marker = self.root / ".legacy-imported.json"
        with thread_file_lock(self.root / ".index.lock"):
            try:
                mapping = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                mapping = {}
            if not isinstance(mapping, dict):
                mapping = {}
            raw_records = self._read_index_at(self.index_path)
            records = [record for record in raw_records if record.thread_id != DEV_THREAD_ID]
            known = {record.thread_id for record in records}
            changed = len(records) != len(raw_records)
            for old in self._read_index_at(source_index):
                if old.thread_id == DEV_THREAD_ID or old.thread_id in mapping:
                    continue
                candidate = old.thread_id
                if candidate in known or (self.root / f"{candidate}.json").exists():
                    candidate = ("legacy_" + old.thread_id)[:64]
                while candidate in known or (self.root / f"{candidate}.json").exists():
                    candidate = f"thread_{uuid.uuid4().hex[:12]}"
                title = old.title + " (previous development)" if candidate != old.thread_id else old.title
                for suffix in (".json", ".pending.json"):
                    source = self.dev_root / f"{old.thread_id}{suffix}"
                    destination = self.root / f"{candidate}{suffix}"
                    if source.is_file() and not source.is_symlink() and not destination.exists():
                        shutil.copy2(source, destination)
                        os.chmod(destination, stat.S_IRUSR | stat.S_IWUSR)
                records.append(ThreadRecord(
                    candidate, title, old.created_at, old.updated_at, old.trust_mode,
                    old.approval_timeout_seconds,
                ))
                known.add(candidate)
                mapping[old.thread_id] = candidate
                changed = True
            if changed:
                self._write_index(records)
                self._atomic_write(marker, json.dumps(mapping, indent=2) + "\n")

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
        return next((record for record in self.list_threads() if record.thread_id == thread_id), None)

    def resolve(self, reference: str) -> ThreadRecord:
        """Resolve an existing thread by ID or exact title without creating one."""
        reference = reference.strip()
        if not reference:
            raise ValueError("thread ID or title cannot be empty")
        records = self.list_threads()
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
        if thread_id == DEV_THREAD_ID and self.scope != "development":
            raise ValueError("the development thread is only available in development mode")
        index = self.dev_root / "index.json" if thread_id == DEV_THREAD_ID else self.index_path
        with thread_file_lock(index.parent / ".index.lock"):
            records = self._read_index_at(index)
            existing = next((record for record in records if record.thread_id == thread_id), None)
            if existing is not None:
                return existing
            default_title = "Wallace development" if thread_id == DEV_THREAD_ID else (
                "Main" if thread_id == "main" else thread_id
            )
            record = ThreadRecord(thread_id, title or default_title, _now(), _now())
            records.append(record)
            self._write_index_at(index, records)
            if self.scope == "legacy":
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
        records = self._read_index()
        if self.scope == "development":
            dev = next((record for record in self._read_index_at(self.dev_root / "index.json")
                        if record.thread_id == DEV_THREAD_ID), None)
            if dev is not None:
                records.append(dev)
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def active_thread_id(self) -> str:
        """Return the last selected, still-existing thread, or a safe fallback."""
        records = self.list_threads()
        preferred = DEV_THREAD_ID if self.scope == "development" else "main"
        fallback = next((record.thread_id for record in records if record.thread_id == preferred), None)
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

    def create(
        self, title: str | None = None, thread_id: str | None = None,
        trust_mode: str = "no-trust",
    ) -> ThreadRecord:
        trust_mode = validate_trust_mode(trust_mode)
        if thread_id is None:
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        thread_id = self.validate_id(thread_id)
        if thread_id == DEV_THREAD_ID:
            raise ValueError("the permanent development thread is reserved")
        with thread_file_lock(self.root / ".index.lock"):
            records = self._read_index()
            if any(record.thread_id == thread_id for record in records):
                raise ValueError(f"thread already exists: {thread_id}")
            record = ThreadRecord(thread_id, (title or thread_id).strip()[:120], _now(), _now(), trust_mode)
            records.append(record)
            self._write_index(records)
            return record

    def set_trust_mode(
        self, thread_id: str, trust_mode: str, approval_timeout_seconds: int | str | None = None,
    ) -> ThreadRecord:
        """Only the host's thread commands call this; the model has no tool for it."""
        thread_id = self.validate_id(thread_id)
        trust_mode = validate_trust_mode(trust_mode)
        if approval_timeout_seconds is not None and trust_mode != "informed-trust":
            raise ValueError("a timeout can only be set for informed-trust")
        timeout = validate_approval_timeout(
            approval_timeout_seconds if approval_timeout_seconds is not None
            else DEFAULT_APPROVAL_TIMEOUT_SECONDS
        )
        index = self.dev_root / "index.json" if thread_id == DEV_THREAD_ID and self.scope == "development" else self.index_path
        with thread_file_lock(index.parent / ".index.lock"):
            records = self._read_index_at(index)
            for record in records:
                if record.thread_id == thread_id:
                    record.trust_mode = trust_mode
                    if trust_mode == "informed-trust":
                        record.approval_timeout_seconds = timeout
                    record.updated_at = _now()
                    self._write_index_at(index, records)
                    return record
        raise ValueError(f"unknown thread: {thread_id}")

    def rename(self, thread_id: str, title: str) -> ThreadRecord:
        thread_id = self.validate_id(thread_id)
        if thread_id == DEV_THREAD_ID:
            raise ValueError("the permanent development thread cannot be renamed")
        title = title.strip()
        if not title:
            raise ValueError("thread title cannot be empty")
        with thread_file_lock(self.root / ".index.lock"):
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
        if thread_id == DEV_THREAD_ID:
            raise ValueError("the permanent development thread cannot be removed")
        with thread_file_lock(self.root / ".index.lock"):
            records = self._read_index()
            removed = next((record for record in records if record.thread_id == thread_id), None)
            if removed is None:
                raise ValueError(f"unknown thread: {thread_id}")
            if len(records) == 1 and self.scope != "development":
                raise ValueError("cannot remove the last conversation thread")
            removing_active = self.active_thread_id() == thread_id
            remaining = [record for record in records if record.thread_id != thread_id]
            self._write_index(remaining)
        if removing_active:
            self.select_thread(next(
                (record.thread_id for record in self.list_threads() if record.thread_id == "main"),
                DEV_THREAD_ID if self.scope == "development" else remaining[0].thread_id,
            ))
        self.conversation_path(thread_id).unlink(missing_ok=True)
        self.pending_path(thread_id).unlink(missing_ok=True)
        return removed

    def touch(self, thread_id: str) -> None:
        thread_id = self.validate_id(thread_id)
        index = self.dev_root / "index.json" if thread_id == DEV_THREAD_ID and self.scope == "development" else self.index_path
        with thread_file_lock(index.parent / ".index.lock"):
            records = self._read_index_at(index)
            for record in records:
                if record.thread_id == thread_id:
                    record.updated_at = _now()
                    self._write_index_at(index, records)
                    return

    def conversation_path(self, thread_id: str) -> Path:
        thread_id = self.validate_id(thread_id)
        if thread_id == DEV_THREAD_ID and self.scope != "development":
            raise ValueError("the development thread is only available in development mode")
        root = self.dev_root if thread_id == DEV_THREAD_ID else self.root
        return root / f"{thread_id}.json"

    def pending_path(self, thread_id: str) -> Path:
        thread_id = self.validate_id(thread_id)
        if thread_id == DEV_THREAD_ID and self.scope != "development":
            raise ValueError("the development thread is only available in development mode")
        root = self.dev_root if thread_id == DEV_THREAD_ID else self.root
        return root / f"{thread_id}.pending.json"

    def store(self, thread_id: str):
        """Return a ConversationStore without importing main at module load time."""
        self.ensure(thread_id)
        from main import ConversationStore

        return ConversationStore(self.conversation_path(thread_id), self.pending_path(thread_id))

