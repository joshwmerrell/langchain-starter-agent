"""Conservative retention and archival of durable plan artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .generators import _read_index, _write_index, _write_private, plans_root
from .models import PlanSpec, utc_now


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


class ArchiveManager:
    def __init__(self, project_root: Path, inactive_days: int = 30, purge_days: int = 90) -> None:
        if inactive_days < 1 or purge_days < inactive_days:
            raise ValueError("retention windows must be positive and ordered")
        self.project_root = project_root
        self.inactive_days = inactive_days
        self.purge_days = purge_days

    def cleanup(self, now: datetime | None = None) -> dict[str, int]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        root = plans_root(self.project_root)
        sessions_root = root / "sessions"
        archive_root = root / "archive"
        counts = {"archived": 0, "purged": 0, "skipped": 0}
        index = _read_index(self.project_root)
        records = [item for item in index.get("plans", []) if isinstance(item, dict)]

        for plan_path in sorted(sessions_root.glob("*/plan_*.json")) if sessions_root.exists() else []:
            try:
                plan = PlanSpec.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                counts["skipped"] += 1
                continue
            updated = _parse_timestamp(plan.updated_at)
            if plan.status != "complete" or any(step.status == "running" for step in plan.steps) or updated is None:
                counts["skipped"] += 1
                continue
            if now - updated < timedelta(days=self.inactive_days):
                continue
            destination = archive_root / plan.session_id / plan.plan_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_dir = plan_path.parent
            artifact_names = (
                plan_path.name,
                f"plan_{plan.plan_id}.md",
                f"input_{plan.plan_id}.md",
                f"sources_{plan.plan_id}",
                f"outputs_{plan.plan_id}",
            )
            destination.mkdir(parents=True, exist_ok=True)
            for name in artifact_names:
                source = source_dir / name
                if not source.exists():
                    continue
                target = destination / name
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                shutil.move(str(source), str(target))
            plan.archived_at = utc_now()
            archived_json = destination / plan_path.name
            _write_private(self.project_root, archived_json, json.dumps(plan.to_dict(), indent=2) + "\n")
            for item in records:
                if item.get("plan_id") == plan.plan_id:
                    item.update({"status": plan.status, "archived_at": plan.archived_at})
            counts["archived"] += 1

        for archive_session in sorted(archive_root.iterdir()) if archive_root.exists() else []:
            if not archive_session.is_dir():
                continue
            for archived_plan in sorted(archive_session.iterdir()):
                if not archived_plan.is_dir():
                    continue
                plan_json = next(iter(archived_plan.glob("plan_*.json")), None)
                if plan_json is None:
                    continue
                try:
                    plan = PlanSpec.from_dict(json.loads(plan_json.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    counts["skipped"] += 1
                    continue
                archived_at = _parse_timestamp(plan.archived_at)
                if archived_at is None or now - archived_at < timedelta(days=self.purge_days):
                    continue
                shutil.rmtree(archived_plan)
                for item in records:
                    if item.get("plan_id") == plan.plan_id:
                        item.update({"deleted_at": utc_now()})
                counts["purged"] += 1

        index["plans"] = records
        _write_index(self.project_root, index)
        return counts

    def run_at_session_boundary(self) -> dict[str, int]:
        """Run explicit cleanup at startup/shutdown; never delete in-flight work."""
        return self.cleanup()
