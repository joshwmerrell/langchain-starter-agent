"""Wallace's context-aware planning layer."""

from .archive import ArchiveManager
from .cli import (
    accepted_offer_instruction, add_planning_arguments, continuation_instruction, latest_completed_context,
    parse_request, planning_context, resume_or_create,
)
from .executors import PlanExecutor
from .generators import active_plan, create_plan, list_plans, load_plan, save_plan
from .models import PlanSpec, PlanStep, SourceRef
from .quality import response_quality_error
from .threads import (
    ConversationThreadManager, DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ThreadRecord, TRUST_MODES, parse_new_thread_arguments,
    validate_approval_timeout, validate_trust_mode,
)

__all__ = [
    "ArchiveManager",
    "ConversationThreadManager",
    "DEFAULT_APPROVAL_TIMEOUT_SECONDS",
    "PlanExecutor",
    "PlanSpec",
    "PlanStep",
    "SourceRef",
    "ThreadRecord",
    "TRUST_MODES",
    "active_plan",
    "accepted_offer_instruction",
    "add_planning_arguments",
    "create_plan",
    "continuation_instruction",
    "list_plans",
    "latest_completed_context",
    "load_plan",
    "parse_request",
    "planning_context",
    "parse_new_thread_arguments",
    "response_quality_error",
    "resume_or_create",
    "save_plan",
    "validate_trust_mode",
    "validate_approval_timeout",
]
