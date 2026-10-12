"""Request parsing and compact prompt context for every Wallace request."""

from __future__ import annotations

import re
from pathlib import Path

from .generators import create_plan, list_plans, save_plan
from .models import PlanSpec, PlanStep, SourceRef, bounded
from .quality import response_quality_error


_PATH_RE = re.compile(r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_+-]+)")
_INTENTS = (
    ("implement", ("implement", "build", "add", "create", "develop")),
    ("fix", ("fix", "repair", "bug", "debug")),
    ("review", ("review", "inspect", "audit")),
    ("refactor", ("refactor", "restructure", "clean up")),
    ("test", ("test", "verify", "validate")),
    ("research", ("research", "look up", "compare")),
)
_CONTINUE_RE = re.compile(r"^(?:please\s+)?(?:continue|resume|proceed|finish)(?:\b|$)", re.I)
_EXPLICIT_RESUME_RE = re.compile(
    r"\b(?:resume|continue|proceed|finish)\s+(?:(?:with|on)\s+)?"
    r"(?:the\s+)?(?:(?:latest|current|active|existing|previous)\s+){1,3}"
    r"(?:plan|task|work|implementation)\b",
    re.I,
)
_FEATURE_LIST_RE = re.compile(
    r"\b(?:pipeline|features?|requirements?|capabilities|tasks?)\s*:\s*([^\n.]+)",
    re.I,
)


def _requests_continuation(request: str) -> bool:
    if _CONTINUE_RE.match(request):
        return True
    for match in _EXPLICIT_RESUME_RE.finditer(request):
        before = request[max(0, match.start() - 24):match.start()]
        if not re.search(r"\b(?:do not|don't|never|without)\s*$", before, re.I):
            return True
    return False


def _feature_requirements(request: str) -> list[str]:
    """Extract an explicit, comma-separated scope list without inventing features."""
    match = _FEATURE_LIST_RE.search(request)
    if not match:
        return []
    items = [
        re.sub(r"^(?:and|or)\s+", "", item.strip(), flags=re.I).strip()
        for item in match.group(1).split(",")
    ]
    items = [item for item in items if 3 <= len(item) <= 100]
    return items if 2 <= len(items) <= 8 else []


def _implementation_steps(request: str, sources: list[SourceRef]) -> list[PlanStep]:
    features = _feature_requirements(request)
    implementation = [
        PlanStep(
            "",
            item[0].upper() + item[1:] if re.match(r"^(?:implement|add|build|fix)\b", item, re.I)
            else f"Implement {item}",
            f"Implement and verify {item} within the scope of the original request.",
            sources=sources,
            expected_output=f"Working {item}, or a precise blocker and remaining work.",
        )
        for item in features
    ] or [
        PlanStep("", "Implement the requested work", "Apply the smallest coherent change using the existing approved tool flow.", sources=sources, expected_output="The requested behavior or artifact is implemented."),
    ]
    steps = [
        PlanStep("", "Inspect relevant context", "Read referenced files and inspect the current project state.", sources=sources, expected_output="A bounded understanding of the affected files."),
        *implementation,
        PlanStep("", "Validate the result", "Run the narrowest relevant syntax or test checks and record their bounded output.", expected_output="Validation results and any recoverable failure."),
        PlanStep("", "Summarize and hand off", "Explain what changed, what was verified, and any remaining work.", expected_output="A concise user-facing summary."),
    ]
    for number, step in enumerate(steps, 1):
        step.id = f"step-{number:02d}"
    return steps


def _upgrade_generic_plan(project_root: Path, plan: PlanSpec) -> PlanSpec:
    """Give an unedited legacy plan its explicit feature checklist on resume."""
    if (
        plan.status != "active"
        or plan.intent not in {"implement", "fix", "refactor"}
        or len(plan.steps) != 4
        or plan.steps[1].title != "Implement the requested work"
        or any(item.get("ok") and item.get("name") in {"write_safe_file", "replace_in_file"} for item in plan.tool_history)
        or len(_feature_requirements(plan.request)) < 2
    ):
        return plan
    replacement = _implementation_steps(plan.request, plan.sources)
    plan.steps = [plan.steps[0], *replacement[1:-2], *plan.steps[-2:]]
    for number, step in enumerate(plan.steps, 1):
        step.id = f"step-{number:02d}"
    plan.refresh_status()
    save_plan(project_root, plan)
    return plan


def _intent(request: str) -> str:
    lowered = request.lower()
    for name, words in _INTENTS:
        if any(word in lowered for word in words):
            return name
    return "general task"


def _safe_sources(project_root: Path, request: str) -> list[SourceRef]:
    found: list[SourceRef] = []
    seen: set[str] = set()
    for match in _PATH_RE.findall(request):
        candidate = Path(match)
        if candidate.is_absolute() or ".." in candidate.parts or ".wallace" in candidate.parts:
            continue
        resolved = (project_root / candidate).resolve()
        if not resolved.is_relative_to(project_root.resolve()):
            continue
        if any(part.startswith(".") for part in candidate.parts) or candidate.name in {".env", "conversation.json"}:
            continue
        relative = candidate.as_posix()
        if relative not in seen:
            seen.add(relative)
            found.append(SourceRef(relative))
    return found


def parse_request(project_root: Path, request: str, *, session_id: str | None = None) -> PlanSpec:
    sources = _safe_sources(project_root, request)
    intent = _intent(request)
    constraints: list[str] = []
    lowered = request.lower()
    if "without" in lowered or "do not" in lowered or "don't" in lowered:
        constraints.append("Respect the exclusions and limitations stated in the request.")
    if sources:
        constraints.append("Inspect referenced files before modifying them.")
    if intent in {"implement", "fix", "refactor"}:
        steps = _implementation_steps(request, sources)
    elif intent in {"review", "research", "test"}:
        steps = [
            PlanStep("step-01", "Inspect relevant context", "Read the relevant files or run the relevant checks.", sources=sources),
            PlanStep("step-02", "Summarize and hand off", "Answer with findings and any limitations."),
        ]
    else:
        steps = [PlanStep("step-01", "Respond to the request", "Answer clearly and state any limitation.")]
    return create_plan(
        project_root,
        request,
        intent=intent,
        constraints=constraints,
        sources=sources,
        steps=steps,
        expected_outputs=["Implemented change", "Validation result", "Concise summary"],
        session_id=session_id,
    )


def planning_context(plan: PlanSpec, previous_context: str = "") -> str:
    """Build verified plan context; legacy previous_context is intentionally ignored."""
    completed = sum(step.status == "complete" for step in plan.steps)
    pending = plan.next_pending()
    next_text = f"Next step: {pending.id} — {pending.title}." if pending else "All recorded steps are complete."
    checkpoint_lines = []
    for step in plan.steps:
        if step.status == "complete" and step.result:
            checkpoint_lines.append(f"- {step.id}: {bounded(step.result, 300)}")
    checkpoint = "\n".join(checkpoint_lines[-4:]) or "- No completed checkpoints yet."
    sources = ", ".join(f"`{source.path}`" for source in plan.sources[:8]) or "none recorded"
    tool_section = "\n".join(
        f"- {'OK' if item['ok'] else 'FAILED'}: {item['name']} "
        f"{bounded(item['target'], 100)}: {bounded(item['outcome'], 40)}"
        for item in plan.tool_history[-8:]
    ) or "- None recorded."
    remaining = "\n".join(
        f"- {step.id}: {bounded(step.title, 100)}"
        for step in plan.steps if step.status != "complete"
    ) or "- None."
    error_section = f"Last error: {bounded(plan.last_error, 300)}\n" if plan.last_error else ""
    if plan.status == "active" and response_quality_error(plan.summary):
        error_section += (
            "Host note: The previous assistant reply was repetitive and did not "
            "complete the pending step. Do not repeat its preamble; proceed with "
            "the next concrete action.\n"
        )
    return (
        "Use this durable execution plan as the compact working memory for the "
        "current request. Do not repeat completed work.\n"
        f"Plan: {plan.plan_id} ({completed}/{len(plan.steps)} complete)\n"
        f"Original request: {bounded(plan.request, 600)}\n"
        f"Intent: {plan.intent}\n"
        f"{next_text}\n"
        f"Remaining steps:\n{remaining}\n"
        f"Sources: {sources}\n"
        "Completed checkpoints:\n"
        f"{checkpoint}\n"
        "Recent tool calls (failed targets are not validated; do not repeat "
        "unchanged failed commands):\n"
        f"{tool_section}\n"
        f"{error_section}"
        "The plan state above is supplied by the host. Do not try to read private "
        "Wallace state with project file tools.\n"
    )


def latest_completed_context(
    project_root: Path,
    *,
    exclude_plan_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Return one bounded prior plan summary for short follow-up requests."""
    candidates = [
        plan
        for plan in list_plans(project_root)
        if plan.status == "complete"
        and plan.summary
        and plan.plan_id != exclude_plan_id
        and (session_id is None or plan.session_id == session_id)
    ]
    previous = max(candidates, key=lambda plan: plan.updated_at, default=None)
    return previous.summary if previous is not None else ""


def resume_or_create(project_root: Path, request: str, *, session_id: str | None = None) -> PlanSpec:
    normalized = request.strip()
    session_plans = list_plans(project_root)
    if session_id:
        session_plans = [plan for plan in session_plans if plan.session_id == session_id]
    candidates = [plan for plan in session_plans if plan.status == "active"]
    # A retry after a restart resumes the unfinished plan for the same request.
    # A different request gets its own plan, even within the same session.
    matching = [plan for plan in candidates if plan.request == normalized]
    if matching:
        return _upgrade_generic_plan(project_root, max(matching, key=lambda plan: plan.updated_at))
    if _requests_continuation(normalized) and session_plans:
        latest = max(session_plans, key=lambda plan: plan.updated_at)
        if latest.status == "active":
            return _upgrade_generic_plan(project_root, latest)
    return parse_request(project_root, request, session_id=session_id)


def add_planning_arguments(parser) -> None:
    parser.add_argument("--session-id", help="Resume or create plans in this durable session.")
