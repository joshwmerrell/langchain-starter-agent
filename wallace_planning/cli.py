"""Request parsing and compact prompt context for every Wallace request."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .generators import create_plan, list_plans, save_plan
from .models import PlanSpec, PlanStep, SourceRef, bounded
from .quality import response_quality_error


_PATH_RE = re.compile(r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9_+-]+)")
_INTENTS = (
    ("implement", ("implement", "build", "add", "create", "develop")),
    ("fix", ("fix", "repair", "bug", "debug")),
    ("review", ("review", "inspect", "audit", "diagnose", "look at", "describe")),
    ("refactor", ("refactor", "restructure", "clean up")),
    ("test", ("test", "verify", "validate")),
    ("research", ("research", "look up", "compare")),
)
_DECISION_REQUEST_RE = re.compile(
    r"\b(?:determine|decide|recommend|prioriti[sz]e|identify)\s+"
    r"(?:what|which|whether)\b|\bwhat\s+should\b",
    re.I,
)
_DECISION_THEN_ACT_RE = re.compile(
    r"\b(?:then|afterwards|after that|go ahead and)\s+"
    r"(?:implement|build|develop|create|add|fix|change|edit)\b",
    re.I,
)
_CONTINUE_RE = re.compile(r"^(?:please\s+)?(?:continue|resume|proceed|finish)(?:\b|$)", re.I)
_AFFIRMATIVE_OFFER_RE = re.compile(
    r"^(?:yes|yeah|sure|okay|ok)(?:[,\s]+(?:please\s+)?(?:proceed|go ahead|do it))?[.!\s]*$"
    r"|^(?:please\s+)?(?:go ahead|do it)[.!\s]*$",
    re.I,
)
_ASSISTANT_OFFER_RE = re.compile(
    r"(?:Would you like me to|Shall I|Should I)\s+([^?\n]{4,350})\?\s*$",
    re.I,
)
_RETRY_RE = re.compile(r"^(?:please\s+)?(?:try\s+again|retry(?:\s+that)?)[.!?\s]*$", re.I)
_VERIFY_FOLLOWUP_RE = re.compile(
    r"^(?:please\s+)?(?:verify|check|confirm)\s+(?:its|that|the)\s+"
    r"(?:completion|result|deletion|work)\b",
    re.I,
)
_EXPLICIT_RESUME_RE = re.compile(
    r"\b(?:resume|continue|proceed|finish)\s+(?:(?:with|on)\s+)?"
    r"(?:the\s+)?(?:(?:latest|current|active|existing|previous)\s+){1,3}"
    r"(?:plan|task|work|implementation)\b",
    re.I,
)
_CODE_CHECK_RE = re.compile(
    r"\b(?:code|source|files?|website|site|app|scripts?|program|build|syntax|tools?)\b"
    r"|\b[A-Za-z0-9_.-]+\.(?:py|js|mjs|ts|tsx|html|css|sh|go|rs)\b",
    re.I,
)
_DELETION_RE = re.compile(
    r"\b(?:delete|deleting|remove|removing|erase|purge)\b.{0,120}"
    r"\b(?:files?|site|directory|folder|assets?|pages?)\b",
    re.I | re.S,
)
_CAPABILITY_QUESTION_RE = re.compile(
    r"\b(?:does|do|is|are)\b[^?\n]{0,120}\b(?:exist|supported|available)\b",
    re.I,
)
_EXPLICIT_ACTION_RE = re.compile(
    r"\b(?:please\s+)?(?:add|implement|fix|create|build|change|update)\b",
    re.I,
)
_REPORTED_OUTPUT_RE = re.compile(
    r"\b(?:here is|here's) the output\s*:|"
    r"\b(?:following|this) (?:error|output|response|log)\s*:|"
    r"\b(?:error|output) response\s*:",
    re.I,
)
_DIAGNOSE_BEFORE_CHANGE_RE = re.compile(
    r"\b(?:come (?:back )?to me|report (?:back )?to me|ask me|check with me)\b"
    r".{0,100}\bbefore\b.{0,60}\b(?:fix|fixing|implement|implementing|edit|editing|change|changing)\b",
    re.I | re.S,
)


def _requests_continuation(request: str) -> bool:
    if _CONTINUE_RE.match(request) or _RETRY_RE.fullmatch(request) or _VERIFY_FOLLOWUP_RE.match(request):
        return True
    for match in _EXPLICIT_RESUME_RE.finditer(request):
        before = request[max(0, match.start() - 24):match.start()]
        if not re.search(r"\b(?:do not|don't|never|without)\s*$", before, re.I):
            return True
    return False


def continuation_instruction(request: str, messages: list[dict[str, str]]) -> str:
    """Recover the latest detailed user correction behind a terse `continue`."""
    if not _requests_continuation(request) or len(request) > 100:
        return request
    for message in reversed(messages):
        content = message.get("content", "")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and len(content) > 100
            and _requests_continuation(content)
        ):
            return content
    return request


def accepted_offer_instruction(request: str, messages: list[dict[str, str]]) -> str:
    """Ground a terse assent in the immediately preceding assistant offer only."""
    prior = list(messages)
    if prior and prior[-1] == {"role": "user", "content": request}:
        prior.pop()
    if not _AFFIRMATIVE_OFFER_RE.fullmatch(request.strip()):
        # A failed literal tool call is not a completed turn. A subsequent
        # "continue" should recover the offer the user had just accepted.
        if not (
            len(request) <= 100
            and _requests_continuation(request)
            and len(prior) >= 3
            and prior[-1].get("role") == "assistant"
            and re.search(r"</?tool_call(?:\s|>)", prior[-1].get("content", ""), re.I)
            and prior[-2].get("role") == "user"
            and _AFFIRMATIVE_OFFER_RE.fullmatch(prior[-2].get("content", "").strip())
        ):
            return request
        prior = prior[:-2]
    if not prior or prior[-1].get("role") != "assistant":
        return request
    answer = prior[-1].get("content", "")
    if not isinstance(answer, str):
        return request
    offer = _ASSISTANT_OFFER_RE.search(answer.strip())
    if not offer:
        return request
    action = offer.group(1).strip()
    instruction = (
        f"The user accepted your immediately preceding offer to {action}. "
        "Carry out that offered next step now; do not repeat already completed work."
    )
    # This is a candidate from the previous reply, not proof of its README or
    # installation instructions. It is enough to move from discovery to setup.
    repositories = re.findall(r"https://github\.com/[\w.-]+/[\w.-]+", answer)
    if repositories:
        instruction += (
            f" Previously identified GitHub repository: {repositories[-1]}. "
            "Verify its setup instructions with an actual tool result before configuring it."
        )
    return instruction


def _no_op_retry_plan(plan: PlanSpec) -> bool:
    """Legacy one-step retry records are not the task being retried."""
    return bool(
        _RETRY_RE.fullmatch(plan.request.strip())
        and plan.intent == "general task"
        and not plan.edit_evidence.get("targets")
        and not plan.unresolved_failures
    )


def _implementation_steps(request: str, sources: list[SourceRef]) -> list[PlanStep]:
    """Track observable work without fabricating feature-level proof."""
    steps = [
        PlanStep("", "Inspect relevant context", "Read referenced files and inspect the current project state.", sources=sources, expected_output="A bounded understanding of the affected files."),
        PlanStep("", "Apply requested changes", "Make the requested changes and record affected files. The original request defines the scope.", sources=sources, expected_output="Observed edits, without inferring that each requested feature is proven."),
        PlanStep("", "Validate the result", "Run checks that cover the changed files; distinguish source checks from behavioral tests.", expected_output="Passing checks with their actual scope, or remaining failures."),
        PlanStep("", "Summarize and hand off", "Explain what changed, what was verified, and any remaining work.", expected_output="A concise user-facing summary."),
    ]
    for number, step in enumerate(steps, 1):
        step.id = f"step-{number:02d}"
    return steps


def _review_steps(sources: list[SourceRef]) -> list[PlanStep]:
    return [
        PlanStep("step-01", "Inspect relevant context", "Read the relevant files or run the relevant checks.", sources=sources),
        PlanStep("step-02", "Summarize and hand off", "Answer with findings and any limitations."),
    ]


def _upgrade_generic_plan(project_root: Path, plan: PlanSpec) -> PlanSpec:
    """Replace legacy heuristic feature counters with observed work stages."""
    if (
        plan.status != "active"
        or plan.intent not in {"implement", "fix", "refactor"}
        or len(plan.steps) < 4
        or (len(plan.steps) == 4 and plan.steps[1].title != "Implement the requested work")
    ):
        return plan
    replacement = _implementation_steps(plan.request, plan.sources)
    plan.steps = [plan.steps[0], replacement[1], *plan.steps[-2:]]
    for number, step in enumerate(plan.steps, 1):
        step.id = f"step-{number:02d}"
    plan.record_progress([])
    save_plan(project_root, plan)
    return plan


def _intent(request: str) -> str:
    # Pasted command output often contains action words (e.g. `/mcp add`).
    # Classify the user's instruction, not the help text in reported output.
    reported_output = _REPORTED_OUTPUT_RE.search(request)
    user_text = request[:reported_output.start()] if reported_output else request
    lowered = user_text.lower()
    question = _CAPABILITY_QUESTION_RE.search(user_text)
    if (question
            and not _EXPLICIT_ACTION_RE.search(user_text[:question.start()])
            and not _EXPLICIT_ACTION_RE.search(user_text[question.end():])):
        return "review"
    if _DIAGNOSE_BEFORE_CHANGE_RE.search(user_text):
        return "review"
    if _DELETION_RE.search(user_text):
        return "delete"
    # The verbs may name candidate work, not authorize that work. A roadmap
    # decision should produce a recommendation unless execution is explicit.
    if is_decision_request(user_text):
        return "review"
    if re.search(r"\b(?:tell me|what is)\b", lowered) and _CODE_CHECK_RE.search(user_text):
        return "review"
    if (
        reported_output
        and re.search(r"\b(?:error|failed|failure|unavailable|not connected)\b", user_text + reported_output.group(), re.I)
        and not _EXPLICIT_ACTION_RE.search(user_text)
    ):
        return "review"
    for name, words in _INTENTS:
        if any(re.search(rf"\b{re.escape(word)}\b(?!\s*\()", lowered) for word in words):
            return name
    return "general task"


def is_decision_request(request: str) -> bool:
    """Whether action words refer to a choice, rather than an edit request."""
    reported_output = _REPORTED_OUTPUT_RE.search(request)
    user_text = request[:reported_output.start()] if reported_output else request
    if _DIAGNOSE_BEFORE_CHANGE_RE.search(user_text):
        return True
    if (
        reported_output
        and re.search(r"\b(?:error|failed|failure|unavailable|not connected)\b", user_text + reported_output.group(), re.I)
        and not _EXPLICIT_ACTION_RE.search(user_text)
    ):
        return True
    return bool(
        _DECISION_REQUEST_RE.search(request)
        and not _DECISION_THEN_ACT_RE.search(request)
    )


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
    elif intent == "delete":
        steps = [
            PlanStep("step-01", "Inspect deletion scope", "Identify exact target and dependent files/tests before deleting."),
            PlanStep("step-02", "Delete the requested target", "Run the approved deletion command only for the target."),
            PlanStep("step-03", "Verify cleanup", "Confirm target absence, inspect Git status, and run the full project tests."),
            PlanStep("step-04", "Summarize and hand off", "Report verified results and remaining references or failures."),
        ]
    elif intent == "test" and (_CODE_CHECK_RE.search(request) or _VERIFY_FOLLOWUP_RE.match(request)):
        steps = [
            PlanStep("step-01", "Inspect relevant context", "Read the relevant files and existing tests.", sources=sources),
            PlanStep("step-02", "Run relevant checks", "Run source checks or tests and resolve failures before claiming success."),
            PlanStep("step-03", "Summarize and hand off", "Report observed results and any remaining limitations."),
        ]
    elif intent in {"review", "research", "test"}:
        steps = _review_steps(sources)
    else:
        steps = [PlanStep("step-01", "Respond to the request", "Answer clearly and state any limitation.")]
    return create_plan(
        project_root,
        request,
        intent=intent,
        constraints=constraints,
        sources=sources,
        steps=steps,
        expected_outputs=(
            ["Observed changes", "Scoped check results", "Summary and limitations"]
            if intent in {"implement", "fix", "refactor", "delete"}
            else ["Evidence-backed response", "Limitations and remaining work"]
        ),
        session_id=session_id,
    )


def planning_context(
    plan: PlanSpec, previous_context: str = "", *, latest_instruction: str = ""
) -> str:
    """Build verified plan context; legacy previous_context is intentionally ignored."""
    completed = sum(step.status == "complete" for step in plan.steps)
    pending = plan.next_pending()
    next_text = f"Next step: {pending.id} — {pending.title}." if pending else "All recorded steps are complete."
    checkpoint_lines = []
    for step in plan.steps:
        if step.status == "complete" and step.result and step is not plan.steps[-1]:
            checkpoint_lines.append(f"- {step.id}: {bounded(step.result, 200)}")
    checkpoint = "\n".join(checkpoint_lines[-4:]) or "- No completed checkpoints yet."
    sources = ", ".join(f"`{source.path}`" for source in plan.sources[:8]) or "none recorded"
    tool_section = "\n".join(
        f"- {'OK' if item['ok'] else 'FAILED'}: {item['name']} "
        f"{bounded(item['target'], 100)}: {bounded(item['outcome'], 40)}"
        + (f" ({bounded(item.get('diagnostic'), 100)})" if item.get("diagnostic") else "")
        for item in plan.tool_history[-8:]
    ) or "- None recorded."
    unchecked = plan.unchecked_targets()
    unchecked_section = (
        "Files still needing matching checks: " + ", ".join(unchecked[:8]) + "\n"
        if unchecked else ""
    )
    if plan.edit_evidence.get("unscoped_mutations"):
        unchecked_section += "A command or external tool changed unknown files; reconcile its effects before claiming completion.\n"
    verification_note = plan.verification_note()
    tool_assessment_section = (
        "Tool execution assessment: " + plan.execution_assessment() + "\n"
        if re.search(r"\btools?\b", plan.request, re.IGNORECASE) else ""
    )
    remaining = "\n".join(
        f"- {step.id}: {bounded(step.title, 100)}"
        for step in plan.steps if step.status != "complete"
    ) or "- None."
    failures_section = "".join(
        f"- {item['name']} {bounded(item['target'], 100)}: "
        f"{bounded(item.get('diagnostic') or item.get('outcome'), 100)}\n"
        for item in plan.unresolved_failures[-4:]
    )
    if failures_section:
        failures_section = "Unresolved checks (retry with corrected inputs):\n" + failures_section
    error_section = (
        f"Last error: {bounded(plan.last_error, 300)}\n"
        if plan.last_error and not plan.last_error.startswith("Wallace has not verified this request as complete.")
        else ""
    )
    instruction_section = (
        "Latest detailed user instruction (overrides the original request where they conflict):\n"
        f"{bounded(latest_instruction, 1_200)}\n"
        if latest_instruction and latest_instruction != plan.request else ""
    )
    deletion_section = ""
    if plan.intent == "delete":
        target = plan.steps[1].tool_args.get("_deletion_target") if len(plan.steps) > 1 else None
        checked_path = shlex.quote(str(target)) if target else "<relative-target>"
        deletion_section = (
            "Deletion verification: after removing the exact target, run "
            f"`test ! -e {checked_path}` with execute_command, "
            "then get_git_status and run_tests with test_path='.'. "
            "Check remaining references before claiming no trace remains.\n"
        )
    elif _DELETION_RE.search(plan.request) and any(
        item.get("ok") and item.get("name") == "execute_command"
        and re.match(r"^(?:rm|rmdir|unlink)\b", str(item.get("target", "")))
        for item in plan.tool_history
    ):
        deletion_section = (
            "A removal command already ran. Do not remove the target again. "
            "The next action is to verify absence and run `run_tests` with "
            "test_path='.'; report any real failures.\n"
        )
    if plan.status == "active" and response_quality_error(plan.summary):
        error_section += (
            "Host note: The previous assistant reply was repetitive and did not "
            "complete the pending step. Do not repeat its preamble; proceed with "
            "the next concrete action.\n"
        )
    check_limits = f"Check limits: {verification_note}\n" if verification_note else ""
    return (
        "Complete the full user request in this turn. Stages are checkpoints, "
        "not stopping points. Reuse completed work.\n"
        f"{instruction_section}"
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
        f"{unchecked_section}"
        f"{check_limits}"
        f"{tool_assessment_section}"
        f"{failures_section}"
        f"{error_section}"
        f"{deletion_section}"
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
    matching = [
        plan for plan in candidates
        if plan.request == normalized and not _no_op_retry_plan(plan)
    ]
    if matching:
        return _prepare_resumed_plan(project_root, max(matching, key=lambda plan: plan.updated_at))
    if _requests_continuation(normalized) and session_plans:
        # Older releases stored terse retries as completed one-step plans.
        # Ignore those no-op records so a later "continue" can still recover
        # the unfinished substantive task in the same thread. A later
        # verification-only response must not bury the active work plan.
        substantive = [plan for plan in session_plans if not _no_op_retry_plan(plan)]
        latest = max(substantive or session_plans, key=lambda plan: plan.updated_at)
        unfinished = [
            plan for plan in candidates
            if not _no_op_retry_plan(plan) and not _VERIFY_FOLLOWUP_RE.match(plan.request)
        ]
        if unfinished and (
            latest.status == "active" or _VERIFY_FOLLOWUP_RE.match(latest.request)
        ):
            return _prepare_resumed_plan(project_root, max(unfinished, key=lambda plan: plan.updated_at))
    return parse_request(project_root, request, session_id=session_id)


def _prepare_resumed_plan(project_root: Path, plan: PlanSpec) -> PlanSpec:
    """Drop historical tool-invocation typos, not genuine failed checks."""
    if (
        plan.status == "active"
        and plan.intent in {"implement", "fix", "refactor"}
        and _intent(plan.request) == "review"
        and not plan.edit_evidence.get("targets")
        and not any(
            item.get("ok") and item.get("name") in {"write_safe_file", "replace_in_file", "edit_file_lines"}
            for item in plan.tool_history
        )
    ):
        plan.intent = "review"
        plan.steps = _review_steps(plan.sources)
        plan.expected_outputs = ["Evidence-backed findings", "Recommendation and next step"]
        plan.summary = ""
        plan.last_error = ""
        plan.record_progress([])
        save_plan(project_root, plan)
    plan = _upgrade_generic_plan(project_root, plan)
    invalid = [
        item for item in plan.unresolved_failures
        if item.get("name") == "check_python_syntax"
        and Path(str(item.get("target", ""))).suffix != ".py"
    ]
    if invalid:
        plan.record_progress([])
        save_plan(project_root, plan)
    return plan


def add_planning_arguments(parser) -> None:
    parser.add_argument("--session-id", help="Resume or create plans in this durable session.")
