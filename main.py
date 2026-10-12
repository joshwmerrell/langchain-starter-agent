import asyncio
import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from ollama import ResponseError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

try:
    # Built into LangChain (beta); install with: uv add "langchain[mcp]"
    from langchain.mcp import MCPAdapter
    _MCP_IMPORT_ERROR = None
except ImportError as _error:
    MCPAdapter = None
    _MCP_IMPORT_ERROR = _error

from tools import TOOLS, TOOLS_REQUIRING_APPROVAL, preview_write
from wallace_planning import (
    ArchiveManager,
    ConversationThreadManager,
    planning_context,
    response_quality_error,
    resume_or_create,
    save_plan,
)

load_dotenv()

console = Console()

CONVERSATION_PATH = Path(__file__).resolve().parent / ".wallace" / "conversation.json"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer configuration value with a safe fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = _env_int("WALLACE_NUM_CTX", 32768)
OLLAMA_NUM_PREDICT = _env_int("WALLACE_NUM_PREDICT", 4096)
OLLAMA_KEEP_ALIVE = os.getenv("WALLACE_KEEP_ALIVE", "15m")

# Keep restored text history bounded for the visible TUI log. The model does
# not replay this full history; durable plan checkpoints carry forward only
# the useful context between requests.
MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_CHARS = 60_000

MCP_URL = "https://docs.langchain.com/mcp"
MCP_TIMEOUT_SECONDS = 15

# Each tool call costs two graph steps (model + tool), so this allows roughly
# 25 tool calls in one turn.
RECURSION_LIMIT = 50

# A malformed tool call is a model-output problem. Keep retries bounded and
# resume the checkpointed turn rather than replaying the original request.
MALFORMED_TOOL_CALL_RETRY_PROMPT = (
    "Continue from the completed tool results already in this turn; do not "
    "repeat a successful call. The last tool arguments were malformed JSON. "
    "If a tool is still needed, make one small valid call. Use replace_in_file "
    "for small edits and write_safe_file only for short new files."
)

EMPTY_RESPONSE_RETRY_PROMPT = (
    "Your turn has no final text. Continue the user's unfinished task from the "
    "completed tool results; do not repeat successful reads. Make the next "
    "small tool call if needed. When finished or blocked, give one concise "
    "plain-text response stating only what was verified and what remains."
)

# Tools that pause for the user's approval before they run (defined in tools.py).
APPROVAL_REQUIRED = {
    name: {"allowed_decisions": ["approve", "reject"]}
    for name in sorted(TOOLS_REQUIRING_APPROVAL)
}

# Tool names and descriptions are already sent to the model with every request,
# so this prompt only covers behavior, not a catalog of tools.
SYSTEM_PROMPT = """
You are Wallace, a capable, concise, and professional all-around assistant.
Help the user with creative, practical, and productivity work as well as
software and computing tasks. Be polite, direct, and adaptable to the user's
needs.

Communicate in the user's language when you can. Work with programming and
other computing languages as the task requires. Be transparent about
uncertainty; do not claim fluency or expertise you cannot demonstrate.

The long-term goal is for Wallace to support web research, authorized cloud
workspaces such as Google Workspace, and voice conversations through the
computer's microphone and speaker. These capabilities are only available when
appropriate tools or integrations are actually provided in this session. Never
claim to browse the general web, access a cloud workspace, hear audio, or speak
aloud unless a corresponding available tool has done so. For now, use the
available project and LangChain documentation tools and clearly explain any
capability or access limitation.

CRITICAL INSTRUCTION:
Whenever you invoke a tool, wait for its result before deciding the next step.
Use further tools when the request still requires implementation or validation;
do not stop after the first inspection result. Once the work is finished or
genuinely blocked, write a final text response summarizing what happened and
what remains. Never end your turn with an empty text response or only a raw
tool call.

Use your tools whenever a request involves the project's files, running a
command or tests, checking Python syntax, reviewing Git changes, or looking up
LangChain documentation. Do not claim you cannot access project files when a
tool can; call the tool and use its result. If a tool fails or is unavailable,
say so clearly instead of guessing.

Do not emit internal reasoning tags such as `<think>` or `</think>`. Do not
repeat a preamble; either take the next concrete action or give a concise,
factual explanation of what blocks it.

Work efficiently:
- Explore with `get_project_tree` and `search_in_files`, then read only what
  you need. For large files, use `get_code_symbols` to find line numbers and
  `read_safe_file` with start_line/end_line.
- For architecture questions, use `summarize_project_file`,
  `analyze_project_dependencies`, and `analyze_git_change_impact` to inspect
  imports, dependents, and likely change impact.
- Change existing files with `replace_in_file` (copy old_text exactly from the
  file). Use `write_safe_file` only for short new files, not for changes to
  existing files. Keep each edit small enough that the full tool arguments can
  fit comfortably in the model output budget.
- After editing Python, run `check_python_syntax`, and `run_tests` if the
  project has tests. File-edit tools reject Python that fails compilation;
  an `Error:` result means the file was not changed. Correct the edit before
  claiming progress, and never describe a step as complete without checking
  its result.
- `execute_command` already starts in the project root. Use relative paths;
  do not assume that `/workspace` exists or prepend `cd /workspace`.
- Before calling a tool, check its results already present in this turn and
  the recent tool calls in the plan. Do not rerun an identical successful
  command or file read unless the underlying file changed or the prior result
  was incomplete. Do not rerun an identical failed command without correcting
  its inputs or the cause of failure.

Every request has a compact durable execution plan supplied by the host. Use
its next pending step and completed checkpoints as working memory. Do not
reconstruct the full prior conversation or repeat completed work. For an
implementation request with enough requirements already supplied, perform the
next concrete pending change and verify it; do not stop at a preamble or ask
the user to restate scope already present in the plan. Ask only when a missing
choice genuinely blocks safe progress. The host supplies plan state directly;
do not try to open private plan files with project tools. Finish the current
request with a concise text summary after tool work.

`execute_command`, `write_safe_file`, and `replace_in_file` need the user's
approval before they run. If the user rejects an action, do not retry it;
briefly say what you did not do and ask how they would like to proceed.
"""


# ---------------------------------------------------------------------------
# Shared helpers (used by both this CLI and tui_agent.py)
# ---------------------------------------------------------------------------

class ConversationStore:
    """Persist completed text turns so a TUI reload can restore the context."""

    def __init__(self, path: Path = CONVERSATION_PATH, pending_path: Path | None = None) -> None:
        self.path = path
        self._pending_path = pending_path

    def load(self) -> list[dict[str, str]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []

        history: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if (
                role in {"user", "assistant"}
                and isinstance(content, str)
                and content.strip()
            ):
                history.append({"role": role, "content": content})
        return self.trim(history)

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
        history = self.trim(history)
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
                json.dump(list(history), temporary, ensure_ascii=False, indent=2)
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

    def clear(self) -> None:
        """Remove persisted conversational history, if it exists."""
        try:
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


def assistant_response_text(messages: Sequence[BaseMessage]) -> str | None:
    """Return a final assistant message, never preliminary text or tool output."""
    message_list = list(messages)
    # A model can narrate its intent before calling tools. That narration is
    # not a completion, even if no later AI text arrives.
    last_tool = max(
        (index for index, msg in enumerate(message_list) if isinstance(msg, ToolMessage)),
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
    """Factual host-side handoff when Ollama never supplies final text."""
    parts = [
        "Ollama produced no final text after one recovery attempt; "
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
        item.get("ok") and item.get("name") in {"replace_in_file", "write_safe_file"}
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
        else:
            target = ""
        output = str(message.content).strip()
        exit_match = re.match(r"Exit code:\s*(\d+)", output)
        failed = (
            getattr(message, "status", None) == "error"
            or output.startswith(("Error", "Access denied"))
            or (exit_match is not None and exit_match.group(1) != "0")
        )
        # Never persist returned file text or command output in the plan.
        outcome = (
            f"Exit code: {exit_match.group(1)}"
            if exit_match is not None
            else "failed" if failed else "ok"
        )
        activity.append({
            "id": call_id,
            "name": name,
            "target": target,
            "outcome": outcome,
            "ok": not failed,
        })
    return activity


async def malformed_retry_payload(agent, thread_config, original_prompt: str) -> dict:
    """Continue a checkpointed turn; replay the request only if nothing survived."""
    try:
        state = await agent.aget_state(thread_config)
        checkpointed = bool(state.values.get("messages"))
    except Exception:
        checkpointed = False
    content = (
        MALFORMED_TOOL_CALL_RETRY_PROMPT
        if checkpointed
        else f"{original_prompt}\n\n{MALFORMED_TOOL_CALL_RETRY_PROMPT}"
    )
    return {"messages": [{"role": "user", "content": content}]}


def malformed_tool_call_diagnostic(error: Exception) -> str:
    """Report only the tool name and parser failure, never generated arguments."""
    detail = str(error)
    tool_match = re.search(
        r'invalid tool call arguments\s+for\s+["\']([A-Za-z_][A-Za-z0-9_]*)["\']',
        detail,
        flags=re.I,
    )
    tool = f" for {tool_match.group(1)}" if tool_match else ""
    issue = (
        "unexpected end of JSON input"
        if "unexpected end of json" in detail.lower()
        else "invalid JSON"
    )
    return f"Ollama returned invalid tool-call arguments{tool} ({issue})."


def describe_error(error: Exception) -> str:
    """Turn an exception from an agent run into a user-facing message."""
    if isinstance(error, RepetitiveResponseError):
        return str(error)
    if is_malformed_tool_call_error(error):
        return (
            f"{malformed_tool_call_diagnostic(error)} Wallace stopped after one "
            "recovery attempt. The request is unfinished; continue from the "
            "active plan after checking Ollama's context and output limits."
        )
    if isinstance(error, GraphRecursionError):
        return (
            "The agent used up its step limit before finishing. Try a narrower "
            "request, or ask it to continue."
        )
    # ollama raises the builtin ConnectionError when the server isn't reachable.
    if isinstance(error, (httpx.HTTPError, ResponseError, ConnectionError)):
        return (
            "The local Ollama request failed. Check that Ollama is running and "
            f"the configured model is available, then try again. Details: {error}"
        )
    return f"Unexpected error ({type(error).__name__}): {error}"


def is_malformed_tool_call_error(error: Exception) -> bool:
    """Whether Ollama rejected incomplete JSON for a generated tool call."""
    detail = str(error).lower()
    return (
        "invalid tool call arguments" in detail
        and ("unexpected end of json" in detail or "invalid json" in detail)
    )


def extract_action_requests(interrupts) -> list[dict]:
    """Pull pending tool-approval requests out of a LangGraph interrupt payload."""
    requests: list[dict] = []
    for item in interrupts or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            requests.extend(value.get("action_requests", []))
    return requests


def format_action_request(request: dict, max_chars: int = 3000) -> str:
    """Readable description of a tool call awaiting approval.

    Added lines start with "+" and removed lines with "-" so UIs can colour them.
    """
    name = request.get("name", "tool")
    args = request.get("args") or {}
    if name == "execute_command":
        body = f"$ {args.get('command', '')}"
    elif name == "write_safe_file":
        body = preview_write(str(args.get("path", "")), str(args.get("content", "")))
    elif name == "replace_in_file":
        removed = [f"- {line}" for line in str(args.get("old_text", "")).splitlines()]
        added = [f"+ {line}" for line in str(args.get("new_text", "")).splitlines()]
        body = f"Path: {args.get('path', '')}\n\n" + "\n".join(removed + added)
    else:
        body = "\n".join(f"{key}: {value}" for key, value in args.items())
    if len(body) > max_chars:
        body = body[:max_chars] + "\n...[truncated]"
    return f"{name}\n{body}"


def make_decision(approved: bool) -> dict:
    if approved:
        return {"type": "approve"}
    return {
        "type": "reject",
        "message": "The user rejected this action. Do not retry it; ask how to proceed.",
    }


def resume_command(decisions: list[dict]) -> Command:
    """Command that resumes an interrupted run with one decision per request."""
    return Command(resume={"decisions": decisions})


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

def _default_status(message: str) -> None:
    console.print(f"[yellow]{escape(message)}[/yellow]")


def new_thread_config() -> dict:
    """Create a fresh graph thread for one request.

    Planning artifacts, rather than an ever-growing LangGraph checkpoint, are
    Wallace's cross-turn memory.  A thread is still reused within a turn so
    approvals and tool results remain intact.
    """
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT,
    }


async def _load_docs_tools(notify: Callable[[str], None]) -> list:
    """Load LangChain documentation tools over MCP; failure is non-fatal."""
    if MCPAdapter is None:
        notify(
            "LangChain documentation tools are unavailable: could not import "
            f"langchain.mcp ({_MCP_IMPORT_ERROR}). Try: uv add \"langchain[mcp]\". "
            f"Running under {sys.executable}; continuing with local tools."
        )
        return []

    async def fetch() -> list:
        # Each returned tool opens its own short-lived session per call, so
        # the tools stay usable after this block exits.
        async with MCPAdapter(MCP_URL) as adapter:
            return await adapter.list_tools()

    try:
        return await asyncio.wait_for(fetch(), timeout=MCP_TIMEOUT_SECONDS)
    except Exception as error:  # network, timeout, or protocol errors
        notify(
            "LangChain documentation tools are unavailable; continuing with "
            f"local project tools. Details: {error}"
        )
        return []


async def setup_agent(on_status: Callable[[str], None] | None = None):
    """Build the agent. Warnings go to `on_status` (defaults to the console)
    so a TUI can show them without printing over its own screen."""
    notify = on_status or _default_status

    tools = [*await _load_docs_tools(notify), *TOOLS]

    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
        repeat_last_n=256,
        repeat_penalty=1.2,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        middleware=[HumanInTheLoopMiddleware(interrupt_on=APPROVAL_REQUIRED)],
        checkpointer=InMemorySaver(),  # required for approvals (pause/resume)
    )
    thread_config = new_thread_config()
    return agent, thread_config


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _ask_cli_approval(request: dict) -> bool:
    console.print(
        Panel(
            Text(format_action_request(request)),
            title="Approval required",
            border_style="yellow",
        )
    )
    return input("Approve? [y/N]: ").strip().lower() in {"y", "yes"}


def _print_threads(manager: ConversationThreadManager, current_id: str) -> None:
    console.print("[cyan]Conversation threads[/cyan]")
    for record in manager.list_threads():
        marker = "*" if record.thread_id == current_id else " "
        console.print(f"{marker} {record.thread_id}: {record.title}")


def _handle_thread_command(
    command: str,
    manager: ConversationThreadManager,
    current_id: str,
) -> str | None:
    """Handle a slash command and return a new current thread ID if selected."""
    try:
        parts = shlex.split(command)
    except ValueError as error:
        console.print(f"[yellow]Invalid thread command: {error}[/yellow]")
        return current_id
    if not parts:
        return current_id
    action = parts[0].lower()
    try:
        if action in {"/threads", "/thread"}:
            _print_threads(manager, current_id)
        elif action == "/new":
            record = manager.create(" ".join(parts[1:]) or None)
            console.print(f"[green]Created thread {record.thread_id}: {record.title}[/green]")
            return record.thread_id
        elif action == "/switch" and len(parts) >= 2:
            record = manager.resolve(" ".join(parts[1:]))
            console.print(f"[green]Switched to {record.thread_id}: {record.title}[/green]")
            return record.thread_id
        elif action == "/rename" and len(parts) >= 2:
            record = manager.rename(current_id, " ".join(parts[1:]))
            console.print(f"[green]Renamed current thread to: {record.title}[/green]")
        elif action in {"/remove", "/delete"} and len(parts) >= 2:
            removed = manager.remove(" ".join(parts[1:]))
            console.print(f"[green]Removed thread: {removed.thread_id}[/green]")
            if removed.thread_id == current_id:
                replacement = manager.list_threads()[0]
                return replacement.thread_id
        elif action == "/clear":
            manager.store(current_id).clear()
            console.print("[green]Current thread history cleared.[/green]")
        else:
            console.print(
                "[yellow]Thread commands: /threads, /new [title], /switch <id or title>, "
                "/rename <title>, /remove <id or title>, /clear[/yellow]"
            )
    except (OSError, ValueError) as error:
        console.print(f"[yellow]Thread command failed: {error}[/yellow]")
    return current_id


async def _invoke_with_approvals(agent, thread_config, prompt: str) -> dict:
    """Run one turn, pausing for approval whenever the agent asks for it."""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    recovery_attempted = False
    while True:
        try:
            with console.status("Thinking..."):
                result = await agent.ainvoke(payload, thread_config)
        except Exception as error:
            if not recovery_attempted and is_malformed_tool_call_error(error):
                recovery_attempted = True
                console.print(
                    "[yellow]The model returned malformed tool-call JSON; "
                    "recovering from the saved turn state.[/yellow]"
                )
                payload = await malformed_retry_payload(
                    agent, thread_config, prompt
                )
                continue
            raise
        requests = extract_action_requests(result.get("__interrupt__"))
        if not requests:
            return result
        decisions = [make_decision(_ask_cli_approval(request)) for request in requests]
        payload = resume_command(decisions)


async def run_agent(
    *,
    use_planning: bool = True,
    session_id: str | None = None,
    thread_id: str | None = None,
    new_thread: bool = False,
    thread_title: str | None = None,
) -> None:
    project_root = Path(__file__).resolve().parent
    ArchiveManager(project_root).run_at_session_boundary()
    console.print("[cyan]Wallace planning is active for every request.[/cyan]")
    thread_manager = ConversationThreadManager(project_root)
    selected_id = thread_id or session_id or "main"
    if new_thread:
        selected_id = thread_manager.create(thread_title).thread_id
    else:
        selected_id = thread_manager.ensure(selected_id).thread_id
    conversation_store = thread_manager.store(selected_id)
    conversation_history = conversation_store.load()
    console.print(f"[cyan]Conversation thread: {selected_id}[/cyan]")
    with console.status("Starting agent..."):
        agent, thread_config = await setup_agent()

    while True:
        print("\n-------- User ---------")
        try:
            prompt = input("Input: ")
        except EOFError:
            break
        if prompt.strip().lower() == "exit":
            break
        if not prompt.strip():
            continue

        if prompt.startswith("/"):
            next_id = _handle_thread_command(prompt, thread_manager, selected_id)
            if next_id != selected_id:
                selected_id = next_id or selected_id
                conversation_store = thread_manager.store(selected_id)
                conversation_history = conversation_store.load()
                thread_config = new_thread_config()
                console.print(f"[cyan]Now using conversation thread: {selected_id}[/cyan]")
            continue

        plan = None
        agent_prompt = prompt
        conversation_store.save_pending(prompt)
        conversation_history = ConversationStore.trim(
            [*conversation_history, {"role": "user", "content": prompt}]
        )
        conversation_store.save(conversation_history)
        try:
            plan = resume_or_create(project_root, prompt, session_id=selected_id)
            console.print(
                f"[cyan]Plan {plan.plan_id} ready; next step: "
                f"{plan.next_pending().title if plan.next_pending() else 'complete'}[/cyan]"
            )
            agent_prompt = planning_context(plan) + "\nUser request:\n" + prompt
        except (OSError, ValueError, TypeError) as error:
            console.print(f"[yellow]Planning setup failed; continuing normally: {error}[/yellow]")

        try:
            result = await _invoke_with_approvals(agent, thread_config, agent_prompt)
        except Exception as error:
            if plan is not None:
                activity_messages = []
                try:
                    state = await agent.aget_state(thread_config)
                    activity_messages = state.values.get("messages", [])
                except Exception:
                    pass
                try:
                    plan.record_progress(
                        model_tool_activity(activity_messages),
                        error=(
                            describe_error(error)
                            if is_malformed_tool_call_error(error)
                            else str(error)
                        ),
                    )
                    save_plan(project_root, plan)
                except Exception:
                    pass
            console.print(f"[yellow]{escape(describe_error(error))}[/yellow]")
            thread_config = new_thread_config()
            continue

        response = assistant_response_text(result.get("messages", []))
        if response is None:
            console.print("[yellow]Continuing the unfinished turn once...[/yellow]")
            summary_error = None
            try:
                retry_result = await _invoke_with_approvals(
                    agent, thread_config, EMPTY_RESPONSE_RETRY_PROMPT
                )
                response = assistant_response_text(retry_result.get("messages", []))
                result = {
                    **retry_result,
                    "messages": [
                        *result.get("messages", []),
                        *retry_result.get("messages", []),
                    ],
                }
            except Exception as error:
                summary_error = error
            if response is None:
                activity_messages = result.get("messages", [])
                try:
                    state = await agent.aget_state(thread_config)
                    activity_messages = state.values.get("messages", activity_messages)
                except Exception:
                    pass
                activity = model_tool_activity(activity_messages)
                if plan is not None:
                    plan.record_progress(
                        activity,
                        error="No final assistant text after one recovery attempt.",
                    )
                status = no_text_status(plan, activity)
                if summary_error is not None:
                    status = f"{describe_error(summary_error)} {status}"
                if plan is not None:
                    plan.record_progress(
                        [],
                        error=status,
                    )
                    save_plan(project_root, plan)
                console.print(f"[yellow]{escape(status)}[/yellow]")
                thread_config = new_thread_config()
                continue

        quality_error = response_quality_error(response)
        if quality_error is not None:
            if plan is not None:
                plan.record_progress(
                    model_tool_activity(result.get("messages", [])),
                    error=quality_error,
                )
                save_plan(project_root, plan)
            console.print(f"[yellow]{escape(quality_error)}[/yellow]")
            thread_config = new_thread_config()
            continue

        if plan is not None:
            plan.record_progress(
                model_tool_activity(result.get("messages", [])),
                response=response,
            )
            save_plan(project_root, plan)

        conversation_history.append({"role": "assistant", "content": response})
        conversation_history = ConversationStore.trim(conversation_history)
        conversation_store.save(conversation_history)
        conversation_store.clear_pending()
        thread_manager.touch(selected_id)

        # Do not carry this turn's full tool transcript into the next request.
        thread_config = new_thread_config()

        print("\n--------- AI ----------")
        console.print(Markdown(response))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wallace in the terminal.")
    # Kept as a hidden compatibility flag for existing launch scripts. Planning
    # is now always on, so this flag has no behavioral effect.
    parser.add_argument("--use-planning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    parser.add_argument("--thread-id", help="Open a specific conversation thread.")
    parser.add_argument("--new-thread", action="store_true", help="Start a new conversation thread.")
    parser.add_argument("--thread-title", help="Title for a new conversation thread.")
    args = parser.parse_args()
    if args.thread_id and args.session_id:
        parser.error("use --thread-id or the legacy --session-id, not both")
    asyncio.run(
        run_agent(
            use_planning=args.use_planning,
            session_id=args.session_id,
            thread_id=args.thread_id,
            new_thread=args.new_thread,
            thread_title=args.thread_title,
        )
    )


if __name__ == "__main__":
    main()
