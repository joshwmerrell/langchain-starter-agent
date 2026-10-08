import asyncio
import sys
import uuid
from collections.abc import Callable, Sequence

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

load_dotenv()

console = Console()

MCP_URL = "https://docs.langchain.com/mcp"
MCP_TIMEOUT_SECONDS = 15

# Each tool call costs two graph steps (model + tool), so this allows roughly
# 25 tool calls in one turn.
RECURSION_LIMIT = 50

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
Whenever you invoke a tool, wait for the tool execution result, and then write a 
final text response summarizing or explaining your actions and findings to the 
user. Never end your turn with an empty text response or only a raw tool call.

Use your tools whenever a request involves the project's files, running a
command or tests, checking Python syntax, reviewing Git changes, or looking up
LangChain documentation. Do not claim you cannot access project files when a
tool can; call the tool and use its result. If a tool fails or is unavailable,
say so clearly instead of guessing.

Work efficiently:
- Explore with `get_project_tree` and `search_in_files`, then read only what
  you need. For large files, use `get_code_symbols` to find line numbers and
  `read_safe_file` with start_line/end_line.
- Change existing files with `replace_in_file` (copy old_text exactly from the
  file). Use `write_safe_file` only for new files or complete rewrites.
- After editing Python, run `check_python_syntax`, and `run_tests` if the
  project has tests.

`execute_command`, `write_safe_file`, and `replace_in_file` need the user's
approval before they run. If the user rejects an action, do not retry it;
briefly say what you did not do and ask how they would like to proceed.
"""


# ---------------------------------------------------------------------------
# Shared helpers (used by both this CLI and tui_agent.py)
# ---------------------------------------------------------------------------

def assistant_response_text(messages: Sequence[BaseMessage]) -> str | None:
    """Extract the assistant's text for the current turn.

    Prefers AI-generated text and falls back to tool output if the model
    produced no text summary.
    """
    current_turn_messages = []

    # Collect all messages belonging exclusively to the current turn
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        current_turn_messages.append(msg)

    # 1. Search for any AI text response generated in this turn
    for msg in current_turn_messages:
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                return msg.content.strip()
            elif isinstance(msg.content, list):
                text_parts = [
                    block["text"]
                    for block in msg.content
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
                ]
                if text_parts:
                    return "\n".join(text_parts).strip()

    # 2. Fallback: AI executed tools but provided no text summary
    tool_outputs = []
    for msg in reversed(current_turn_messages):
        if isinstance(msg, ToolMessage):
            content = str(msg.content).strip()
            if content:
                tool_outputs.append(content)

    if tool_outputs:
        return "\n\n".join(tool_outputs)

    return None


# Backwards-compatible alias for the old private name.
_assistant_response_text = assistant_response_text


def describe_error(error: Exception) -> str:
    """Turn an exception from an agent run into a user-facing message."""
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
        model="SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest",
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=16384,
        num_predict=4096,
        keep_alive="15m",
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        middleware=[HumanInTheLoopMiddleware(interrupt_on=APPROVAL_REQUIRED)],
        checkpointer=InMemorySaver(),  # required for approvals (pause/resume)
    )
    thread_config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT,
    }
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


async def _invoke_with_approvals(agent, thread_config, prompt: str) -> dict:
    """Run one turn, pausing for approval whenever the agent asks for it."""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    while True:
        with console.status("Thinking..."):
            result = await agent.ainvoke(payload, thread_config)
        requests = extract_action_requests(result.get("__interrupt__"))
        if not requests:
            return result
        decisions = [make_decision(_ask_cli_approval(request)) for request in requests]
        payload = resume_command(decisions)


async def run_agent() -> None:
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

        try:
            result = await _invoke_with_approvals(agent, thread_config, prompt)
        except Exception as error:
            console.print(f"[yellow]{escape(describe_error(error))}[/yellow]")
            continue

        response = assistant_response_text(result["messages"])
        if response is None:
            console.print(
                "[yellow]The agent finished without a text response. "
                "Please try again.[/yellow]"
            )
            continue

        print("\n--------- AI ----------")
        console.print(Markdown(response))


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()