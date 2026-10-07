import asyncio
import logging
import uuid
from collections.abc import Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain.agents import create_agent
from langchain_core.exceptions import ModelRateLimitError
from langchain.rate_limiters import InMemoryRateLimiter
from langchain.mcp import MCPAdapter
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from ollama import ResponseError

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from tools import (
    edit_file,
    list_directory,
    read_file_lines,
    read_safe_file,
    search_files,
    write_safe_file,
    execute_command,
    check_code_quality,
    get_git_diff,
    get_system_info,
)

load_dotenv()

console = Console()


def _filter_unsupported_schema_warning(record: logging.LogRecord) -> bool:
    return record.getMessage() != (
        "Key 'additionalProperties' is not supported in schema, ignoring"
    )


logging.getLogger("langchain_google_genai._function_utils").addFilter(
    _filter_unsupported_schema_warning
)


def _assistant_response_text(messages: Sequence[BaseMessage]) -> str | None:
    if not messages or not isinstance(messages[-1], AIMessage):
        return None
    return messages[-1].text.strip() or None


SYSTEM_PROMPT = """
You are a concise, professional coding assistant. Be polite and direct, and
do not invent facts.

You have tools for this project. Use them whenever a request requires
inspecting or changing project files, running a command or test, checking
Python syntax, reviewing the Git diff, or looking up LangChain documentation.
Do not claim you cannot access project files when an appropriate tool is
available; call the tool and use its result. If a tool fails or is unavailable,
explain that clearly instead of guessing.

Available project tools:
- `list_directory` lists project files and directories. Use `.` for the
  project root.
- `read_safe_file` reads a project file using its path relative to the root.
- `write_safe_file` creates or replaces a project file when the user asks for
  a change.
- `execute_command` runs a command from the project root.
- `check_code_quality` checks Python syntax for a project file.
- `get_git_diff` shows current uncommitted changes.
- `get_system_info` reports hardware and OS details. Pass a section:
  summary, os, cpu, memory, gpu, disk, temps, network, processes, python.
- `search_files` searches project files for a regex pattern.
- `read_file_lines` reads a range of lines from a file.
- `edit_file` replaces one exact piece of text in an existing file.
- `write_safe_file` creates or replaces a project file when the user asks for a change.

Project file tools use paths relative to the project root and deny access to
`.env` files. Commands start in the project root. Use the tools only for
actions relevant to the user's request, and report their actual results.
"""

MCP_URL = "https://docs.langchain.com/mcp"


async def run_agent() -> None:
    tools = [
        list_directory,
        read_safe_file,
        write_safe_file,
        execute_command,
        check_code_quality,
        get_git_diff,
        get_system_info,
        search_files,
        read_file_lines,
        edit_file
    ]
    with console.status("Connecting to LangChain documentation tools..."):
        try:
            async with MCPAdapter(MCP_URL) as adapter:
                tools = [*await adapter.list_tools(), *tools]
        except RuntimeError as error:
            console.print(
                "[yellow]LangChain documentation tools are unavailable; "
                f"continuing with local project tools. Details: {error}[/yellow]"
            )

    # Initialize a rate limiter to control request frequency and avoid rate limit/token spikes
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2.0,  # Limits requests to 2 per second
        check_every_n_seconds=0.1,  # Check every 100ms
        max_bucket_size=5,  # Controls maximum burst size
    )

    # model = ChatGoogleGenerativeAI(
    #     model="gemini-flash-lite-latest",
    #     rate_limiter=rate_limiter
    # )
    model = ChatOllama(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        temperature=0,
        rate_limiter=rate_limiter,
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=InMemorySaver(),
    )
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        print("\n-------- User ---------")
        try:
            prompt = input("Input: ")
            if prompt == "exit":
                break
            if not prompt.strip():
                continue

            try:
                with console.status("Thinking..."):
                    result = await agent.ainvoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        thread_config,
                    )
            except ModelRateLimitError as error:
                console.print(
                    "[yellow]The model service's rate or token quota has been reached. "
                    "Wait for the quota window to reset, then try again. "
                    "If this keeps happening, check the model provider's usage and "
                    f"billing details. Details: {error}[/yellow]"
                )
                continue
            except (httpx.HTTPError, ResponseError) as error:
                console.print(
                    "[yellow]The local Ollama request failed. Check that Ollama is "
                    "running and the configured model is available, then try again. "
                    f"Details: {error}[/yellow]"
                )
                continue
            response = _assistant_response_text(result["messages"])
            if response is None:
                console.print(
                    "[yellow]The agent finished without a text response. "
                    "Please try again.[/yellow]"
                )
                continue
        except EOFError:
            break
        print("\n--------- AI ----------")
        console.print(Markdown(response))


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
