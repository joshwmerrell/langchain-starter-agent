import asyncio
import logging
import uuid
from collections.abc import Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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
    get_git_diff,
    get_system_info,
    replace_lines
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


# def _assistant_response_text(messages: Sequence[BaseMessage]) -> str | None:
#     if not messages or not isinstance(messages[-1], AIMessage):
#         return None
#     return messages[-1].text.strip() or None
# The following code is a fix suggested by Google Gemini.
def _assistant_response_text(messages: Sequence[BaseMessage]) -> str | None:
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

    # 2. Fallback: If AI executed tools but provided no text summary, return the tool output(s)
    tool_outputs = []
    for msg in reversed(current_turn_messages):
        if isinstance(msg, ToolMessage):
            content = str(msg.content).strip()
            if content:
                tool_outputs.append(content)

    if tool_outputs:
        return "\n\n".join(tool_outputs)

    return None


SYSTEM_PROMPT = """
You are a concise, professional coding assistant. Be polite and direct, and
do not invent facts.

CRITICAL INSTRUCTION:
Whenever you invoke a tool, wait for the tool execution result, and then write a 
final text response summarizing or explaining your actions and findings to the 
user. Never end your turn with an empty text response or only a raw tool call.

You have tools for this project. Use them whenever a request requires
inspecting or changing project files, running a command or test, checking
Python syntax, reviewing the Git diff, or looking up LangChain documentation.
Do not claim you cannot access project files when an appropriate tool is
available; call the tool and use its result. If a tool fails or is unavailable,
explain that clearly instead of guessing.

Available project tools:
- `list_directory` lists project files and directories. Use `.` for the project root.
- `read_safe_file` reads a project file using its path relative to the root.
- `write_safe_file` creates or replaces a project file when the user asks for a change.
- `execute_command` runs a command from the project root.
- `get_git_diff` shows current uncommitted changes.
- `get_system_info` reports hardware and OS details. Pass a section:
  summary, os, cpu, memory, gpu, disk, temps, network, processes, python.
- `search_files` searches project files for a regex pattern.
- `read_file_lines` reads a range of lines from a file.
"""


MCP_URL = "https://docs.langchain.com/mcp"


async def setup_agent():
    tools = [
        list_directory,
        read_safe_file,
        write_safe_file,
        execute_command,
        get_git_diff,
        get_system_info,
        search_files,
        read_file_lines,
        edit_file,
        replace_lines
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

    model = ChatOllama(
        model="gemma4:26b",
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=16384,
        num_predict=4096,
        keep_alive="15m",
        rate_limiter=rate_limiter,
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=InMemorySaver(),
    )
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return agent, thread_config


async def run_agent() -> None:
    agent, thread_config = await setup_agent()
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
    # model = ChatGoogleGenerativeAI(
    #     model="gemini-flash-lite-latest",
    #     rate_limiter=rate_limiter
    # )
    model = ChatOllama(
        model="gemma4:26b",
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=16384,
        num_predict=4096,
        keep_alive="15m",
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
