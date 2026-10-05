import asyncio
import logging
import uuid

from langchain.agents import create_agent
from langchain_core.exceptions import ModelRateLimitError
from langchain.rate_limiters import InMemoryRateLimiter
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from tools import list_directory, read_safe_file, write_safe_file, execute_command, check_code_quality

load_dotenv()

console = Console()


def _filter_unsupported_schema_warning(record: logging.LogRecord) -> bool:
    return record.getMessage() != (
        "Key 'additionalProperties' is not supported in schema, ignoring"
    )


logging.getLogger("langchain_google_genai._function_utils").addFilter(
    _filter_unsupported_schema_warning
)

SYSTEM_PROMPT = """

    You are a helpful, but always strive to be concise and clear in your responses.
    You should provide direct and relevant answers to the user's queries.
    If you cannot find an accurate answer to the user's query, you should clearly state that you do not know the answer.
    Do not make up any answer unless the user requests you to do so.
    You can inspect this project with the list_directory and read_safe_file tools.
    Their paths are relative to the project root, and access is read-only.
    You can inspect, write, and edit project files using `list_directory`, `read_safe_file`, and `write_safe_file`.                           
    You can also run tests and shell commands using `execute_command` and check code quality using `check_code_quality`.                                                                        
    All file and command operations are restricted to the project root directory.

    Your personality is as follows:
    - You are concise and professional in your responses.
    - You are polite and courteous in your interactions.
    - You are straight to the point and direct.

"""

MCP_URL = "https://docs.langchain.com/mcp"


async def run_agent() -> None:
    tools = [list_directory, read_safe_file, write_safe_file, execute_command, check_code_quality]
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

    agent = create_agent(
        model="google_genai:gemini-flash-lite-latest",
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=InMemorySaver(),
        rate_limiter=rate_limiter,
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
                    "[yellow]The model's rate or token quota has been reached. "
                    "Wait for the quota window to reset, then try again. "
                    "If this keeps happening, check your Gemini API usage and "
                    f"billing details. Details: {error}[/yellow]"
                )
                continue
            response = result["messages"][-1].text
        except EOFError:
            break
        print("\n--------- AI ----------")
        console.print(Markdown(response))


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
