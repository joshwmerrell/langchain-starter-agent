import asyncio
import logging
import uuid

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.memory import InMemorySaver

# from deepagents import createFilesystemMiddleware, StateBackend

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from tools import list_directory, read_safe_file

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

    Your personality is as follows:
    - You are concise and professional in your responses.
    - You are polite and courteous in your interactions.
    - For moral guidance, you primarily refer to the scriptures and teachings of the Church of Jesus Christ of Latter-day Saints.

"""

MCP_URL = "https://docs.langchain.com/mcp"


async def run_agent() -> None:
    with console.status("Connecting to LangChain documentation tools..."):
        async with MCPAdapter(MCP_URL) as adapter:
            tools = [
                *await adapter.list_tools(),
                list_directory,
                read_safe_file,
            ]

    agent = create_agent(
        model="google_genai:gemini-flash-lite-latest",
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        # middlewares=[createFilesystemMiddleware(backend=StateBackend())],
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

            with console.status("Thinking..."):
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    thread_config,
                )
            response = result["messages"][-1].text
        except EOFError:
            break
        print("\n--------- AI ----------")
        console.print(Markdown(response))


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()