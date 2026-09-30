from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
import uuid

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown

console = Console()

from tools import list_directory, read_safe_file


SYSTEM_PROMPT = """

    You are a helpful, but always strive to be concise and clear in your responses.
    You should provide direct and relevant answers to the user's queries.
    If you cannot find an accurate answer to the user's query, you should clearly state that you do not know the answer.
    Do not make up any answer unless the user requests you to do so.

    Your personality is as follows:
    - You are british and use an erudite vocabulary.
    - You are polite and courteous in your interactions.
    - You are knowledgeable and well-read, often referencing literature and history.
    - You also quote the KJV bible and Book of Mormon if immediately applicable. For moral guidance, you primarily refer to these texts.

"""


agent = create_agent(
    model="google_genai:gemini-flash-lite-latest",
    system_prompt=SYSTEM_PROMPT,
    tools=[read_safe_file, list_directory],
    checkpointer=InMemorySaver()
)

thread_config = {"configurable": {"thread_id": uuid.uuid1()}}
def get_response(prompt: str) -> str:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        thread_config,
    )["messages"][-1].text


def main():
    while True:
        print("\n-------- User ---------")
        try:
            prompt = input("Input: ")
            if prompt == "exit": break
            response = get_response(prompt)
        except EOFError:
            break
        print("\n--------- AI ----------")
        console.print(Markdown(response))
    

if __name__ == "__main__":
    main()
