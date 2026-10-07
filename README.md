# Coding Agent

A command-line coding agent built with LangChain, Ollama, and LangGraph. It can
answer questions, search the LangChain documentation through its MCP server,
inspect project files, write and edit files, and execute shell commands or tests.

## What it can do today

- Answer questions with a local Ollama model (replace the placeholder model in configuration as needed).
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp`.
- List project directories with `list_directory`, read UTF-8 project files
  with `read_safe_file`, and write/edit project files with `write_safe_file`.
- Execute shell commands and tests using `execute_command`.
- Keep conversation state in memory for the duration of one run.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file tools are restricted to the project root directory. Commands run
from the project root using the permissions of the local agent process.

## Long-term goal

The goal is to develop this into a fully capable coding agent that can:

1. Read and understand its own source code and the rest of the project.
2. Propose, write, and edit code changes in any programming language.
3. Run approved commands, tests, and other code to verify changes.
4. Explain what it changed and show a Git diff for review.
5. Communicate fluently in all spoken and programming languages.
6. Speak and listen to the user via the device's microphone and speaker.
7. Have a user interface.

The agent has evolved to support writing/editing files and executing shell commands/tests, functioning as a fully capable coding agent.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install [Ollama](https://ollama.com/download) and download a model for your setup:

   ```powershell
   ollama pull <your-model-name>
   ```

   Make sure the Ollama server is running at `http://localhost:11434`, and update
   the model name in the project configuration to match the model you chose.

3. From the project root, install the editable command:

   ```powershell
   uv tool install --editable .
   uv tool update-shell
   ```

4. Close and reopen your terminal so the updated PATH is available.

## Run and use it

Start the agent from a terminal:

```powershell
agent
```

You can also run it from the project root without installing the command:

```powershell
uv run main.py
```

To use the Terminal User Interface (TUI), run:

```powershell
uv run tui_agent.py
```

Type a question at the `Input:` prompt. For example:

- `Search the LangChain docs for how to create an agent with tools.`
- `List the files in the project root.`
- `Read main.py and explain how the agent starts.`
- `Write a file called test.md and then remove it.`

Type `exit` or send EOF to quit. Documentation tools require a network
connection; if the MCP server cannot be reached, the agent reports the issue
and continues with the local project-reading tools.

The CLI displays a status while it connects to documentation tools and while
it waits for a response. Conversation memory is in-process and is lost when
the program exits. The agent uses its tools to inspect or modify project files;
the model itself does not have direct access to the filesystem. Empty model
responses and Ollama request failures are reported, and the CLI remains ready
for another prompt.

## Updating dependencies

If you change dependencies in `pyproject.toml`, refresh the installed editable
command from the project root:

```powershell
uv tool install --editable . --reinstall
```
