# Coding Agent

A command-line coding agent built with LangChain, Gemini, and LangGraph. It can
answer questions, search the LangChain documentation through its MCP server,
inspect project files, write and edit files, and execute shell commands or tests.

## What it can do today

- Answer questions with `gemini-flash-lite-latest`.
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp`.
- List project directories with `list_directory`, read UTF-8 project files
  with `read_safe_file`, and write/edit project files with `write_safe_file`.
- Execute shell commands and tests using `execute_command`.
- Keep conversation state in memory for the duration of one run.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file access and command execution are restricted to the project root directory.

## Long-term goal

The goal is to develop this into a fully capable coding agent that can:

1. Read and understand its own source code and the rest of the project.
2. Propose, write, and edit code changes in any programming language.
3. Run approved commands, tests, and other code to verify changes.
4. Explain what it changed and show a Git diff for review.
5. Communicate fluently in all spoken and programming languages.
6. Speak and listen to the user via the device's microphone and speaker.

The agent has evolved to support writing/editing files and executing shell commands/tests, functioning as a fully capable coding agent.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Create a `.env` file in the project root containing your Gemini API key:

   ```dotenv
   GOOGLE_API_KEY=your-key-here
   ```

   Keep the key private. The project file tools intentionally deny access to
   `.env` files.

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
the program exits.

## Updating dependencies

If you change dependencies in `pyproject.toml`, refresh the installed editable
command from the project root:

```powershell
uv tool install --editable . --reinstall
```
