# LangChain Starter Agent

A command-line AI agent built with LangChain, Gemini, and LangGraph. It can
answer questions, search the LangChain documentation through its MCP server,
and inspect project files using a pair of read-only tools.

## What it can do today

- Answer questions with `gemini-flash-lite-latest`.
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp`.
- List project directories with `list_directory` and read UTF-8 project files
  with `read_safe_file`.
- Keep conversation state in memory for the duration of one run.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file access is rooted at the directory containing `tools.py`, not the
terminal's current directory. Paths must stay inside that project root;
`.env` and `.env.*` files are excluded. File tools are read-only.

## Long-term goal

The goal is to develop this into a coding agent that can:

1. Read and understand its own source code and the rest of the project.
2. Propose and write code changes.
3. Run approved commands, tests, and other code to verify changes.
4. Explain what it changed and show a Git diff for review.

The current agent is an early step towards that goal: it can inspect source
files, but it **cannot write or delete files, execute shell commands, run
tests, or commit changes**. Having `deepagents` installed does not enable its
filesystem middleware; the current application uses its own read-only tools.
The model may discuss or suggest code changes, but it cannot apply them.

Compared with an IDE coding agent that can edit files and run a terminal, this
agent is currently a documentation-aware read-only assistant. It is also not
yet a self-modifying or autonomous agent. Adding write and execution tools
would require explicit workspace boundaries, user approval, and a safe
execution environment. Those capabilities should be introduced separately
and verified before they are trusted with project changes.

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
