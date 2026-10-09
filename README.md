# Wallace

Wallace is a personal assistant and workhorse being built with LangChain, Ollama,
and LangGraph. Today, Wallace can answer questions, search the LangChain
documentation through its MCP server, inspect project files, write and edit
files, and execute shell commands or tests.

## What it can do today

- Answer questions with a local Ollama model (replace the placeholder model in configuration as needed).
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp`.
- List project directories with `list_directory`, read UTF-8 project files
  with `read_safe_file`, and write/edit project files with `write_safe_file`.
- Execute shell commands and tests using `execute_command`.
- Keep CLI conversation state in memory and persist completed TUI turns for reloads.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file tools are restricted to the project root directory. Commands run
from the project root using the permissions of the local agent process.

## Long-term goal

The goal is to develop Wallace into an all-around creative and productivity
assistant that can:

1. Understand and work with projects, documents, and information across the
   user's workspaces.
2. Create, understand, and edit code in any computing language.
3. Communicate fluently across spoken languages, adapting to the user's needs.
4. Speak and listen to the user through the computer's speaker and microphone.
5. Access the web and, with the user's authorization, cloud workspaces such as
   Google Workspace.
6. Help with a broad range of creative, practical, and productivity tasks as a
   capable everyday workhorse.
7. Provide a useful interface for both text-based and voice-based interaction.

These are long-term goals; web, cloud-workspace, and voice integrations are not
implied to be available in the current version.

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

Wallace configures `uv` to copy cached package files into the environment.
This avoids hardlink warnings on systems where the project and uv cache are on
different filesystems.

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

For development, you can start the TUI under an automatic reload watcher:

```bash
uv run dev_tui.py
```

That command starts Wallace and restarts the TUI when a Python file changes.
If Wallace is in the middle of a turn or waiting for approval, the reload is
queued until the turn finishes, so editing a Python file no longer cuts off the
agent. This is still a process restart rather than an in-place replacement of
Python classes. Completed TUI turns are restored from
`.wallace/conversation.json` after the restart. Delete that file when you want
to start a fresh TUI conversation. The normal `uv run tui_agent.py` command
remains unchanged.

Type a question at the `Input:` prompt. For example:

- `Search the LangChain docs for how to create an agent with tools.`
- `List the files in the project root.`
- `Read main.py and explain how the agent starts.`
- `Write a file called test.md and then remove it.`

Type `exit` or send EOF to quit. Documentation tools require a network
connection; if the MCP server cannot be reached, the agent reports the issue
and continues with the local project-reading tools.

The CLI displays a status while it connects to documentation tools and while
it waits for a response. The CLI conversation is in-process; the TUI saves
completed turns in the local, git-ignored `.wallace/conversation.json` file so
they can be restored after a development reload. The agent uses its tools to inspect or modify project files;
the model itself does not have direct access to the filesystem. Empty model
responses and Ollama request failures are reported, and the CLI remains ready
for another prompt.

## Updating dependencies

If you change dependencies in `pyproject.toml`, refresh the installed editable
command from the project root:

```powershell
uv tool install --editable . --reinstall
```
