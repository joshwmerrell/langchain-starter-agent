# Wallace

Wallace is an experimental, local-first AI agent being developed into a
general-purpose personal assistant and coding workhorse. Today it uses an
Ollama model with LangChain-powered tools and offers both a command-line
interface and a Textual terminal interface. It can inspect and edit project
files, run commands and tests with approval, and keep resumable conversation
threads and plans.

**Status: work in progress.** Wallace and its [roadmap](wallace_roadmap.md) are
unfinished. Cross-language code operations, general web research, cloud access,
and voice interaction are not complete. Model tool calls and multi-step tasks
can still fail; review proposed edits and commands, keep backups, and do not
treat Wallace as production-ready.

## What it can do today

- Answer questions with a local Ollama model. Configure it with the
  `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `WALLACE_NUM_CTX`, `WALLACE_NUM_PREDICT`,
  and `WALLACE_KEEP_ALIVE` environment variables.
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp`.
- List project directories with `get_project_tree`, read UTF-8 project files
  with `read_safe_file`, and write/edit project files with `write_safe_file` or
  `replace_in_file`.
- Execute shell commands and tests using `execute_command` and `run_tests`.
- Persist each conversation thread independently so CLI and TUI sessions can be
  resumed without sharing unrelated history.
- Plan every request with bounded, resumable multi-step work under
  `.wallace/plans/`, keeping the model context focused on the current task.
- Keep independent conversation threads under `.wallace/conversations/`.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file tools are restricted to the project root directory. Commands run
from the project root using the permissions of the local agent process.
Python edits made through `write_safe_file` or `replace_in_file` are compiled
in memory before writing; invalid syntax leaves the original file untouched.
This does not replace running the project's tests after a change.

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

## Install and set up

To run Wallace from this source tree, you need [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Ollama](https://ollama.com/download), and a locally downloaded Ollama model
that supports tool calls. The project requires Python 3.13 or newer; its
`.python-version` selects 3.13, and uv can download it for you. A GPU is
optional, but the model you choose must fit your available memory; see
[Ollama's GPU guidance](https://docs.ollama.com/gpu) if you want acceleration. You do not
need Node.js, ESLint, TypeScript, Omarchy, or a separate `watchfiles` install
just to run the current CLI or TUI. Git is only needed if you must clone the
source repository first.

1. Install uv and Ollama. These are the commands published by their installers;
   you can use their linked download pages instead if you prefer not to pipe
   installer scripts into a shell.

   Linux or macOS (terminal):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   Windows (PowerShell; Windows setup has not been tested in this repository):

   ```powershell
   winget install --id=astral-sh.uv -e
   irm https://ollama.com/install.ps1 | iex
   ```

   Reopen your terminal after installation so `uv` and `ollama` are on `PATH`.

2. From the Wallace project root (the directory containing `pyproject.toml`),
   install its Python dependencies:

   ```bash
   uv sync --locked
   ```

   [uv](https://docs.astral.sh/uv/guides/projects/) creates `.venv` and installs the dependencies declared in
   `pyproject.toml`, including Textual and the development reload watcher. It
   also downloads Python 3.13 if needed. No separate `pip install` commands
   are required. This project configures uv to copy package files rather than
   hardlink them across filesystems.

3. Make sure Ollama is running. Open the Ollama app on macOS/Windows. On Linux,
   use its installed service or run `ollama serve` in a separate terminal if
   the server is not already running. Wallace connects to
   `http://localhost:11434` by default.

4. Download the model configured in `main.py`, then confirm it appears locally:

   ```bash
   ollama pull SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest
   ollama ls
   ```

   That [model](https://ollama.com/SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU%3Alatest)
   is Wallace's current default, not a requirement to use this particular
   model. To use another tool-capable local model, pull it with `ollama pull`
   and set `OLLAMA_MODEL` to its exact name before launching Wallace. For
   example, use `export OLLAMA_MODEL="your-model:tag"` in Bash/Zsh or
   `$env:OLLAMA_MODEL = "your-model:tag"` in PowerShell. The selected model
   must be available to the running Ollama server.

The LangChain documentation lookup needs an internet connection, but Wallace
can still use its local project tools when that service is unavailable. Web,
cloud, and voice features in the roadmap are not installed by these steps.

## Run and use it

Run commands from the project root. Start the TUI with:

```bash
uv run tui_agent.py
```

Or use the command-line interface:

```bash
uv run main.py
```

To verify the Python installation and current code without starting an agent
turn, run:

```bash
uv run python -m unittest discover -s tests -q
```

Wallace plans every request automatically. To create a named conversation
thread when starting the CLI, use:

```bash
uv run main.py --new-thread --thread-title "My task"
```

Each request creates a Markdown plan, a JSON state file, and private input,
source-reference, and output directories under `.wallace/plans/`. A retry after
a restart resumes the unfinished plan for the same request; a short
“continue” request resumes the latest active plan in that conversation thread.
Plans retain a bounded journal of tool names, targets, and outcomes, but not
file contents or edit bodies. An implementation plan is not marked complete
just because Wallace said it would begin: edits, validation, and a final
response must be observed. Wallace starts a fresh model graph thread after
every request; the plan checkpoint, not the full prior tool transcript, carries
forward useful context. Prior assistant replies are not fed back as plan
checkpoints. If Ollama emits malformed tool arguments, Wallace retries once
from the current turn's checkpoint. A second failure stops with a factual
error and leaves the plan active, without asking the model to invent a
partial handoff. If no final text arrives, Wallace continues the turn once,
then reports the observed work and next pending step without claiming the
request is complete. New repetitive model output is likewise rejected rather
than saved as a completed answer; previously saved loops are collapsed in the
TUI without deleting the original conversation. `--use-planning` remains accepted
silently for old launch scripts, but planning is no longer a mode or toggle.

For development, start the TUI under an automatic reload watcher:

```bash
uv run dev_tui.py
```

That command starts Wallace and restarts the TUI when a Python file changes.
If Wallace is in the middle of a turn or waiting for approval, the reload is
queued until the turn finishes, so editing a Python file no longer cuts off the
agent. This is still a process restart rather than an in-place replacement of
Python classes. Completed TUI turns are restored from
`.wallace/conversations/` after the restart. Press Ctrl+L to clear both the
visible log and the saved conversation. The normal `uv run tui_agent.py`
command remains unchanged.

Type a question at the `Input:` prompt. For example:

- `Search the LangChain docs for how to create an agent with tools.`
- `List the files in the project root.`
- `Read main.py and explain how the agent starts.`
- `Write a short note to test.md.`

In the CLI, type `exit` or send EOF to quit; in the TUI, press Ctrl+Q.
Documentation tools require a network
connection; if the MCP server cannot be reached, the agent reports the issue
and continues with the local project-reading tools.

Conversation threads are independent. In either interface, use `/threads` to
list them, `/new [title]` to create one, `/switch <id or title>` to select one,
`/rename <title>` to rename the current thread, and `/clear` to clear only the
current thread. Use `/remove <id or title>` (or `/delete`) to remove a thread;
Wallace keeps the last remaining thread. The CLI also accepts
`--thread-id <id>` or `--new-thread --thread-title "Title"`.
Existing `.wallace/conversation.json` history is
migrated into the default `main` thread the first time Wallace starts.

In the TUI input bar, press Up and Down to browse previous inputs from the
active thread. Your unfinished draft is restored when you move back past the
newest history entry. The header shows the local weekday, date and time. Tool approval
dialogs show `Y` for Yes and `N` for No (`Esc` also rejects).

Press F9 or enter `/theme` to open the theme picker. Wallace includes
`wallace-hacker`, `wallace-retro`, `wallace-elegant`, `wallace-future`,
`wallace-night-mode`, `wallace-red-light`, and the original `wallace-classic`.
You can also enter `/theme hacker` (or another name) directly. The selected
theme is remembered in `.wallace/ui.json` across restarts. On Omarchy systems,
the picker includes installed `omarchy-<name>` palettes; it reads their colors
without changing the desktop theme. `ansi-dark` and `ansi-light` use the
terminal's own color palette, so terminal themes work on other systems too.
Textual's other built-in themes are also available in the picker.

The CLI displays a status while it connects to documentation tools and while
it waits for a response. Both interfaces persist thread histories in the local,
git-ignored `.wallace/conversations/` directory so they can be restored after
a development reload. The agent uses its tools to inspect or modify project files;
the model itself does not have direct access to the filesystem. Empty model
responses and Ollama request failures are reported, and the CLI remains ready
for another prompt. File changes, shell commands, and test execution require
approval in the TUI. If pytest is unavailable, `run_tests` falls back to
Python's built-in unittest discovery.

The planning executor only runs explicitly registered tools and routes approval
decisions through the existing approval callback; it never bypasses tool
approval. Plans are cleaned at session boundaries: completed inactive plans are
archived after 30 days, while older archived artifacts are purged after 90 days
with a metadata record retained in `index.json`. Active or unresolved plans are
kept.

## Updating dependencies

If you change dependencies in `pyproject.toml`, update the lockfile and sync
the project environment from the project root:

```bash
uv lock
uv sync
```
