# Wallace

Wallace is an experimental, local-first AI agent being developed into a
general-purpose personal assistant and coding workhorse. Today it uses a local
Ollama model or an optional OpenAI/Gemini model with LangChain-powered tools.
It offers both a command-line interface and a Textual terminal interface. It
can inspect and edit project
files, run commands and tests with approval, and keep resumable conversation
threads and plans.

**Status: work in progress.** Wallace and its [roadmap](wallace_roadmap.md) are
unfinished. Cross-language code operations, general web research, cloud workspace access,
and voice interaction are not complete. Model tool calls and multi-step tasks
can still fail; review proposed edits and commands, keep backups, and do not
treat Wallace as production-ready.

## What it can do today

- Answer questions with a local Ollama model. Configure it with the
  `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `WALLACE_NUM_CTX`, `WALLACE_NUM_PREDICT`,
  and `WALLACE_KEEP_ALIVE` environment variables, or switch installed models
  from inside Wallace with `/model`.
- Use an OpenAI or Google Gemini model with a key entered in a masked prompt.
  Wallace can also load provider keys from a private `.env` file.
- Search and read LangChain documentation using tools provided by
  `https://docs.langchain.com/mcp` (unless disabled with `/mcp off langchain-docs`).
- Manage remote MCP connections with `/mcp` in the CLI or TUI (or F11 in the
  TUI). Server links and connection states are visible to Wallace through a
  read-only tool.
- List project directories with `get_project_tree`, read UTF-8 project files
  with `read_safe_file`, and write/edit project files with `write_safe_file` or
  `replace_in_file`. If an exact replacement cannot match, `edit_file_lines`
  can edit a small range after re-reading its current line numbers.
- Execute shell commands and tests using `execute_command` and `run_tests`.
  If a resumed validation plan is waiting on a previously failed full-suite
  run and the model skips it, Wallace asks for the normal tool approval and
  runs that suite through the host before accepting completion.
- Persist each conversation thread independently so CLI and TUI sessions can be
  resumed without sharing unrelated history. Older turns are summarized for
  the model while the original text remains in the local thread file.
- Plan every request with bounded, resumable multi-step work under
  `.wallace/plans/`, keeping the model context focused on the current task.
- Keep independent conversation threads under `.wallace/conversations/`.
- Continue working with its local project tools if the documentation MCP
  server is unavailable.

Project file tools are restricted to the project root directory. Commands run
from the project root using the permissions of the local agent process.
Python edits made through `write_safe_file`, `replace_in_file`, or
`edit_file_lines` are compiled in memory before writing; invalid syntax leaves
the original file untouched.
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

To run Wallace from this source tree, you need [uv](https://docs.astral.sh/uv/getting-started/installation/).
For local models, you also need [Ollama](https://ollama.com/download) and a
downloaded model that supports tool calls. The project requires Python 3.13 or newer; its
`.python-version` selects 3.13, and uv can download it for you. A GPU is
optional, but the model you choose must fit your available memory; see
[Ollama's GPU guidance](https://docs.ollama.com/gpu) if you want acceleration. You do not
need Node.js, ESLint, TypeScript, Omarchy, or a separate `watchfiles` install
just to run the current CLI or TUI. Git is only needed if you must clone the
source repository first. On Linux, install your distribution's `bubblewrap`
package (`bwrap`) for Wallace's command and test tools. If Bubblewrap is
missing or cannot create its sandbox, those tools fail closed; file tools and
chat still work. Command tools are not supported on Windows in this version.

1. Install uv and, if using local models, Ollama. These are the commands published by their installers;
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

3. For local models, make sure Ollama is running. Open the Ollama app on macOS/Windows. On Linux,
   use its installed service or run `ollama serve` in a separate terminal if
   the server is not already running. Wallace connects to
   `http://localhost:11434` by default.

4. For local models, download the model configured in `main.py`, then confirm it appears locally:

   ```bash
   ollama pull SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest
   ollama ls
   ```

   That [model](https://ollama.com/SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU%3Alatest)
   is Wallace's current default, not a requirement to use this particular
   model. To use another tool-capable local model, pull it with `ollama pull`
   and choose it in Wallace with `/model <name>`, or set `OLLAMA_MODEL` to its
   exact name before launching Wallace. For example, use
   `export OLLAMA_MODEL="your-model:tag"` in Bash/Zsh or
   `$env:OLLAMA_MODEL = "your-model:tag"` in PowerShell. The selected model
   must be available to the running Ollama server. A model selected inside
   Wallace takes precedence over `OLLAMA_MODEL` on later starts.

The LangChain documentation lookup needs an internet connection, but Wallace
can still use its local project tools when that service is unavailable. General
web research, cloud-workspace access, and voice features in the roadmap are not
installed by these steps. Cloud model access is described below.

## Run and use it

Run commands from the project root. Start the TUI with:

```bash
uv run tui_agent.py
```

Or use the command-line interface:

```bash
uv run main.py
```

The selected thread determines file access. Every ordinary thread uses
`workspace/` inside the project root, even when opened in a development
launcher. Only the permanent `wallace_dev` thread can use Wallace's source
tree, and that thread is visible only in development launchers:
`uv run dev_tui.py`, `uv run dev_cli.py`, or `uv run main.py --dev`.
On a fresh setup the development TUI starts on `wallace_dev`; later it
reopens the last selected thread. `wallace_dev` cannot be renamed or removed,
and normal launchers cannot select it. File, command, test, dependency,
and editor tools use the selected thread's workspace. `.env` and Wallace's
private state are masked from model-invoked command tools even in `wallace_dev`.

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
Plans retain a bounded journal of tool names, targets, outcomes, and safe error
categories, but not file contents or edit bodies. They also keep compact
validation evidence for each changed file and unresolved failed checks when
older tool calls roll out of the journal. An implementation plan is not marked complete
just because Wallace said it would begin: relevant checks for changed files, no
unresolved failed edits or checks, and a final
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

For the development CLI, run `uv run dev_cli.py` (or
`uv run main.py --dev`). Both launchers show the same ordinary threads and
their conversations. The active selection is remembered separately in each
launcher so switching in one window does not steal focus in the other. Both
TUIs refresh shared thread lists and the open conversation about every two
seconds when idle. Restart already-running TUI instances after upgrading to
this version so they load the shared-thread behavior.

That command starts Wallace and restarts the TUI when a Python file changes.
If Wallace is in the middle of a turn or waiting for approval, the reload is
queued until the turn finishes, so editing a Python file no longer cuts off the
agent. This is still a process restart rather than an in-place replacement of
Python classes. Completed ordinary TUI turns are restored from
`.wallace/workspace_conversations/`; `wallace_dev` uses
`.wallace/conversations/`. Press Ctrl+L to clear both the
visible log and the saved conversation. The normal `uv run tui_agent.py`
command remains unchanged.

The watcher now checks changed Python syntax, imports Wallace's modules,
constructs the TUI, and verifies that local tool names are unique before
stopping the running TUI. It reports a failed startup and stays active so a
later file fix can restart Wallace. This is a startup check, not a substitute
for tests or a guarantee that a new feature behaves correctly.

### Let Wallace build tools

Ask Wallace for one small tool and its acceptance tests at a time. Local tools
are registered in `tools.py`; changes to that registry need a passing
`verify_tool_contract` check against a test file that calls the tool's
`.invoke(...)` method before an implementation plan can complete. New local
tools follow the active thread's trust mode until explicitly reviewed as read-only.
Wallace can run `check_source_file` for Python, JavaScript, TypeScript, HTML,
CSS, shell, JSON, TOML, Go, and Rust files, or `run_language_tests` for Python,
Deno, npm, Go, and Cargo projects. TypeScript checks use Deno for Deno projects
and standalone files, or `tsc` for Node projects. Deno checks use cached
dependencies and do not download packages. These fixed validation commands
follow the active thread's trust mode. TypeScript and Rust compiler errors may include type or
dependency failures, not only syntax errors. HTML/CSS checks are basic parse
checks, not browser or interaction tests. Optional runtimes and test runners
must already be installed. A newly added tool is
available to the live agent after an idle reload, then should be tried in a
new turn. The current turn can verify its registry entry and tests but cannot
call a tool that was not in its original agent graph.

For remote MCP servers, type `/mcp` in the CLI or TUI (or press F11 in the TUI)
to see each server's URL, active choice, and current connection state. The built-in
`langchain-docs` server is active by default and cannot be removed, but you can
turn it off. Add a server with `/mcp add <name> <URL> [keywords...]`; use
`/mcp on <name>`, `/mcp off <name>`, `/mcp toggle <name>`, or
`/mcp remove <name>` to manage it. `/mcp reload` retries discovery. Wallace
rebuilds its tool list while keeping the current conversation thread. Active
choices persist across restarts in `.wallace/mcp_servers.json`; “connected”
means the most recent tool discovery succeeded, so that status can change
between sessions. Changes made through the running TUI do not trigger a second
development-TUI restart. CLI changes and manual edits to the private config
still refresh a separately running development TUI.

You can also copy `wallace_mcp.example.json` to `.wallace/mcp_servers.json` and
edit it yourself. This currently supports remote URL-based MCP servers, not
local stdio commands. HTTPS is required except for loopback HTTP. The optional
`keywords` help Wallace offer server tools for relevant requests. All
user-added MCP tool calls follow the active thread's trust mode in every
launcher; duplicate tool names are skipped. MCP servers are external to the
local workspace sandbox and can have broader access, so enable only servers
you trust.
URLs with embedded usernames/passwords, query strings, or fragments are
rejected; do not put tokens in URL paths either, because Wallace can display
those paths to the model. Authenticated services need a separately managed
token/OAuth broker. MCP
servers execute outside Wallace's local file-tool restrictions, so review
their code, permissions, and approval prompts before enabling them. Do not
put API keys in the MCP configuration.

Wallace's model has a read-only tool to view these links and no MCP management
tool. Its project file-edit tools cannot access `.wallace`. Command and test
tools use a Bubblewrap mount sandbox, but this is not a substitute for a
separate OS account or VM against all possible local services and exploits.

Type a question at the `Input:` prompt. For example:

- `Search the LangChain docs for how to create an agent with tools.`
- `List the files in the project root.`
- In `wallace_dev`: `Read main.py and explain how the agent starts.`
- `Write a short note to test.md.`

In the CLI, type `exit` or send EOF to quit; in the TUI, press Ctrl+Q.
Documentation tools require a network
connection; if the MCP server cannot be reached, the agent reports the issue
and continues with the local project-reading tools.

Conversation threads are independent. In either interface, use `/threads` to
list them, `/new [--trust MODE] [title]` to create one, `/switch <id or title>` to select one,
`/rename <title>` to rename the current thread, and `/clear` to clear only the
current thread. Use `/remove <id or title>` (or `/delete`) to remove a thread;
Wallace keeps the last remaining thread. The CLI also accepts
`--thread-id <id>` or `--new-thread --thread-title "Title" --trust-mode MODE`.
Use `/trust` to see the current thread's mode and `/trust <mode>` to change it.
Use `/trust informed-trust 15` to set that thread's auto-approval countdown
to 15 seconds (whole seconds from 1 to 300). `/trust informed-trust` resets
it to the five-second default. The choice is saved with the thread.
Every new thread, including one created without an explicit mode, starts in
`no-trust`. The current mode is shown on startup, after switching threads, and
in the thread list. It persists across restarts. Wallace's model has no
dedicated trust-setting tool. Trust mode changes approval behavior, not the
per-thread workspace boundary.

- `no-trust`: each approval-required action waits for your Y/N decision.
- `informed-trust`: the action preview counts down for 5 seconds by default and then
  approves automatically. Moving/hovering the mouse, clicking, scrolling, or
  pressing a key pauses the TUI countdown indefinitely so you can decide with
  Y/N. In the CLI, any key pauses it; the CLI does not receive mouse hover.
- `yes-trust`: approval-required tools run without a prompt. This includes
  file changes, shell commands, tests, and configured MCP tools. Use this mode
  only when you trust the model and connected tools.
  The Bubblewrap boundary still applies to shell and test commands, but it
  does not make the whole OS account or external services safe.
Older ordinary threads from `.wallace/conversations/` are copied once into
the shared `.wallace/workspace_conversations/` store without deleting the
originals. If an old `main` conflicts with the newer ordinary `main`, it is
preserved as `legacy_main`. Only `wallace_dev` remains in the development-only
store. Previously saved conversations may still mention source files that
were visible before this boundary changed; the new boundary governs fresh tool
access, not historical transcript text. A normal session cannot select the
development thread. An explicit CLI thread
flag overrides the saved selection; if a saved ordinary thread was removed,
Wallace falls back to `main`.

In the TUI input bar, press Up and Down to browse previous inputs from the
active thread. Your unfinished draft is restored when you move back past the
newest history entry. Use Shift+Left/Right to select characters,
Ctrl+Shift+Left/Right to select words, and Ctrl+A to select the whole input.
Ctrl+Left/Right moves by word; Ctrl+Backspace/Alt+Backspace and Ctrl+Delete
delete words. Ctrl+C/X/V copies, cuts, and pastes; Ctrl+Z undoes and Ctrl+Y
or Ctrl+Shift+Z redoes an unsent edit. Home/End and their Shift/Ctrl variants
work as in a single-line editor. The header shows the local weekday, date and
time. Tool approval dialogs show `Y` for Yes and `N` for No (`Esc` also rejects).

Press F5 or enter `/edit` to search for an existing workspace file, then open
it in Neovim, Vim, or vi in the same terminal. `/edit path/to/file` opens a
specific file. The TUI pauses while you edit and resumes afterward; a
development reload waits until the editor closes. Only user-visible,
non-sensitive workspace files appear in the picker or can be opened directly;
the model is not given an editor tool. Save and quit the editor normally
(`:wq` in Vim/Neovim). This is an external terminal editor, not an embedded
Vim clone.

Press F9 or enter `/theme` to open the theme picker. Wallace includes
`wallace-hacker`, `wallace-retro`, `wallace-elegant`, `wallace-dark`,
`wallace-vaporwave`, `wallace-grayscale`, `wallace-night`, `wallace-red`,
and the original `wallace-classic`.
`wallace-dark` keeps black surfaces and grayscale body text, with red accents
for section headings, the active thread, user labels, warnings, errors, and
the working indicator; `wallace-vaporwave` uses the Electric Arcade palette.
You can also enter `/theme hacker` (or another name) directly. Old saved
`wallace-night-mode` and `wallace-red-light` choices load as `wallace-night`
and `wallace-red`; the removed `wallace-future` falls back to classic. The
selected theme is remembered in `.wallace/ui.json` across restarts. On Omarchy
systems, the picker includes installed `omarchy-<name>` palettes. It reads
their colors without changing the desktop theme. `ansi-dark` and `ansi-light`
use the terminal's own palette, so terminal themes work on other systems too.
Textual's other built-in themes are also available in the picker.

In the TUI, press F10 or enter `/models` to list models from Ollama, OpenAI,
and Gemini. Use `/models ollama`, `/models openai`, or `/models gemini` to
show one provider. The CLI supports the same commands. Cloud lists use your
account's API key and may include models that cannot handle Wallace's chat or
tool calls. Use `/model <exact-name>`
for a local model, `/model openai:<model-id>`, or
`/model gemini:<model-id>` to switch without leaving Wallace. The model choice—not its key—is saved in
`.wallace/model.json`. Switching creates a fresh model session but preserves
the current conversation thread and history. Invalid local names and failed
model setup leave the previous selection unchanged.

Use `/status` in the CLI or TUI to see the current thread, model, trust mode,
workspace, and provider-reported input/output token totals collected by this
running Wallace process. Wallace's read-only `get_session_status` tool can
answer the same question in chat. The counter works with any future model
provider whose LangChain response includes token-usage metadata; it is not
limited to OpenAI or Gemini. Counts reset when Wallace restarts and can be
incomplete if a provider omits usage metadata. They do **not** show a cloud
account's remaining balance, rolling rate-limit headroom, or reset time;
check that provider's usage dashboard for those values. `/status` never
reads or displays API-key values.

To use cloud models without entering a key each time, create a `.env` file in
the project root with the provider keys you need:

```dotenv
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

Wallace loads these exact variable names on startup. Existing environment
variables take precedence over values in `.env`. The `.env` file is ignored by
Git; keep it private and never commit or paste real keys into chat. Restart
Wallace after changing `.env`. If a selected provider's key is missing, Wallace
asks for it in a masked TUI dialog or a no-echo CLI prompt. Press F12 or use
`/keys` to list provider key names and whether they are active; values are
never displayed. `/key add <name> <openai|gemini>` opens a masked prompt and
saves the value in the operating system's keyring. `/key on <name>`,
`/key off <name>`, and `/key remove <name>` manage those saved keys. Only one
key per provider can be active. `.env` keys appear as
`environment-openai` and `environment-gemini`; they can be toggled but cannot
be deleted through Wallace—remove them from `.env` or your environment
yourself. `/key session` enters a temporary key for the selected cloud model.
Keyring-backed keys need an available, unlocked OS keyring; Wallace does not
fall back to storing their values in its metadata file.
The key is passed to the provider client for HTTP authentication; Wallace does
not intentionally include it in model messages, tool schemas, or conversation
files. Keys typed in the masked prompt are kept only for that process. The CLI
requires an interactive terminal for manual key entry.

Cloud requests send conversation content and relevant tool interactions to the
provider and may incur charges. Wallace still exposes approved command, test,
and file-edit tools in cloud mode, as requested. These run under your OS user
and can potentially read local secrets or alter code that hot-reloads; a
masked key prompt, scrubbed subprocess environment, and Bubblewrap sandbox
reduce exposure but are **not a perfect same-user security boundary**. The
host process must hold each active key to call its provider, and local services
or future integrations could still introduce leaks. Review approvals carefully;
strong isolation would require a separate credential broker and OS account or
VM. See the [official OpenAI API authentication
guidance](https://developers.openai.com/api/reference/overview#authentication)
and [sandbox security guidance](https://developers.openai.com/api/docs/guides/agents-api/environments/security).

The CLI displays a status while it connects to documentation tools and while
it waits for a response. Both interfaces persist thread histories in the local,
git-ignored `.wallace/conversations/` directory so they can be restored after
a development reload. Wallace retains each thread's saved transcript and
sends the model a bounded rolling summary of older turns, a recent exchange,
and the current request. When that exchange grows too long, the selected model makes an
extra request to update the summary without tools; that turn may take
longer. If summarization fails, the raw history stays on disk and Wallace
warns that some older context may be absent from that turn.
Previous assistant replies in memory are labeled unverified; consequential
claims still need checking against the project. The TUI restores only recent
turns in its visible log. History discarded by older Wallace versions cannot
be recovered, and summaries may omit details even though the saved transcript
is intact. Clearing or removing a thread also removes its summary and history.
Obviously copied or looping assistant replies remain in the saved transcript
but are omitted from model-facing memory. For simple requests without tool
activity, Wallace retries such a reply once with a clean prompt. Local date
and time questions use the computer's clock, not the model's training data.

The agent uses its tools to inspect or modify project files; the model itself
does not have direct access to the filesystem. Empty model
responses and Ollama request failures are reported, and the CLI remains ready
for another prompt. File changes, shell commands, and test execution follow
the active thread's trust mode. If pytest is unavailable, `run_tests` falls back to
Python's built-in unittest discovery.

The planning executor only runs explicitly registered tools and routes approval
decisions through the active thread's approval policy; it cannot change that
policy. Plans are cleaned at session boundaries: completed inactive plans are
archived after 30 days, while older archived artifacts are purged after 90 days
with a metadata record retained in `index.json`. Active or unresolved plans are
kept.
Deletion plans require a successful removal command, a separate target-absence
check, Git status, and passing full-project tests before Wallace can claim the
cleanup is verified. A short “verify its completion” or “continue” message
resumes the relevant unfinished plan in the same thread. Removing files from
the working tree does not erase them from older Git commits.

## Updating dependencies

If you change dependencies in `pyproject.toml`, update the lockfile and sync
the project environment from the project root:

```bash
uv lock
uv sync
```
