# Wallace

Wallace is an experimental local-first assistant and coding agent with CLI and
Textual interfaces. It runs Ollama locally or an optional OpenAI/Gemini model,
with file editing, commands/tests, MCP services, and resumable conversations.
General web browsing, cloud-workspace integration, and voice are roadmap work,
not built-in features. See [wallace_roadmap.md](wallace_roadmap.md).

## Quick start

Use Python 3.13+, [uv](https://docs.astral.sh/uv/), and a tool-capable Ollama
model. Install Linux Bubblewrap (`bwrap`) for command and test tools. Those tools
fail closed without it; chat and file tools still work. Command sandboxing is
Linux-specific; other platforms are not supported for command execution.

From the project directory:

```bash
uv sync --locked
ollama pull SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest
uv run tui_agent.py
```

Start Ollama first with its service or `ollama serve` (`http://localhost:11434`).
The model above is the default; use `/model <name>` to choose another installed model.

```bash
uv run main.py                         # CLI
uv run dev_tui.py                      # Development TUI with idle reloads
uv run dev_cli.py                      # Development CLI
uv run main.py --dev                   # Equivalent development CLI entry
uv run main.py --new-thread --thread-title "My task" --trust-mode no-trust
uv run main.py --thread-id <id>
```

Exit the CLI with `exit` or EOF; quit the TUI with Ctrl+Q. The development TUI
queues reloads until a turn, approval, or external editor finishes. It checks
syntax and startup before restarting. This does not replace behavioral tests.

## Workspaces and threads

Every ordinary thread works in this project's `workspace/`, including threads
opened in development launchers. Only the permanent `wallace_dev` thread can
access Wallace's source tree. It is available only in development launchers and
cannot be renamed or removed. Trust settings never widen this boundary.

File tools resolve paths and symlinks, reject escapes, and exclude private state,
credential filenames, Git internals, environments, and package directories.
Shell/test tools mount the active workspace, mask private paths, scrub common
credential variables, and close interactive input. They retain network access.
Read-only Git status/diff tools intentionally inspect the repository as a whole,
including source changes, even from an ordinary thread.

| Command | Purpose |
| --- | --- |
| `/threads` or `/thread` | List threads and their trust modes |
| `/new [--trust MODE] [title]` | Create an independent conversation |
| `/switch <id or title>` | Select a conversation |
| `/rename <title>` | Rename the current ordinary thread |
| `/remove <id or title>` or `/delete ...` | Remove a thread; retain at least one |
| `/clear` | Clear the current conversation history |
| `/trust [MODE [SECONDS]]` | Show or change this thread's approval settings |
| `/status` | Show workspace, model, trust, and observed token usage |

Ordinary threads are shared between CLI/TUI and development launchers. Each
launcher remembers its own selection. Idle TUIs refresh shared conversations.
History and summaries are stored locally under `.wallace/workspace_conversations/`;
`wallace_dev` uses `.wallace/conversations/`. Legacy threads migrate without
removing their original transcripts. Earlier transcript text does not grant
fresh source access.

## Approval modes

New threads default to `no-trust`. Modes persist per thread. Reviewed read-only
tools run without approval; file mutations, commands, tests, `check_source_file`,
and user-configured MCP tools follow the selected mode. New registered local
tools require approval unless explicitly reviewed as read-only.

| Mode | Behavior |
| --- | --- |
| `no-trust` | Wait for an explicit approve/reject decision |
| `informed-trust` | Show the action, then auto-approve after the countdown |
| `yes-trust` | Run approval-required actions without a prompt |

`/trust informed-trust 15` sets a 15-second countdown. Whole seconds from 1 to
300 are accepted; `/trust informed-trust` restores the five-second default.
Any key pauses the CLI countdown. In the TUI, keyboard or mouse interaction
pauses it until you decide. Y approves; N or Esc rejects. The filesystem sandbox
still applies in every mode; connected MCP services have their own permissions.

## Models and keys

Use `/model` or `/models` to list models. `/models ollama`, `/models openai`, and
`/models gemini` filter the list. Switch with `/model <ollama-name>`,
`/model openai:<model-id>`, or `/model gemini:<model-id>`. Model changes retain
the current conversation; the saved choice overrides the startup default.
Cloud listings reflect the account and can include models unsuitable for tools.

`/keys` lists provider key names and availability. `/key add <name> <openai|gemini>`
opens masked entry and stores the secret in the OS keyring. `/key on <name>`,
`/key off <name>`, and `/key remove <name>` manage saved keys; one per provider
can be active. `/key session` or `/key` enters a temporary key for the selected
cloud model. Saved keys require an unlocked keyring; metadata contains no key values.

Alternatively, set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in the host environment
or a private project `.env`. Existing environment values take precedence.
Environment keys appear as `environment-openai`/`environment-gemini`; they can
be toggled, but must be removed from their source rather than through `/key`.
Restart after `.env` changes. Enter secrets only in masked prompts or private
configuration. Cloud calls send relevant conversation/tool content to the
provider and may incur charges. `/status` counts reset on restart and report
observed tokens, not balances, rate-limit headroom, or usage missing from provider metadata.

Local settings live in [wallace_config.py](wallace_config.py): `OLLAMA_MODEL`,
`OLLAMA_BASE_URL`, `WALLACE_NUM_CTX` (default 32768), `WALLACE_NUM_PREDICT`
(default 4096), and `WALLACE_KEEP_ALIVE` (default `15m`).
`WALLACE_REASONING=auto` probes Ollama model capabilities once per model and keeps
supported thinking separate from final text. Use `true`, `false`, or `default`
to override it. Compact memory summaries disable thinking when supported.

## MCP services

The read-only `langchain-docs` MCP integration is enabled by default. It can be
disabled but not removed. Local tools remain usable when discovery fails.

| Command | Purpose |
| --- | --- |
| `/mcp` | List endpoints, enabled choices, discovery status, and tools |
| `/mcp add <name> <URL> [keywords...]` | Add an HTTP server |
| `/mcp add <name> --stdio <command> [args...]` | Add a local MCP process |
| `/mcp on\|off\|toggle\|remove <name>` | Manage a server |
| `/mcp auth <name> <ENV_NAME>` | Reference a host variable for an HTTP bearer token |
| `/mcp connect <name>` | Retry an enabled server |
| `/mcp reload` | Reload/retry configured connections |

Remote endpoints require HTTPS; HTTP is accepted on loopback. Credential-bearing
URLs, query strings, and fragments are rejected. Configuration lives in
`.wallace/mcp_servers.json`. Stdio `env` mappings must reference host variables
such as `${TOKEN}`, rather than contain literal credentials. `/key` configures
model providers, not MCP authentication. Interactive OAuth is not implemented.

“Connected” means the latest discovery succeeded. Server changes rebuild tools
while preserving the conversation; duplicate tool names are skipped. Configured
MCP calls follow thread trust settings. MCP processes/services operate outside
Wallace's local workspace sandbox; enable services with suitable access.

## TUI controls and direct commands

| Shortcut or command | Action |
| --- | --- |
| Enter / Esc | Send / cancel the current request |
| Ctrl+T | Toggle thread panel |
| F5 or `/edit [relative path]` | Pick/open a safe workspace file in Neovim, Vim, or vi |
| F6; F7/F8 | Toggle workspace panel; narrow/widen its display |
| F9 or `/theme [name]` | Choose a remembered UI theme |
| F10 / F11 / F12 | Models / MCP / provider keys |
| Ctrl+L / Ctrl+Q | Clear current history / quit |
| `/run <command>` | Run a scoped noninteractive command directly |

The TUI input supports thread-specific Up/Down history, selection, clipboard
shortcuts, word movement/deletion, and undo/redo. Mouse-selected chat text can be
copied with Ctrl+C. The external editor pauses the UI; save and exit normally.
Themes include Wallace palettes, terminal ANSI palettes, Textual themes, and
installed Omarchy palettes when present; selecting one does not alter the desktop.

`/run` also works in the CLI. It bypasses an extra model approval because you
entered the command yourself. It uses the same 30-second sandbox, can read Git
configuration, and does not send output to the model or save a conversation turn.
Each command starts in the selected workspace; `cd` does not persist.

## Context, execution, and validation

The compact behavior prompt in [wallace_context.py](wallace_context.py) is shared
by both interfaces. [ethics_manifesto.md](ethics_manifesto.md) remains unchanged
and is loaded as standing guidance for every agent, including ordinary threads.
Missing/invalid guidance produces a startup warning and a brief fallback.

A stable core exposes file exploration/editing, commands, history search, and
`discover_tools`. Optional checks, dependency tools, system information, and MCP
capabilities can be discovered by name/description. An empty discovery query
browses the catalog; returned capabilities become callable on the next model step.
`search_conversation` retrieves older details only from the current thread.
History and summaries are bounded; past assistant claims remain unverified
context. Full local transcripts are retained separately.

CLI and TUI use one `TurnRunner` for approvals, streaming, retries, and observed
results. Plans under `.wallace/plans/` checkpoint tool outcomes after results
arrive, including interrupted turns. Fresh requests do not inherit the full old
tool transcript. Repeated identical calls are stopped; malformed calls or missing
final text get bounded recovery. Checkpoints reduce repeated work but do not
guarantee exactly-once execution: after a crash or uncertain external result,
inspect the real state before retrying a mutation.
If the model stops before an observed work stage is complete, the shared runner
can request up to two concrete continuations. Rejected approvals disable that
automatic continuation. Ordinary text artifacts use post-write readback;
source files need matching code checks. Unsupported test-success claims are
replaced with the checks Wallace actually observed.

Reads, search results, previews, and command output have size limits. Oversized
single lines are shortened with an extraction hint. Exact edits require a unique
anchor, validate Python before writing, and reject changed contents. Shell tools
have a 30-second limit; fixed source checks use 60 seconds and tests 180 seconds.
A zero-test discovery is reported as unfinished verification.

Python, JS/TS, HTML/CSS, shell, JSON/TOML, Go, and Rust have source checks;
Python, Deno/npm, Go, and Cargo have test runners. Optional runtimes/dependencies
must already be installed. Deno checks use cached dependencies. Syntax/type
checks establish only their source scope; HTML/CSS parsing does not test a
browser. Passing tests apply to their recorded targets and do not prove every
requested behavior. A plan marked complete is evidence bookkeeping, not a
functional guarantee. Workspace-authored tools are not automatically registered.

## Development and checks

```bash
uv run python -B -m unittest discover -s tests -q
```

Tests cover core behavior, providers, and the TUI without live model calls.
New built-in tools belong in `tools.py` and need relevant contract tests;
`verify_tool_contract` checks invocation, assertions, execution, and registration.
Use the development reload or restart before calling a newly registered tool.

The main modules are `main.py`/`tui_agent.py` (interfaces), `tools.py` (local tools),
`wallace_execution.py` (shared turn execution), `wallace_context.py` (prompt),
`wallace_memory.py` (thread memory), `wallace_tool_selection.py` (discovery),
`wallace_config.py` (settings), `wallace_models.py`/`wallace_keys.py` (providers),
`wallace_mcp.py` (connections), and `wallace_planning/` (durable plans/threads).
