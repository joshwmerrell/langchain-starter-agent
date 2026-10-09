"""Textual TUI front end for Wallace.

Run with:  python tui_agent.py
Keys:      Enter = send, Esc = cancel current request, Ctrl+L = clear log, Ctrl+Q = quit
           Select chat text with the mouse and press Ctrl+C; Input supports Ctrl+C/Ctrl+V.
Approvals: y = approve, n / Esc = reject (when the agent wants to run a command or change a file)
"""

import asyncio
from fnmatch import fnmatch
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.selection import Selection
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    RichLog,
    Static,
)

from main import (
    assistant_response_text,
    describe_error,
    extract_action_requests,
    format_action_request,
    is_malformed_tool_call_error,
    make_decision,
    MALFORMED_TOOL_CALL_RETRY_PROMPT,
    resume_command,
    setup_agent,
)
from tools import SAFE_ROOT, SENSITIVE_PATTERNS, SKIP_DIRS


WALLACE_THEME = {
    "primary": "#7dd3fc",
    "secondary": "#c4b5fd",
    "success": "#86efac",
    "warning": "#fcd34d",
    "error": "#fb7185",
    "user": "#93c5fd",
    "assistant": "#c4b5fd",
    "muted": "#94a3b8",
}


def _short(value: object, limit: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_tool_call(call: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 40)!r}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('name', 'tool')}({args})"


def _styled_detail(detail: str) -> Text:
    """Colour added (+), removed (-) and hunk (@) lines of an approval preview."""
    styles = {
        "+": WALLACE_THEME["success"],
        "-": WALLACE_THEME["error"],
        "@": WALLACE_THEME["primary"],
    }
    text = Text()
    for line in detail.splitlines():
        text.append(line + "\n", style=styles.get(line[:1], ""))
    return text


def _workspace_tree(max_depth: int = 3, max_entries: int = 140) -> Text:
    """Build a bounded, secret-filtered tree for the workspace panel."""
    text = Text()
    count = 0

    def allowed(path: Path) -> bool:
        parts = path.relative_to(SAFE_ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            return False
        return not any(
            fnmatch(part.lower(), pattern)
            for part in parts
            for pattern in SENSITIVE_PATTERNS
        )

    def visit(directory: Path, prefix: str, depth: int) -> None:
        nonlocal count
        if count >= max_entries:
            return
        try:
            entries = sorted(
                (entry for entry in directory.iterdir() if allowed(entry.resolve())),
                key=lambda entry: (entry.is_file(), entry.name.lower()),
            )
        except OSError:
            return

        for index, entry in enumerate(entries):
            if count >= max_entries:
                text.append(f"{prefix}└── … more entries\n", style=WALLACE_THEME["muted"])
                return
            count += 1
            connector = "└── " if index == len(entries) - 1 else "├── "
            if entry.is_dir():
                text.append(f"{prefix}{connector}{entry.name}/\n", style=WALLACE_THEME["primary"])
                if depth < max_depth:
                    child_prefix = prefix + ("    " if index == len(entries) - 1 else "│   ")
                    visit(entry, child_prefix, depth + 1)
            else:
                text.append(f"{prefix}{connector}{entry.name}\n")

    text.append(f"{SAFE_ROOT.name}/\n", style=f"bold {WALLACE_THEME['primary']}")
    visit(SAFE_ROOT, "", 1)
    if count >= max_entries:
        text.append("… tree truncated\n", style=WALLACE_THEME["muted"])
    return text


class WorkspacePanel(Vertical):
    """Live, read-only view of the agent's project workspace."""

    def compose(self) -> ComposeResult:
        yield Static("WORKSPACE", id="workspace_title")
        with VerticalScroll(id="workspace_tree_scroll"):
            yield Static(id="workspace_tree")
        yield Static("ACTIVITY", id="workspace_activity_title")
        yield Static("Waiting for work", id="workspace_activity")

    def refresh_view(self, activity: str | None = None) -> None:
        self.query_one("#workspace_tree", Static).update(_workspace_tree())
        if activity is not None:
            self.query_one("#workspace_activity", Static).update(activity)


class SelectableRichLog(RichLog):
    """RichLog with clipboard extraction for Textual's mouse selection."""

    ALLOW_SELECT = True

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        # RichLog stores rendered output as Strips rather than child widgets.
        # Textual can highlight those lines, but needs this hook to extract
        # the selected text for Ctrl+C.
        text = "\n".join(line.text for line in self.lines)
        return selection.extract(text).rstrip(), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        self.refresh()

    def render_line(self, y: int):
        """Render with cell offsets so Textual can map mouse positions."""
        scroll_x, scroll_y = self.scroll_offset
        absolute_y = scroll_y + y
        line = self._render_line(
            absolute_y,
            scroll_x,
            self.scrollable_content_region.width,
        )
        strip = line.apply_style(self.rich_style)
        selection = self.text_selection
        if selection is not None:
            span = selection.get_span(absolute_y)
            if span is not None:
                start, end = span
                if end == -1:
                    end = strip.cell_length
                start -= scroll_x
                end -= scroll_x
                start = max(0, start)
                end = min(strip.cell_length, end)
                if end > start:
                    selection_style = self.screen.get_component_rich_style(
                        "screen--selection"
                    )
                    strip = Strip.join(
                        (
                            strip.crop(0, start),
                            strip.crop(start, end).apply_style(selection_style),
                            strip.crop(end),
                        )
                    )
        return strip.apply_offsets(scroll_x, absolute_y)


class ApprovalScreen(ModalScreen[bool]):
    """Ask the user to approve or reject a risky tool call."""

    DEFAULT_CSS = f"""
    ApprovalScreen {{
        align: center middle;
    }}
    #dialog {{
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick {WALLACE_THEME["primary"]};
        background: $surface;
        padding: 1 2;
    }}
    #approval_title {{
        text-style: bold;
        margin-bottom: 1;
    }}
    #approval_detail {{
        height: auto;
        max-height: 20;
        margin-bottom: 1;
    }}
    #approval_buttons {{
        height: auto;
        align-horizontal: right;
    }}
    #approval_buttons Button {{
        margin-left: 2;
    }}
    #approve {{
        color: {WALLACE_THEME["success"]};
        border: solid {WALLACE_THEME["success"]};
    }}
    #reject {{
        color: {WALLACE_THEME["error"]};
        border: solid {WALLACE_THEME["error"]};
    }}
    """

    BINDINGS = [
        ("y", "approve", "Approve"),
        ("n", "reject", "Reject"),
        ("escape", "reject", "Reject"),
    ]

    def __init__(self, detail: str) -> None:
        super().__init__()
        self.detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("The agent wants to run this action", id="approval_title")
            with VerticalScroll(id="approval_detail"):
                yield Static(_styled_detail(self.detail))
            with Horizontal(id="approval_buttons"):
                yield Button("Reject (n)", variant="error", id="reject")
                yield Button("Approve (y)", variant="success", id="approve")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class AgentTUI(App):
    TITLE = "Wallace"
    # Keep arbitrary text selection enabled so RichLog text can be selected
    # with the mouse and copied with the screen's built-in Ctrl+C action.
    ALLOW_SELECT = True

    CSS = f"""
    #chat_log {{
        height: 1fr;
        border: solid {WALLACE_THEME["primary"]};
        margin: 1;
    }}
    #main_layout {{
        height: 1fr;
    }}
    #chat_column {{
        width: 1fr;
    }}
    #workspace_panel {{
        width: 30%;
        min-width: 24;
        max-width: 42;
        border-left: solid {WALLACE_THEME["secondary"]};
        padding: 0 1;
    }}
    #workspace_title, #workspace_activity_title {{
        height: 1;
        text-style: bold;
        color: {WALLACE_THEME["primary"]};
    }}
    #workspace_tree_scroll {{
        height: 1fr;
        scrollbar-size: 1 1;
    }}
    #workspace_tree {{
        width: 1fr;
        color: $text;
    }}
    #workspace_activity_title {{
        border-top: solid {WALLACE_THEME["secondary"]};
        padding-top: 1;
    }}
    #workspace_activity {{
        height: auto;
        max-height: 5;
        color: {WALLACE_THEME["muted"]};
    }}
    #thinking {{
        display: none;
        height: 3;
        margin: 0 1;
        padding: 0 1;
        border: round {WALLACE_THEME["secondary"]};
        background: $surface;
        color: {WALLACE_THEME["primary"]};
        content-align: left middle;
    }}
    #prompt {{
        margin: 0 1 1 1;
        border: round {WALLACE_THEME["primary"]};
        &:focus {{
            border: round {WALLACE_THEME["secondary"]};
        }}
    }}
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("f6", "toggle_workspace", "Workspace"),
        ("f7", "narrow_workspace", "Narrow workspace"),
        ("f8", "widen_workspace", "Widen workspace"),
        ("escape", "cancel", "Cancel request"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.thread_config = None
        self.workspace_panel = None
        self._workspace_width = 30

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main_layout"):
            with Vertical(id="chat_column"):
                yield SelectableRichLog(id="chat_log", highlight=True, markup=True, wrap=True)
                yield Static(id="thinking")
                yield Input(placeholder="Starting Wallace...", id="prompt", disabled=True)
            yield WorkspacePanel(id="workspace_panel")
        yield Footer()

    def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat_log", RichLog)
        self.prompt_input = self.query_one("#prompt", Input)
        self.thinking_panel = self.query_one("#thinking", Static)
        self.workspace_panel = self.query_one("#workspace_panel", WorkspacePanel)
        self.workspace_panel.refresh_view()
        self.set_interval(2.0, self._refresh_workspace)
        self._thinking_messages = (
            "Mapping the next move",
            "Reading the signals",
            "Assembling the answer",
            "Checking the details",
            "Making the tools earn their keep",
            "Turning the crank",
            "Connecting the useful pieces",
            "Putting the finishing touches on it",
        )
        self._thinking_index = 0
        self._thinking_spinner_index = 0
        self._thinking_spinners = (
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
        )
        self._busy = False
        self._rotate_thinking_status = False
        self.set_interval(0.35, self._advance_thinking_status)
        self.chat_log.write(
            Text("Welcome to Wallace.", style=f"bold {WALLACE_THEME['success']}")
        )
        self.chat_log.write(
            "Wallace is being built to help with creative, practical, "
            "productivity, and computing tasks, across spoken and computing "
            "languages, with voice interaction. This TUI currently uses text; "
            "general web, Google Workspace, and microphone/speaker access "
            "require integrations that are not configured yet."
        )
        self.chat_log.write(
            "Tip: select chat text with the mouse and press Ctrl+C to copy it. "
            "The input bar supports normal Ctrl+C/Ctrl+V shortcuts."
        )
        self.chat_log.write(
            Text("Initializing Wallace...", style=WALLACE_THEME["warning"])
        )
        self.initialize_agent()

    # ---------- setup ----------

    @work(exclusive=True, group="setup")
    async def initialize_agent(self) -> None:
        try:
            self.agent, self.thread_config = await setup_agent(on_status=self._write_warning)
        except Exception as error:
            self.chat_log.write(
                Text(f"Failed to start agent: {error}", style=WALLACE_THEME["error"])
            )
            self.prompt_input.placeholder = "Wallace unavailable. Press Ctrl+Q to quit."
            return

        self.chat_log.write(
            Text("Wallace is ready.", style=f"bold {WALLACE_THEME['success']}")
        )
        self._set_busy(False)

    # ---------- input handling ----------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.prompt_input.disabled = busy
        self.prompt_input.placeholder = (
            "Wallace is working... (Esc to cancel)"
            if busy
            else "Ask Wallace something (Enter to send)"
        )
        self.thinking_panel.display = busy
        if busy:
            self._thinking_index = 0
            self._thinking_spinner_index = 0
            self._set_thinking_status(rotate=True)
        if not busy:
            self._rotate_thinking_status = False
            self.prompt_input.focus()

    def _set_thinking_status(self, *, rotate: bool, message: str | None = None) -> None:
        self._rotate_thinking_status = rotate
        if message is None:
            message = self._thinking_messages[self._thinking_index]
        self.thinking_panel.update(
            Text(
                f"WALLACE  /  {self._thinking_spinners[self._thinking_spinner_index]}  {message}",
                style=f"bold {WALLACE_THEME['primary']}",
            )
        )

    def _advance_thinking_status(self) -> None:
        if not self._busy or not self._rotate_thinking_status:
            return
        self._thinking_spinner_index = (
            self._thinking_spinner_index + 1
        ) % len(self._thinking_spinners)
        if self._thinking_spinner_index == 0:
            self._thinking_index = (
                self._thinking_index + 1
            ) % len(self._thinking_messages)
        self._set_thinking_status(rotate=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        self.prompt_input.value = ""

        if not prompt or self.agent is None:
            return
        if prompt.lower() == "exit":
            self.exit()
            return

        user_line = Text()
        user_line.append("\nUser: ", style=f"bold {WALLACE_THEME['user']}")
        user_line.append(prompt)
        self.chat_log.write(user_line)

        self._set_busy(True)
        self.run_turn(prompt)

    # ---------- agent turn ----------

    @work(exclusive=True, group="turn")
    async def run_turn(self, prompt: str) -> None:
        turn_messages = []
        payload = {"messages": [{"role": "user", "content": prompt}]}
        recovery_attempts = 0
        try:
            while True:
                interrupts = []
                try:
                    async for update in self.agent.astream(
                        payload, self.thread_config, stream_mode="updates"
                    ):
                        for key, output in update.items():
                            if key == "__interrupt__":
                                interrupts.extend(output)
                                continue
                            messages = output.get("messages") if isinstance(output, dict) else None
                            if not isinstance(messages, list):
                                continue
                            for msg in messages:
                                turn_messages.append(msg)
                                self._show_progress(msg)
                except Exception as error:
                    if recovery_attempts < 1 and is_malformed_tool_call_error(error):
                        recovery_attempts += 1
                        self._write_warning(
                            "The model returned malformed tool-call JSON; "
                            "retrying once with a concise tool instruction."
                        )
                        payload = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": MALFORMED_TOOL_CALL_RETRY_PROMPT,
                                }
                            ]
                        }
                        continue
                    raise

                requests = extract_action_requests(interrupts)
                if not requests:
                    break

                # The agent paused for approval: ask, then resume the same run.
                decisions = []
                for request in requests:
                    self._set_thinking_status(
                        rotate=False,
                        message="Waiting for your approval",
                    )
                    approved = await self._ask_approval(request)
                    decisions.append(make_decision(approved))
                    self.chat_log.write(
                        Text("  ✓ approved", style=WALLACE_THEME["success"])
                        if approved
                        else Text("  ✗ rejected", style=WALLACE_THEME["error"])
                    )
                self._thinking_index = 0
                self._set_thinking_status(rotate=True)
                payload = resume_command(decisions)

            response = assistant_response_text(turn_messages)
            if response is None:
                self.chat_log.write(
                    Text(
                        "The agent finished without a text response.",
                        style=WALLACE_THEME["error"],
                    )
                )
            else:
                self.chat_log.write(
                    Text("AI:", style=f"bold {WALLACE_THEME['assistant']}")
                )
                self.chat_log.write(Markdown(response))

        except Exception as error:
            self._write_warning(describe_error(error))
        finally:
            self._set_busy(False)

    async def _ask_approval(self, request: dict) -> bool:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.push_screen(
            ApprovalScreen(format_action_request(request)),
            lambda result: future.done() or future.set_result(bool(result)),
        )
        return await future

    def _show_progress(self, msg) -> None:
        """Show tool activity as it happens so a slow local model doesn't look hung."""
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name", "tool")
            self._set_thinking_status(rotate=False, message=f"Using {name}")
            self.chat_log.write(
                Text(
                    f"  → {_format_tool_call(call)}",
                    style=f"dim {WALLACE_THEME['primary']}",
                )
            )
            if self.workspace_panel is not None:
                self.workspace_panel.refresh_view(f"Running {name}")
        if getattr(msg, "type", None) == "tool":
            name = getattr(msg, "name", None) or "tool"
            self._thinking_index = 0
            self._set_thinking_status(
                rotate=True,
                message=f"Reviewing {name} results",
            )
            failed = getattr(msg, "status", None) == "error" or str(msg.content).startswith("Error")
            if failed:
                self.chat_log.write(
                    Text(f"  ✗ {name} failed", style=f"dim {WALLACE_THEME['error']}")
                )
            else:
                self.chat_log.write(
                    Text(
                        f"  ✓ {name} finished",
                        style=f"dim {WALLACE_THEME['success']}",
                    )
                )
            if self.workspace_panel is not None:
                self.workspace_panel.refresh_view(f"Finished {name}")

    def _write_warning(self, message: str) -> None:
        self.chat_log.write(Text(message, style=WALLACE_THEME["warning"]))
        if self.workspace_panel is not None:
            self.workspace_panel.refresh_view(message)

    # ---------- actions ----------

    def action_clear_log(self) -> None:
        self.chat_log.clear()

    def _refresh_workspace(self) -> None:
        if self.workspace_panel is not None and self.workspace_panel.display:
            self.workspace_panel.refresh_view()

    def action_toggle_workspace(self) -> None:
        if self.workspace_panel is None:
            return
        self.workspace_panel.display = not self.workspace_panel.display

    def action_narrow_workspace(self) -> None:
        if self.workspace_panel is None:
            return
        self._workspace_width = max(20, self._workspace_width - 5)
        self.workspace_panel.styles.width = f"{self._workspace_width}%"

    def action_widen_workspace(self) -> None:
        if self.workspace_panel is None:
            return
        self._workspace_width = min(45, self._workspace_width + 5)
        self.workspace_panel.styles.width = f"{self._workspace_width}%"

    def action_cancel(self) -> None:
        if self.prompt_input.disabled and self.agent is not None:
            self.workers.cancel_group(self, "turn")
            self.chat_log.write(
                Text("Request cancelled.", style=WALLACE_THEME["warning"])
            )


if __name__ == "__main__":
    AgentTUI().run()
