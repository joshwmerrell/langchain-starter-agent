"""Textual TUI front end for Wallace.

Run with:  python tui_agent.py
Keys:      Enter = send, Esc = cancel current request, Ctrl+L = clear log, Ctrl+Q = quit
           Select chat text with the mouse and press Ctrl+C; Input supports Ctrl+C/Ctrl+V.
Threads:   /threads, /new [title], /switch <id or title>, /rename <title>, /remove <id or title>, /clear
Themes:    F9 or /theme to choose; /theme <name> to apply directly
Approvals: y = approve, n / Esc = reject (when the agent wants to run a command or change a file)
"""

import asyncio
import os
import re
import shlex
import tempfile
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Span, Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
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
    ConversationStore,
    EMPTY_RESPONSE_RETRY_PROMPT,
    describe_error,
    extract_action_requests,
    format_action_request,
    is_malformed_tool_call_error,
    malformed_retry_payload,
    make_decision,
    model_tool_activity,
    new_thread_config,
    no_text_status,
    RepetitiveResponseError,
    response_quality_error,
    resume_command,
    setup_agent,
)
from tools import SAFE_ROOT, SENSITIVE_PATTERNS, SKIP_DIRS
from wallace_planning import (
    ArchiveManager,
    ConversationThreadManager,
    planning_context,
    resume_or_create,
    save_plan,
)
from wallace_themes import (
    DEFAULT_THEME,
    WALLACE_PRESETS,
    discover_omarchy_themes,
    load_theme_choice,
    rich_palette,
    save_theme_choice,
)


def _write_reload_status(status: str) -> None:
    status_path = os.environ.get("WALLACE_RELOAD_STATUS")
    if not status_path:
        return
    try:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(status)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except OSError:
        pass


WALLACE_THEME = rich_palette(WALLACE_PRESETS[0])
RED_LIGHT_STYLE = Style(color="#d80000", bgcolor="#050000")


def _overlay_strip_style(strip: Strip, style: Style) -> Strip:
    """Apply a foreground/background style over existing Rich segment colors."""
    return Strip(
        (
            Segment(segment.text, (segment.style or Style()) + style, segment.control)
            for segment in strip
        ),
        strip.cell_length,
    )


def _short(value: object, limit: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_tool_call(call: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 40)!r}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('name', 'tool')}({args})"


class HistoryInput(Input):
    """Input widget with per-thread Up/Down command history."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._setting_history_value = False
        self._history_event_value: str | None = None

    def load_history(self, values: list[str]) -> None:
        self._input_history = []
        for value in values:
            self.remember_input(value)
        self._history_index = None
        self._history_draft = ""
        self._history_event_value = None

    def remember_input(self, value: str) -> None:
        value = value.strip()
        if not value:
            return
        if not self._input_history or self._input_history[-1] != value:
            self._input_history.append(value)
        # Keep navigation bounded independently of the visible conversation log.
        self._input_history = self._input_history[-100:]
        self._history_index = None
        self._history_event_value = None

    def _set_history_value(self, value: str) -> None:
        # Textual may deliver Input.Changed after this method returns. Keep the
        # selected value identifiable until that event is handled, otherwise
        # on_input_changed would reset navigation after every arrow press.
        self._history_event_value = value
        self._setting_history_value = True
        try:
            self.value = value
            self.cursor_position = len(value)
        finally:
            self._setting_history_value = False

    async def _on_key(self, event: Key) -> None:
        if event.key == "up":
            if not self._input_history:
                await super()._on_key(event)
                return
            event.stop()
            event.prevent_default()
            if self._history_index is None:
                self._history_draft = self.value
                self._history_index = len(self._input_history)
            self._history_index = max(0, self._history_index - 1)
            self._set_history_value(self._input_history[self._history_index])
        elif event.key == "down":
            if self._history_index is None:
                await super()._on_key(event)
                return
            event.stop()
            event.prevent_default()
            next_index = self._history_index + 1
            if next_index >= len(self._input_history):
                self._history_index = None
                self._set_history_value(self._history_draft)
            else:
                self._history_index = next_index
                self._set_history_value(self._input_history[next_index])
            return

        await super()._on_key(event)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._setting_history_value:
            return
        if self._history_event_value is not None and event.value == self._history_event_value:
            self._history_event_value = None
            return
        self._history_event_value = None
        self._history_index = None


def _tool_call_key(call: dict) -> tuple:
    """Stable key for repeated streaming representations of one tool call."""
    call_id = call.get("id") or call.get("tool_call_id")
    if call_id:
        return ("id", str(call_id))
    args = call.get("args") or {}
    return ("signature", str(call.get("name", "tool")), repr(sorted(args.items())))


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


def _allowed_workspace_path(path: Path) -> bool:
    """Whether a resolved workspace path is safe to show or track."""
    try:
        parts = path.relative_to(SAFE_ROOT).parts
    except ValueError:
        return False
    if any(part in SKIP_DIRS for part in parts):
        return False
    return not any(
        fnmatch(part.lower(), pattern)
        for part in parts
        for pattern in SENSITIVE_PATTERNS
    )


def _workspace_file_state() -> dict[str, tuple[int, int]]:
    """Return safe workspace files mapped to (size, mtime_ns)."""
    state: dict[str, tuple[int, int]] = {}
    for root, directories, files in os.walk(SAFE_ROOT):
        directories[:] = sorted(
            directory for directory in directories if directory not in SKIP_DIRS
        )
        for name in files:
            path = (Path(root) / name).resolve()
            if not _allowed_workspace_path(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(SAFE_ROOT).as_posix()
            state[relative] = (stat.st_size, stat.st_mtime_ns)
    return state


def _workspace_tree(max_depth: int = 3, max_entries: int = 140) -> Text:
    """Build a bounded, secret-filtered tree for the workspace panel."""
    text = Text()
    count = 0

    def allowed(path: Path) -> bool:
        return _allowed_workspace_path(path)

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._file_state: dict[str, tuple[int, int]] | None = None
        self._pending_changes: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("WORKSPACE", id="workspace_title")
        with VerticalScroll(id="workspace_tree_scroll"):
            yield Static(id="workspace_tree")
        yield Static("ACTIVITY", id="workspace_activity_title")
        yield Static("Waiting for work", id="workspace_activity")

    def refresh_view(self, activity: str | None = None) -> list[str]:
        current_state = _workspace_file_state()
        changes: list[str] = []
        if self._file_state is not None:
            for path in sorted(current_state.keys() - self._file_state.keys()):
                changes.append(f"added {path}")
            for path in sorted(self._file_state.keys() - current_state.keys()):
                changes.append(f"removed {path}")
            for path in sorted(current_state.keys() & self._file_state.keys()):
                if current_state[path] != self._file_state[path]:
                    changes.append(f"modified {path}")
        self._file_state = current_state
        if changes:
            self._pending_changes.extend(changes)
        self.query_one("#workspace_tree", Static).update(_workspace_tree())
        if activity is not None:
            self.query_one("#workspace_activity", Static).update(activity)
        elif changes:
            preview = ", ".join(changes[:3])
            if len(changes) > 3:
                preview += f" (+{len(changes) - 3} more)"
            self.query_one("#workspace_activity", Static).update(
                f"Workspace changed: {preview}"
            )
        return changes

    def consume_changes(self) -> list[str]:
        changes = list(dict.fromkeys(self._pending_changes))
        self._pending_changes.clear()
        return changes


class ThreadPanel(Vertical):
    """Visible, read-only index of the user's conversation threads."""

    def compose(self) -> ComposeResult:
        yield Static("THREADS", id="thread_title")
        with VerticalScroll(id="thread_list_scroll"):
            yield Static(id="thread_list")
        yield Static("/new  /switch  /rename  /remove", id="thread_hint")

    def refresh_threads(self, manager: ConversationThreadManager, current_id: str) -> None:
        text = Text()
        records = manager.list_threads()
        if not records:
            text.append("No threads yet\n", style=WALLACE_THEME["muted"])
        for record in records:
            marker = "● " if record.thread_id == current_id else "  "
            style = (
                f"bold {WALLACE_THEME['success']}"
                if record.thread_id == current_id
                else ""
            )
            text.append(f"{marker}{record.title}\n", style=style)
            text.append(f"   {record.thread_id}\n", style=WALLACE_THEME["muted"])
        self.query_one("#thread_list", Static).update(text)


class SelectableRichLog(RichLog):
    """RichLog with clipboard extraction for Textual's mouse selection."""

    ALLOW_SELECT = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._theme_entries: list[tuple[object, dict[str, str], tuple, dict]] = []
        self._rendering_deferred = False

    def on_resize(self, event) -> None:
        # RichLog replays early writes through self.write when it first learns
        # its width. Those writes are already in our theme replay list.
        self._rendering_deferred = True
        try:
            super().on_resize(event)
        finally:
            self._rendering_deferred = False

    def write(self, content: object, *args, **kwargs):
        if not self._rendering_deferred:
            saved = content.copy() if isinstance(content, Text) else content
            self._theme_entries.append((saved, dict(WALLACE_THEME), args, kwargs))
            self._theme_entries = self._theme_entries[-3000:]
        return super().write(content, *args, **kwargs)

    def clear(self):
        self._theme_entries.clear()
        return super().clear()

    def retheme(self) -> None:
        """Re-render existing entries so an active conversation changes too."""
        old_scroll_y = self.scroll_offset.y
        entries = list(self._theme_entries)
        super().clear()
        for content, old_palette, args, kwargs in entries:
            if isinstance(content, Text):
                content = content.copy()
                replacements = {
                    value.lower(): WALLACE_THEME[key]
                    for key, value in old_palette.items()
                    if value != "dim" and value != WALLACE_THEME[key]
                }
                if replacements:
                    pattern = re.compile(
                        r"(?<![A-Za-z0-9_])(?:"
                        + "|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True))
                        + r")(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )

                    def recolor(style):
                        if isinstance(style, str):
                            return pattern.sub(lambda match: replacements[match.group().lower()], style)
                        return style

                    content.style = recolor(content.style)
                    content.spans = [
                        Span(span.start, span.end, recolor(span.style))
                        for span in content.spans
                    ]
            super().write(content, *args, **kwargs)
        self.scroll_to(y=old_scroll_y, animate=False)

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
        if self.app.theme == "wallace-red-light":
            # Rich Markdown and syntax highlighting can supply their own blue
            # or green colors; keep even that content monochrome in red light.
            strip = _overlay_strip_style(strip, RED_LIGHT_STYLE)
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
                            _overlay_strip_style(strip.crop(start, end), selection_style),
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
        border: thick $primary;
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
    #approval_hint {{
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
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
            yield Static("Y = Yes   N = No   Esc = No", id="approval_hint")
            with Horizontal(id="approval_buttons"):
                yield Button("No (N)", variant="error", id="reject")
                yield Button("Yes (Y)", variant="success", id="approve")

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
        border: solid $primary;
        margin: 1;
    }}
    #main_layout {{
        height: 1fr;
    }}
    #thread_panel {{
        width: 24%;
        min-width: 22;
        max-width: 34;
        border-right: solid $secondary;
        padding: 0 1;
    }}
    #thread_title {{
        height: 1;
        text-style: bold;
        color: $primary;
    }}
    #thread_list_scroll {{
        height: 1fr;
        scrollbar-size: 1 1;
    }}
    #thread_list {{
        width: 1fr;
        color: $text;
    }}
    #thread_hint {{
        height: 2;
        color: $text-muted;
    }}
    #chat_column {{
        width: 1fr;
    }}
    #workspace_panel {{
        width: 30%;
        min-width: 24;
        max-width: 42;
        border-left: solid $secondary;
        padding: 0 1;
    }}
    #workspace_title, #workspace_activity_title {{
        height: 1;
        text-style: bold;
        color: $primary;
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
        border-top: solid $secondary;
        padding-top: 1;
    }}
    #workspace_activity {{
        height: auto;
        max-height: 5;
        color: $text-muted;
    }}
    #thinking {{
        display: none;
        height: 3;
        margin: 0 1;
        padding: 0 1;
        border: round $secondary;
        background: $surface;
        color: $primary;
        content-align: left middle;
    }}
    #prompt {{
        margin: 0 1 1 1;
        border: round $primary;
        &:focus {{
            border: round $secondary;
        }}
    }}
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("ctrl+t", "toggle_threads", "Threads"),
        ("f6", "toggle_workspace", "Workspace"),
        Binding("f7", "narrow_workspace", "Narrow workspace", show=False),
        Binding("f8", "widen_workspace", "Widen workspace", show=False),
        ("f9", "change_theme", "Themes"),
        Binding("escape", "cancel", "Cancel request", show=False),
    ]

    def __init__(self) -> None:
        self._theme_ready = False
        super().__init__()
        for theme in (*WALLACE_PRESETS, *discover_omarchy_themes()):
            self.register_theme(theme)
        saved_theme = load_theme_choice(SAFE_ROOT)
        self.theme = saved_theme if self.get_theme(saved_theme) else DEFAULT_THEME
        self._last_persisted_theme = self.theme
        WALLACE_THEME.update(rich_palette(self.current_theme))
        self.agent = None
        self.thread_config = None
        self.workspace_panel = None
        self._workspace_width = 30
        self.thread_manager = ConversationThreadManager(SAFE_ROOT)
        self.thread_id = self.thread_manager.ensure("main").thread_id
        self.conversation_store = self.thread_manager.store(self.thread_id)
        self._conversation_history: list[dict[str, str]] = []
        self._active_plan = None

    def watch_theme(self, theme_name: str) -> None:
        theme = self.get_theme(theme_name)
        if theme is None:
            return
        WALLACE_THEME.update(rich_palette(theme))
        if not getattr(self, "_theme_ready", False):
            return
        self.chat_log.retheme()
        self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
        self.workspace_panel.refresh_view()
        if self._busy:
            self._set_thinking_status(rotate=self._rotate_thinking_status)
        if theme_name != self._last_persisted_theme:
            try:
                save_theme_choice(SAFE_ROOT, theme_name)
                self._last_persisted_theme = theme_name
            except OSError as error:
                self._write_warning(f"Could not save theme choice: {error}")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main_layout"):
            yield ThreadPanel(id="thread_panel")
            with Vertical(id="chat_column"):
                yield SelectableRichLog(id="chat_log", max_lines=3000, min_width=1, highlight=False, markup=True, wrap=True)
                yield Static(id="thinking")
                yield HistoryInput(placeholder="Starting Wallace...", id="prompt", disabled=True)
            yield WorkspacePanel(id="workspace_panel")
        yield Footer()

    def on_mount(self) -> None:
        ArchiveManager(SAFE_ROOT).run_at_session_boundary()
        self.chat_log = self.query_one("#chat_log", RichLog)
        self.prompt_input = self.query_one("#prompt", HistoryInput)
        self.thinking_panel = self.query_one("#thinking", Static)
        self.workspace_panel = self.query_one("#workspace_panel", WorkspacePanel)
        self.thread_panel = self.query_one("#thread_panel", ThreadPanel)
        self._theme_ready = True
        self._update_header_datetime()
        self.set_interval(15.0, self._update_header_datetime)
        self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
        self.workspace_panel.refresh_view()
        self.set_interval(2.0, self._refresh_workspace)
        self._conversation_history = self.conversation_store.load()
        self.prompt_input.load_history(
            [
                message["content"]
                for message in self._conversation_history
                if message["role"] == "user"
            ]
        )
        pending_prompt = self.conversation_store.load_pending()
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
        _write_reload_status("busy")
        self._rotate_thinking_status = False
        self.set_interval(0.35, self._advance_thinking_status)
        self.chat_log.write(Text("Welcome to Wallace.", style=f"bold {WALLACE_THEME['success']}"))
        self.chat_log.write(
            f"Thread: {self.thread_id}. /threads /new /switch /rename /remove /clear /theme"
        )
        self._restore_conversation()
        if pending_prompt:
            self.chat_log.write(
                Text(
                    "The previous request was interrupted before completion. "
                    "Ask Wallace to continue it when ready.",
                    style=WALLACE_THEME["warning"],
                )
            )
        self.chat_log.write(
            Text("Initializing Wallace...", style=WALLACE_THEME["warning"])
        )
        self.initialize_agent()

    def _restore_conversation(self) -> None:
        if not self._conversation_history:
            return
        self.chat_log.write(
            Text(
                f"Restored {sum(message['role'] == 'assistant' for message in self._conversation_history)} previous turn(s).",
                style=WALLACE_THEME["success"],
            )
        )
        for message in self._conversation_history:
            if message["role"] == "user":
                line = Text()
                line.append("\nUser: ", style=f"bold {WALLACE_THEME['user']}")
                line.append(message["content"])
                self.chat_log.write(line)
            else:
                self.chat_log.write(
                    Text("AI:", style=f"bold {WALLACE_THEME['assistant']}")
                )
                if response_quality_error(message["content"]):
                    self.chat_log.write(Text(
                        "Previous AI reply omitted from display: repetitive model "
                        "output. The saved conversation is unchanged.",
                        style=WALLACE_THEME["warning"],
                    ))
                else:
                    self.chat_log.write(Markdown(message["content"]))

    # ---------- setup ----------

    @work(exclusive=True, group="setup")
    async def initialize_agent(self) -> None:
        try:
            self.agent, self.thread_config = await setup_agent(on_status=self._write_warning)
        except Exception as error:
            self.chat_log.write(
                Text(f"Failed to start agent: {error}", style=WALLACE_THEME["error"])
            )
            # A failed setup must not leave dev_tui.py waiting forever for an
            # agent that will never become idle.
            self._set_busy(False)
            self.prompt_input.disabled = True
            self.prompt_input.placeholder = "Wallace unavailable. Press Ctrl+Q to quit."
            return

        self.chat_log.write(
            Text("Wallace is ready.", style=f"bold {WALLACE_THEME['success']}")
        )
        self._set_busy(False)

    # ---------- input handling ----------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        _write_reload_status("busy" if busy else "idle")
        self.prompt_input.disabled = busy
        self.prompt_input.placeholder = (
            "Wallace is working... (Esc to cancel)"
            if busy
            else "Ask Wallace something (Enter to send; ↑/↓ history)"
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

        if not prompt:
            return
        if prompt.lower() == "exit":
            self.exit()
            return
        if prompt.startswith("/"):
            self._handle_thread_command(prompt)
            return
        if self.agent is None:
            return

        self.conversation_store.save_pending(prompt)
        self._conversation_history = self.conversation_store.trim(
            [*self._conversation_history, {"role": "user", "content": prompt}]
        )
        self.prompt_input.remember_input(prompt)
        self.conversation_store.save(self._conversation_history)

        user_line = Text()
        user_line.append("\nUser: ", style=f"bold {WALLACE_THEME['user']}")
        user_line.append(prompt)
        self.chat_log.write(user_line)

        self._set_busy(True)
        agent_prompt = prompt
        self._active_plan = None
        try:
            self._active_plan = resume_or_create(
                SAFE_ROOT, prompt, session_id=self.thread_id
            )
            agent_prompt = planning_context(self._active_plan) + "\nUser request:\n" + prompt
            self.chat_log.write(
                Text(
                    f"Plan {self._active_plan.plan_id}: "
                    f"{self._active_plan.next_pending().title if self._active_plan.next_pending() else 'complete'}",
                    style=WALLACE_THEME["primary"],
                )
            )
        except (OSError, ValueError, TypeError) as error:
            self._write_warning(f"Planning setup failed; continuing normally: {error}")
        if self.workspace_panel is not None:
            changes = self.workspace_panel.consume_changes()
            if changes:
                change_text = ", ".join(changes[:8])
                if len(changes) > 8:
                    change_text += f" (+{len(changes) - 8} more)"
                agent_prompt = (
                    "Workspace changes detected since the previous turn: "
                    f"{change_text}. Re-read the affected files with the project "
                    "tools before relying on their previous contents.\n\n"
                    f"{agent_prompt}"
                )
        self.run_turn(
            agent_prompt,
            user_prompt=prompt,
            # ConversationStore remains available for the visible TUI log and
            # reload recovery, but is deliberately not replayed into the model.
            # The durable plan is the compact cross-turn context.
            seed_messages=None,
        )

    # ---------- agent turn ----------

    @work(exclusive=True, group="turn")
    async def run_turn(
        self,
        prompt: str,
        *,
        user_prompt: str | None = None,
        seed_messages: list[dict[str, str]] | None = None,
    ) -> None:
        turn_messages = []
        payload = {"messages": [*(seed_messages or []), {"role": "user", "content": prompt}]}
        recovery_attempted = False
        empty_response_retried = False
        response = None
        self._shown_tool_calls: set[tuple] = set()
        self._shown_tool_results: set[tuple] = set()
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
                            if messages is None:
                                continue
                            if not isinstance(messages, (list, tuple)):
                                messages = [messages]
                            for msg in messages:
                                turn_messages.append(msg)
                                self._show_progress(msg)
                except Exception as error:
                    if not recovery_attempted and is_malformed_tool_call_error(error):
                        recovery_attempted = True
                        self._write_warning(
                            "The model returned malformed tool-call JSON; "
                            "recovering from the saved turn state."
                        )
                        payload = await malformed_retry_payload(
                            self.agent, self.thread_config, prompt
                        )
                        continue
                    raise

                requests = extract_action_requests(interrupts)
                if not requests:
                    # The checkpoint includes tool results that a streamed
                    # update can omit. Decide whether a final answer exists
                    # before leaving the approval/recovery loop.
                    response = assistant_response_text(turn_messages)
                    try:
                        state = await self.agent.aget_state(self.thread_config)
                        stored_messages = state.values.get("messages", [])
                        stored_response = assistant_response_text(stored_messages)
                        if stored_response is not None:
                            response = stored_response
                        turn_messages.extend(stored_messages)
                    except Exception:
                        pass
                    if response is None and not empty_response_retried:
                        empty_response_retried = True
                        self._write_warning(
                            "The model returned no final text; continuing the "
                            "unfinished turn once."
                        )
                        payload = {"messages": [
                            {"role": "user", "content": EMPTY_RESPONSE_RETRY_PROMPT}
                        ]}
                        continue
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

            if response is None:
                activity = model_tool_activity(turn_messages)
                if self._active_plan is not None:
                    self._active_plan.record_progress(
                        activity,
                        error="No final assistant text after one recovery attempt.",
                    )
                status = no_text_status(self._active_plan, activity)
                if self._active_plan is not None:
                    self._active_plan.record_progress(
                        [],
                        error=status,
                    )
                    save_plan(SAFE_ROOT, self._active_plan)
                self.chat_log.write(
                    Text(
                        status,
                        style=WALLACE_THEME["error"],
                    )
                )
            else:
                quality_error = response_quality_error(response)
                if quality_error is not None:
                    raise RepetitiveResponseError(quality_error)
                self.chat_log.write(
                    Text("AI:", style=f"bold {WALLACE_THEME['assistant']}")
                )
                self.chat_log.write(Markdown(response))
                if self._active_plan is not None:
                    self._active_plan.record_progress(
                        model_tool_activity(turn_messages),
                        response=response,
                    )
                    save_plan(SAFE_ROOT, self._active_plan)
                    self.thread_manager.touch(self.thread_id)
                    self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
                if user_prompt is not None:
                    self._conversation_history.append(
                        {"role": "assistant", "content": response}
                    )
                    self._conversation_history = self.conversation_store.trim(
                        self._conversation_history
                    )
                    self.conversation_store.save(self._conversation_history)
                    self.conversation_store.clear_pending()

        except Exception as error:
            if self._active_plan is not None:
                try:
                    state = await self.agent.aget_state(self.thread_config)
                    turn_messages.extend(state.values.get("messages", []))
                except Exception:
                    pass
                try:
                    recorded_error = (
                        describe_error(error)
                        if is_malformed_tool_call_error(error)
                        else str(error)
                    )
                    self._active_plan.record_progress(
                        model_tool_activity(turn_messages), error=recorded_error
                    )
                    save_plan(SAFE_ROOT, self._active_plan)
                except Exception:
                    pass
            self._write_warning(describe_error(error))
        finally:
            # Keep only the current turn's tool transcript in the graph.
            self.thread_config = new_thread_config()
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
            call_key = _tool_call_key(call)
            if call_key in self._shown_tool_calls:
                continue
            self._shown_tool_calls.add(call_key)
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
            result_id = getattr(msg, "tool_call_id", None) or getattr(msg, "id", None)
            result_key = (
                ("id", str(result_id))
                if result_id
                else ("result", name, str(getattr(msg, "content", "")))
            )
            if result_key in self._shown_tool_results:
                return
            self._shown_tool_results.add(result_key)
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

    def _update_header_datetime(self) -> None:
        """Keep the local weekday, date and time alongside Wallace."""
        self.sub_title = datetime.now().astimezone().strftime("%A, %Y-%m-%d %H:%M")

    def _switch_thread(self, thread_id: str) -> None:
        if self._busy:
            self._write_warning("Finish or cancel the current request before switching threads.")
            return
        record = self.thread_manager.resolve(thread_id)
        self.thread_id = record.thread_id
        self.conversation_store = self.thread_manager.store(self.thread_id)
        self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
        self._conversation_history = self.conversation_store.load()
        self.prompt_input.load_history(
            [
                message["content"]
                for message in self._conversation_history
                if message["role"] == "user"
            ]
        )
        self._active_plan = None
        self.thread_config = new_thread_config()
        self.chat_log.clear()
        self.chat_log.write(Text(f"Thread: {record.thread_id} — {record.title}", style=WALLACE_THEME["success"]))
        self._restore_conversation()
        if self.conversation_store.load_pending():
            self.chat_log.write(Text("This thread has an interrupted request; send it again to resume.", style=WALLACE_THEME["warning"]))

    def _handle_thread_command(self, command: str) -> None:
        try:
            parts = shlex.split(command)
        except ValueError as error:
            self._write_warning(f"Invalid thread command: {error}")
            return
        if not parts:
            return
        action = parts[0].lower()
        try:
            if action in {"/theme", "/themes"}:
                if len(parts) == 1:
                    self.action_change_theme()
                else:
                    reference = "-".join(parts[1:]).lower()
                    reference = {
                        "retro-terminal": "retro",
                        "sophistication": "elegant",
                        "sophisticated": "elegant",
                        "futuristic": "future",
                        "night": "night-mode",
                        "red": "red-light",
                        "redlight": "red-light",
                    }.get(reference, reference)
                    candidates = (reference, f"wallace-{reference}", f"omarchy-{reference}")
                    selected = next((name for name in candidates if self.get_theme(name)), None)
                    if selected is None:
                        raise ValueError(f"unknown theme: {' '.join(parts[1:])}")
                    self.theme = selected
                    self.chat_log.write(f"Theme: {selected}")
            elif action in {"/threads", "/thread"}:
                self.chat_log.write(Text("Conversation threads:", style=WALLACE_THEME["primary"]))
                for record in self.thread_manager.list_threads():
                    marker = "*" if record.thread_id == self.thread_id else " "
                    self.chat_log.write(f"{marker} {record.thread_id}: {record.title}")
            elif action == "/new":
                record = self.thread_manager.create(" ".join(parts[1:]) or None)
                self._switch_thread(record.thread_id)
            elif action == "/switch" and len(parts) >= 2:
                record = self.thread_manager.resolve(" ".join(parts[1:]))
                self._switch_thread(record.thread_id)
            elif action == "/rename" and len(parts) >= 2:
                record = self.thread_manager.rename(self.thread_id, " ".join(parts[1:]))
                self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
                self.chat_log.write(Text(f"Renamed thread to: {record.title}", style=WALLACE_THEME["success"]))
            elif action in {"/remove", "/delete"} and len(parts) >= 2:
                removed = self.thread_manager.remove(" ".join(parts[1:]))
                removing_current = removed.thread_id == self.thread_id
                if removing_current:
                    replacement = self.thread_manager.list_threads()[0]
                    self._switch_thread(replacement.thread_id)
                    self.chat_log.write(Text(f"Removed thread: {removed.thread_id}", style=WALLACE_THEME["success"]))
                else:
                    self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
                    self.chat_log.write(Text(f"Removed thread: {removed.thread_id}", style=WALLACE_THEME["success"]))
            elif action == "/clear":
                self.action_clear_log()
            else:
                self._write_warning("Commands: /threads, /new [title], /switch <id or title>, /rename <title>, /remove <id or title>, /clear")
        except (OSError, ValueError) as error:
            self._write_warning(f"Thread command failed: {error}")

    def action_clear_log(self) -> None:
        self.chat_log.clear()
        self._conversation_history.clear()
        self.conversation_store.clear()
        self.prompt_input.load_history([])
        self.chat_log.write(
            Text("Conversation history cleared.", style=WALLACE_THEME["success"])
        )

    def _refresh_workspace(self) -> None:
        if self.workspace_panel is not None:
            self.workspace_panel.refresh_view()

    def action_toggle_workspace(self) -> None:
        if self.workspace_panel is None:
            return
        self.workspace_panel.display = not self.workspace_panel.display

    def action_toggle_threads(self) -> None:
        if hasattr(self, "thread_panel"):
            self.thread_panel.display = not self.thread_panel.display

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

    def on_unmount(self) -> None:
        _write_reload_status("idle")


if __name__ == "__main__":
    AgentTUI().run()
