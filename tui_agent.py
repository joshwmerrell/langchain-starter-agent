"""Textual TUI front end for Wallace.

Run with:  python tui_agent.py
Keys:      Enter = send, Esc = cancel current request, Ctrl+L = clear log, Ctrl+Q = quit
           Select chat text with the mouse and press Ctrl+C; Input supports Ctrl+C/Ctrl+V.
Threads:   /threads, /new [--trust MODE] [title], /trust [MODE], /switch, /rename, /remove, /clear
Editor:    F5 or /edit [relative path] opens a safe workspace file in Neovim/Vim
Themes:    F9 or /theme to choose; /theme <name> to apply directly
Models:    F10 or /model to list; /model <name> to switch; /key opens masked entry
MCP:       F11 or /mcp to list; /mcp add|remove|on|off|reload to manage
Approvals: y = approve, n / Esc = reject (when the agent wants to run a command or change a file)
"""

import asyncio
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from fnmatch import fnmatch
from functools import partial
from pathlib import Path

from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Span, Text
from textual import work
from textual.app import App, ComposeResult, SuspendNotSupported
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
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from main import (
    CLOUD_KEY_WARNING,
    CLOUD_TOOL_WARNING,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    assistant_response_text,
    clean_response_retry_prompt,
    ConversationStore,
    EMPTY_RESPONSE_RETRY_PROMPT,
    describe_error,
    extract_action_requests,
    format_action_request,
    is_malformed_tool_call_error,
    local_datetime_response,
    list_models_for_provider,
    malformed_retry_payload,
    make_decision,
    memory_budget_for,
    model_tool_activity,
    new_thread_config,
    no_text_status,
    RepetitiveResponseError,
    recover_pending_full_suite,
    response_quality_error,
    resume_command,
    setup_agent,
    summarize_thread_memory,
    with_thread_memory,
    verified_plan_response,
    tool_result_failed,
)
from tools import SAFE_ROOT, SENSITIVE_PATTERNS, SKIP_DIRS
from wallace_mcp import (
    all_mcp_servers,
    apply_mcp_command,
    connection_status,
    parse_mcp_command,
    read_mcp_configuration,
    restore_connection_statuses,
    save_mcp_configuration,
    snapshot_connection_statuses,
)
from wallace_models import (
    CloudCredentials,
    MissingCredentialError,
    load_model_choice,
    list_installed_models,
    parse_model_selection,
    save_model_choice,
    use_environment_key,
)
from wallace_planning import (
    ArchiveManager,
    ConversationThreadManager,
    TRUST_MODES,
    continuation_instruction,
    parse_new_thread_arguments,
    planning_context,
    resume_or_create,
    save_plan,
    validate_trust_mode,
)
from wallace_sky import sky_icons
from wallace_themes import (
    DARK_HEADING_COLOR,
    DARK_TEXT_ACCENTS,
    DARK_WORKING_COLOR,
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


def _grayscale_strip(
    strip: Strip,
    *,
    black_background: bool = False,
    preserve_foreground: frozenset[str] = frozenset(),
) -> Strip:
    """Remove syntax hues while retaining intentional theme text accents."""
    def gray(color, *, foreground: bool) -> str | None:
        if color is None:
            return None
        triplet = color.get_truecolor()
        if triplet is None:
            return None
        original = f"#{triplet.red:02x}{triplet.green:02x}{triplet.blue:02x}"
        if foreground and original in preserve_foreground:
            return original
        level = round(0.2126 * triplet.red + 0.7152 * triplet.green + 0.0722 * triplet.blue)
        level = max(176, level) if foreground else (0 if black_background else min(51, level))
        return f"#{level:02x}{level:02x}{level:02x}"

    segments = []
    for segment in strip:
        style = segment.style or Style()
        foreground = gray(style.color, foreground=True)
        background = gray(style.bgcolor, foreground=False)
        if foreground is not None or background is not None:
            style = style + Style(color=foreground, bgcolor=background)
        segments.append(Segment(segment.text, style, segment.control))
    return Strip(segments, strip.cell_length)


def _short(value: object, limit: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_tool_call(call: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 40)!r}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('name', 'tool')}({args})"


class HistoryInput(Input):
    """Single-line editor with per-thread Up/Down command history."""

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding(
            "ctrl+backspace,alt+backspace",
            "delete_left_word",
            "Delete previous word",
            show=False,
        ),
        Binding("ctrl+delete", "delete_right_word", "Delete next word", show=False),
        Binding("ctrl+home", "home", "Move to start", show=False),
        Binding("ctrl+end", "end", "Move to end", show=False),
        Binding("ctrl+shift+home", "home(True)", "Select to start", show=False),
        Binding("ctrl+shift+end", "end(True)", "Select to end", show=False),
        Binding("ctrl+z", "undo_edit", "Undo", show=False),
        Binding("ctrl+y,ctrl+shift+z", "redo_edit", "Redo", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._input_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._setting_history_value = False
        self._history_event_value: str | None = None
        self._undo_stack: list[tuple[str, tuple[int, int]]] = []
        self._redo_stack: list[tuple[str, tuple[int, int]]] = []
        self._restoring_edit = False

    def reset_edit_history(self) -> None:
        """Keep undo/redo within the current unsent input, not another thread."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _edit_state(self) -> tuple[str, tuple[int, int]]:
        return self.value, (self.selection.start, self.selection.end)

    def replace(self, text: str, start: int, end: int) -> None:
        before = self._edit_state()
        super().replace(text, start, end)
        if not self._restoring_edit and self.value != before[0]:
            self._undo_stack.append(before)
            self._undo_stack = self._undo_stack[-100:]
            self._redo_stack.clear()

    def _restore_edit(self, state: tuple[str, tuple[int, int]]) -> None:
        self._restoring_edit = True
        try:
            value, (start, end) = state
            self.value = value
            self.selection = type(self.selection)(start, end)
        finally:
            self._restoring_edit = False

    def action_undo_edit(self) -> None:
        if self._undo_stack:
            self._redo_stack.append(self._edit_state())
            self._restore_edit(self._undo_stack.pop())

    def action_redo_edit(self) -> None:
        if self._redo_stack:
            self._undo_stack.append(self._edit_state())
            self._restore_edit(self._redo_stack.pop())

    def load_history(self, values: list[str]) -> None:
        self.reset_edit_history()
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
        self.reset_edit_history()
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
        yield Static("/new  /switch  /trust  /edit", id="thread_hint")

    def refresh_threads(self, manager: ConversationThreadManager, current_id: str) -> None:
        text = Text()
        records = manager.list_threads()
        if not records:
            text.append("No threads yet\n", style=WALLACE_THEME["muted"])
        for record in records:
            marker = "● " if record.thread_id == current_id else "  "
            style = (
                f"bold {DARK_WORKING_COLOR if self.app.theme == 'wallace-dark' else WALLACE_THEME['success']}"
                if record.thread_id == current_id
                else ""
            )
            text.append(f"{marker}{record.title}\n", style=style)
            text.append(f"   {record.thread_id} · {record.trust_mode}\n", style=WALLACE_THEME["muted"])
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
        if self.app.theme == "wallace-red":
            # Rich Markdown and syntax highlighting can supply their own blue
            # or green colors; keep even that content monochrome in red light.
            strip = _overlay_strip_style(strip, RED_LIGHT_STYLE)
        elif self.app.theme in {"wallace-dark", "wallace-grayscale"}:
            strip = _grayscale_strip(
                strip,
                black_background=self.app.theme == "wallace-dark",
                preserve_foreground=(
                    DARK_TEXT_ACCENTS if self.app.theme == "wallace-dark" else frozenset()
                ),
            )
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

    def __init__(self, detail: str, *, countdown_seconds: float | None = None) -> None:
        super().__init__()
        self.detail = detail
        self.countdown_seconds = countdown_seconds
        self._deadline: float | None = None
        self._countdown_paused = False
        self._countdown_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("The agent wants to run this action", id="approval_title")
            with VerticalScroll(id="approval_detail"):
                yield Static(_styled_detail(self.detail))
            yield Static("Y = Yes   N = No   Esc = No", id="approval_hint")
            with Horizontal(id="approval_buttons"):
                yield Button("No (N)", variant="error", id="reject")
                yield Button("Yes (Y)", variant="success", id="approve")

    def on_mount(self) -> None:
        if self.countdown_seconds is not None:
            self._deadline = time.monotonic() + self.countdown_seconds
            self._update_countdown()
            self._countdown_timer = self.set_interval(0.1, self._update_countdown)

    def _update_countdown(self) -> None:
        if self._deadline is None or self._countdown_paused:
            return
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            if self._countdown_timer is not None:
                self._countdown_timer.stop()
            self.dismiss(True)
            return
        self.query_one("#approval_hint", Static).update(
            f"Auto-approve in {math.ceil(remaining)}s · move mouse or press a key to pause · Y/N"
        )

    def _pause_countdown(self) -> None:
        if self._deadline is None or self._countdown_paused:
            return
        self._countdown_paused = True
        if self._countdown_timer is not None:
            self._countdown_timer.stop()
        self.query_one("#approval_hint", Static).update(
            "Countdown paused. Read the action, then press Y = Yes or N = No (Esc = No)."
        )

    def on_mouse_move(self, event) -> None:
        self._pause_countdown()

    def on_mouse_down(self, event) -> None:
        self._pause_countdown()

    def on_mouse_scroll_up(self, event) -> None:
        self._pause_countdown()

    def on_mouse_scroll_down(self, event) -> None:
        self._pause_countdown()

    def on_key(self, event: Key) -> None:
        self._pause_countdown()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class WorkspaceFileScreen(ModalScreen[str | None]):
    """User-only, searchable picker for an existing non-sensitive workspace file."""

    DEFAULT_CSS = """
    WorkspaceFileScreen { align: center middle; }
    #file_dialog {
        width: 85%; max-width: 100; height: 80%;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    #file_title { height: 1; text-style: bold; margin-bottom: 1; }
    #file_search { height: 3; }
    #file_options { height: 1fr; }
    #file_hint { height: auto; color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths
        self._matches = paths[:100]

    def compose(self) -> ComposeResult:
        with Vertical(id="file_dialog"):
            yield Static("Edit a workspace file", id="file_title")
            yield Input(placeholder="Filter by filename or path", id="file_search")
            yield OptionList(*(Option(path, id=path) for path in self._matches), id="file_options")
            yield Static(
                "Enter opens the first match; ↓ selects a result. Esc cancels. "
                "Private files are excluded.",
                id="file_hint",
            )

    def on_mount(self) -> None:
        self.query_one("#file_search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "file_search":
            return
        query = event.value.casefold().strip()
        self._matches = [path for path in self.paths if query in path.casefold()][:100]
        options = self.query_one("#file_options", OptionList)
        options.clear_options()
        options.add_options(Option(path, id=path) for path in self._matches)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._matches:
            self.dismiss(self._matches[0])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def on_key(self, event: Key) -> None:
        if event.key == "down" and self.query_one("#file_search", Input).has_focus:
            self.query_one("#file_options", OptionList).focus()
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)


class CloudKeyScreen(ModalScreen[str | None]):
    """Accept a key without putting it in the chat input or command history."""

    DEFAULT_CSS = """
    CloudKeyScreen { align: center middle; }
    #cloud_key_dialog {
        width: 70%;
        max-width: 76;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #cloud_key_title { text-style: bold; margin-bottom: 1; }
    #cloud_key_hint { height: auto; color: $text-muted; margin-bottom: 1; }
    #cloud_key_input { width: 1fr; margin-bottom: 1; }
    #cloud_key_buttons { height: auto; align-horizontal: right; }
    #cloud_key_buttons Button { margin-left: 2; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def compose(self) -> ComposeResult:
        with Vertical(id="cloud_key_dialog"):
            yield Static(f"Enter {self.provider} API key", id="cloud_key_title")
            yield Static(
                "Hidden input. This entry applies to the running session; Esc cancels.",
                id="cloud_key_hint",
            )
            yield Input(password=True, placeholder="API key", id="cloud_key_input")
            with Horizontal(id="cloud_key_buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Use key", variant="primary", id="submit")

    def on_mount(self) -> None:
        self.query_one("#cloud_key_input", Input).focus()

    def _submit(self) -> None:
        field = self.query_one("#cloud_key_input", Input)
        key = field.value
        field.value = ""
        self.dismiss(key or None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "submit":
            self._submit()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
    App.-theme-wallace-dark #thread_title,
    App.-theme-wallace-dark #workspace_title,
    App.-theme-wallace-dark #workspace_activity_title {{
        color: {DARK_HEADING_COLOR};
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
    App.-theme-wallace-dark #thinking {{
        color: {DARK_WORKING_COLOR};
    }}
    #prompt {{
        margin: 0 1 1 1;
        border: round $primary;
        &:focus {{
            border: round $secondary;
        }}
        & > .input--selection {{
            background: $primary;
            color: $background;
        }}
    }}
    App.-theme-wallace-dark #prompt {{
        color: {DARK_WORKING_COLOR};
    }}
    App.-theme-wallace-dark Button {{
        background: #000000;
        color: $text;
        border: solid $secondary;
    }}
    App.-theme-wallace-dark Button:hover,
    App.-theme-wallace-dark Button:focus {{
        background: #000000;
        color: $text;
        border: solid $primary;
    }}
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("ctrl+t", "toggle_threads", "Threads"),
        ("f5", "edit_file", "Edit file"),
        ("f6", "toggle_workspace", "Workspace"),
        Binding("f7", "narrow_workspace", "Narrow workspace", show=False),
        Binding("f8", "widen_workspace", "Widen workspace", show=False),
        ("f9", "change_theme", "Themes"),
        ("f10", "show_models", "Models"),
        ("f11", "show_mcp", "MCP"),
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
        self.model_name = load_model_choice(SAFE_ROOT, OLLAMA_MODEL)
        self.credentials = CloudCredentials()
        self.agent = None
        self.thread_config = None
        self.workspace_panel = None
        self._workspace_width = 30
        self.thread_manager = ConversationThreadManager(SAFE_ROOT)
        self.thread_manager.ensure("main")
        self.thread_id = self.thread_manager.active_thread_id()
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
        self._update_header_datetime()
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
        self.chat_log = self.query_one("#chat_log", SelectableRichLog)
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
        self._write_chat_block(Text("Welcome to Wallace.", style=f"bold {WALLACE_THEME['success']}"))
        self._write_chat_block(
            f"Thread: {self.thread_id} · trust: {self.thread_manager.resolve(self.thread_id).trust_mode}. "
            "/threads /new /switch /trust /edit /rename /remove /clear /theme /model /mcp"
        )
        self._announce_trust_risk(self.thread_manager.resolve(self.thread_id).trust_mode)
        self._restore_conversation()
        if pending_prompt:
            self._write_chat_block(
                Text(
                    "The previous request was interrupted before completion. "
                    "Ask Wallace to continue it when ready.",
                    style=WALLACE_THEME["warning"],
                )
            )
        self._write_chat_block(
            Text("Initializing Wallace...", style=WALLACE_THEME["warning"])
        )
        self.initialize_agent()

    def _write_chat_block(self, content: object) -> None:
        """Separate conversation turns and startup notices by one blank row."""
        if self.chat_log._theme_entries:
            self.chat_log.write("")
        self.chat_log.write(content)

    def _announce_trust_risk(self, mode: str) -> None:
        if mode == "informed-trust":
            self._write_chat_block(Text(
                "Informed-trust: approval-required actions auto-run after 10 seconds unless you interact with the preview.",
                style=WALLACE_THEME["warning"],
            ))
        elif mode == "yes-trust":
            self._write_chat_block(Text(
                "Yes-trust: approval-required actions run without prompts, including commands that can access files under your OS account.",
                style=WALLACE_THEME["warning"],
            ))

    def _user_chat_line(self, content: str) -> Text:
        line = Text()
        line.append("User: ", style=f"bold {WALLACE_THEME['user']}")
        if self.theme == "wallace-dark":
            line.append(content, style=WALLACE_THEME["user"])
        else:
            line.append(content)
        return line

    def _restore_conversation(self) -> None:
        if not self._conversation_history:
            return
        previous_assistant = None
        current_request = ""
        self._write_chat_block(
            Text(
                f"Restored {sum(message['role'] == 'assistant' for message in self._conversation_history)} previous turn(s).",
                style=WALLACE_THEME["success"],
            )
        )
        for message in self._conversation_history:
            if message["role"] == "user":
                current_request = message["content"]
                self._write_chat_block(self._user_chat_line(message["content"]))
            else:
                self._write_chat_block(
                    Text("AI:", style=f"bold {WALLACE_THEME['assistant']}")
                )
                if response_quality_error(
                    message["content"],
                    previous_assistant=previous_assistant,
                    current_request=current_request,
                ):
                    self.chat_log.write(Text(
                        "Previous AI reply omitted from display: unusable model "
                        "output. The saved conversation is unchanged.",
                        style=WALLACE_THEME["warning"],
                    ))
                else:
                    self.chat_log.write(Markdown(message["content"]))
                previous_assistant = message["content"]

    # ---------- setup ----------

    @work(exclusive=True, group="setup")
    async def initialize_agent(self) -> None:
        try:
            provider, _ = parse_model_selection(self.model_name)
            if provider != "ollama" and not await self._ensure_cloud_key(provider):
                self._set_busy(False)
                self.prompt_input.placeholder = "Cloud key needed. Use /key or /model to continue."
                return
            self.agent, self.thread_config = await setup_agent(
                on_status=self._write_warning,
                model_name=self.model_name,
                credentials=self.credentials,
            )
        except Exception as error:
            self._write_warning(f"Failed to start agent: {error}")
            # A failed setup must not leave dev_tui.py waiting forever for an
            # agent that will never become idle.
            self._set_busy(False)
            self.prompt_input.placeholder = "Agent unavailable. Use /model to select another model."
            return

        self._write_chat_block(
            Text(f"Wallace is ready. Model: {self.model_name}", style=f"bold {WALLACE_THEME['success']}")
        )
        self._set_busy(False)

    async def _prompt_cloud_key(self, provider: str) -> bool:
        self._write_warning(CLOUD_KEY_WARNING)
        key = await self.push_screen_wait(CloudKeyScreen(provider))
        if key is None:
            return False
        try:
            self.credentials.set(provider, key)
        except ValueError as error:
            self._write_warning(str(error))
            return False
        return True

    async def _ensure_cloud_key(self, provider: str) -> bool:
        if self.credentials.get(provider) is not None:
            return True
        if use_environment_key(self.credentials, provider):
            return True
        return await self._prompt_cloud_key(provider)

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
                style=f"bold {DARK_WORKING_COLOR if self.theme == 'wallace-dark' else WALLACE_THEME['primary']}",
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

    def action_show_models(self) -> None:
        """Show models available through the configured providers."""
        if not self._busy:
            self._set_busy(True)
            self.show_models()

    def action_show_mcp(self) -> None:
        """Show saved endpoint choices and observed live connection state."""
        try:
            servers = all_mcp_servers(SAFE_ROOT)
        except (OSError, ValueError) as error:
            self._write_warning(f"Could not list MCP servers: {error}")
            return
        self.chat_log.write(Text("MCP servers:", style=WALLACE_THEME["primary"]))
        for server in servers:
            kind = "built-in, cannot remove" if server.builtin else "user-added"
            active = "active" if server.enabled else "inactive"
            self.chat_log.write(
                f"  {server.name} [{kind}] — {active}, {connection_status(server)}\n"
                f"    {server.url}"
            )
        self.chat_log.write(
            "Use /mcp add <name> <URL> [keywords...], /mcp on|off|remove <name>, "
            "or /mcp reload. 'Connected' means the last discovery succeeded; "
            "active choices are saved across restarts."
        )

    @work(exclusive=True, group="mcp")
    async def change_mcp(self, operation: str, name: str = "", url: str = "", keywords: tuple[str, ...] = ()) -> None:
        """Apply a user command and rebuild the live agent without losing its thread."""
        self._set_busy(True)
        self._set_thinking_status(rotate=False, message="Updating MCP connections")
        previous_config: dict | None = None
        previous_status = snapshot_connection_statuses()
        changed = False
        try:
            previous_config = read_mcp_configuration(SAFE_ROOT)
            changed = apply_mcp_command(
                SAFE_ROOT, operation, name, url, keywords, live_applied=True
            )
            if not changed and operation != "reload":
                self.chat_log.write(f"MCP server {name} is already {operation}.")
                return
            if self.agent is not None:
                replacement_agent, replacement_config = await setup_agent(
                    on_status=self._write_warning,
                    model_name=self.model_name,
                    credentials=self.credentials,
                )
                self.agent, self.thread_config = replacement_agent, replacement_config
            description = (
                "MCP connections refreshed."
                if operation == "reload" else f"MCP {operation} {name} saved."
            )
            if self.agent is None:
                description += " Connections will load when Wallace starts."
            self.chat_log.write(Text(description, style=WALLACE_THEME["success"]))
            self.action_show_mcp()
        except Exception as error:
            if changed and previous_config is not None:
                try:
                    save_mcp_configuration(SAFE_ROOT, previous_config, live_applied=True)
                except (OSError, ValueError) as rollback_error:
                    self._write_warning(f"MCP rollback failed: {rollback_error}")
            restore_connection_statuses(previous_status)
            self._write_warning(f"Could not update MCP servers: {self.credentials.redact(str(error))}")
        finally:
            self._set_busy(False)

    @work(exclusive=True, group="model")
    async def show_models(self, provider: str | None = None) -> None:
        self._set_busy(True)
        self._set_thinking_status(rotate=False, message="Checking available models")
        try:
            self.chat_log.write(Text(f"Current model: {self.model_name}", style=WALLACE_THEME["primary"]))
            providers = (provider,) if provider else ("ollama", "openai", "gemini")
            for source in providers:
                try:
                    models = await list_models_for_provider(source, self.credentials)
                    self.chat_log.write(f"{source.title()} models:")
                    if not models:
                        self.chat_log.write("  No models returned.")
                    for name in models:
                        full_name = name if source == "ollama" else f"{source}:{name}"
                        marker = "*" if full_name == self.model_name else " "
                        self.chat_log.write(f"{marker} {full_name}")
                except MissingCredentialError:
                    variable = "OPENAI_API_KEY" if source == "openai" else "GOOGLE_API_KEY"
                    self._write_warning(f"{source.title()}: set {variable} in .env or the environment to list account models.")
                except Exception as error:
                    self._write_warning(f"Could not list {source} models: {error}")
        finally:
            self.chat_log.write(
                "Use /models [ollama|openai|gemini] to filter; /model <name> to switch. "
                "Cloud lists may include models Wallace cannot use for chat/tools."
            )
            self._set_busy(False)

    @work(exclusive=True, group="model")
    async def change_model(self, requested: str) -> None:
        """Switch the live agent and persist only after a successful rebuild."""
        self._set_busy(True)
        self._set_thinking_status(rotate=False, message="Switching model")
        previous_provider, _ = parse_model_selection(self.model_name)
        previous_key = self.credentials.get(previous_provider)
        try:
            if requested == self.model_name and self.agent is not None:
                self.chat_log.write(f"Already using model: {requested}")
                return
            provider, _ = parse_model_selection(requested)
            if provider == "ollama":
                models = await list_installed_models(OLLAMA_BASE_URL)
                if requested not in models:
                    self._write_warning(
                        f"Model {requested!r} is not installed. Use /model to list available models."
                    )
                    return
            elif self.credentials.get(provider) is None:
                if not await self._ensure_cloud_key(provider):
                    return
            replacement_agent, replacement_config = await setup_agent(
                on_status=self._write_warning,
                model_name=requested,
                credentials=self.credentials,
            )
            save_model_choice(SAFE_ROOT, requested)
            self.agent, self.thread_config = replacement_agent, replacement_config
            self.model_name = requested
            if provider == "ollama":
                self.credentials.clear()
            self.chat_log.write(Text(
                f"Now using model: {requested}. Conversation history is preserved.",
                style=WALLACE_THEME["success"],
            ))
        except Exception as error:
            detail = self.credentials.redact(str(error))
            if previous_key is not None:
                detail = detail.replace(previous_key.get_secret_value(), "[REDACTED API KEY]")
                self.credentials.set(previous_provider, previous_key.get_secret_value())
            else:
                self.credentials.clear()
            self._write_warning(f"Could not switch model: {detail}")
        finally:
            self._set_busy(False)

    @work(exclusive=True, group="model")
    async def change_cloud_key(self) -> None:
        provider, _ = parse_model_selection(self.model_name)
        if provider == "ollama":
            self._write_warning("Select an openai:<model> or gemini:<model> first.")
            self._set_busy(False)
            return
        previous_key = self.credentials.get(provider)
        self._set_busy(True)
        try:
            if not await self._prompt_cloud_key(provider):
                return
            replacement_agent, replacement_config = await setup_agent(
                on_status=self._write_warning,
                model_name=self.model_name,
                credentials=self.credentials,
            )
            self.agent, self.thread_config = replacement_agent, replacement_config
            self.chat_log.write(Text("Cloud key updated for this session.", style=WALLACE_THEME["success"]))
        except Exception as error:
            detail = self.credentials.redact(str(error))
            if previous_key is not None:
                detail = detail.replace(previous_key.get_secret_value(), "[REDACTED API KEY]")
                self.credentials.set(provider, previous_key.get_secret_value())
            else:
                self.credentials.clear()
            self._write_warning(f"Could not update cloud key: {detail}")
        finally:
            self._set_busy(False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        self.prompt_input.value = ""
        self.prompt_input.reset_edit_history()

        if not prompt:
            return
        if prompt.lower() == "exit":
            self.exit()
            return
        if prompt.startswith("/"):
            self._handle_thread_command(prompt)
            return
        if self.agent is None:
            self.prompt_input.value = prompt
            self._write_warning("Wallace is unavailable. Use /model or /key to continue.")
            return

        try:
            self.conversation_store.save_pending(prompt)
            self.conversation_store.append("user", prompt)
        except (OSError, ValueError) as error:
            self.prompt_input.value = prompt
            self._write_warning(f"Could not save conversation turn: {error}")
            return
        self._conversation_history = self.conversation_store.load()
        self.prompt_input.remember_input(prompt)

        self._write_chat_block(self._user_chat_line(prompt))

        self._set_busy(True)
        agent_prompt = prompt
        self._active_plan = None
        try:
            self._active_plan = resume_or_create(
                SAFE_ROOT, prompt, session_id=self.thread_id
            )
            agent_prompt = planning_context(
                self._active_plan,
                latest_instruction=continuation_instruction(
                    prompt, self.conversation_store.load_all()
                ),
            ) + "\nUser request:\n" + prompt
            self._write_chat_block(
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
        host_response = local_datetime_response(prompt)
        if host_response is not None:
            self._write_chat_block(Text("AI:", style=f"bold {WALLACE_THEME['assistant']}"))
            self.chat_log.write(Markdown(host_response))
            try:
                if self._active_plan is not None:
                    self._active_plan.record_progress([], response=host_response)
                    save_plan(SAFE_ROOT, self._active_plan)
                self.conversation_store.append("assistant", host_response)
                self._conversation_history = self.conversation_store.load()
                self.conversation_store.clear_pending()
                self.thread_manager.touch(self.thread_id)
                self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
            except (OSError, ValueError) as error:
                self._write_warning(f"Could not save local date/time answer: {error}")
            self._set_busy(False)
            return
        self.run_turn(
            agent_prompt,
            user_prompt=prompt,
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
        quality_retried = False
        response = None
        self._shown_tool_calls: set[tuple] = set()
        self._shown_tool_results: set[tuple] = set()
        initial_prompt = prompt
        try:
            if user_prompt is not None:
                memory = ""
                try:
                    memory, memory_warning = await self.conversation_store.memory_context(
                        user_prompt,
                        partial(
                            summarize_thread_memory,
                            model_name=self.model_name,
                            credentials=self.credentials,
                        ),
                        max_chars=memory_budget_for(prompt),
                    )
                    if memory_warning:
                        self._write_warning(memory_warning)
                except (OSError, ValueError) as error:
                    self._write_warning(f"Conversation memory unavailable: {error}")
                initial_prompt = with_thread_memory(memory, prompt)
                payload["messages"][-1]["content"] = initial_prompt
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
                            self.agent, self.thread_config, initial_prompt
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
                    if response is not None and user_prompt is not None:
                        previous_assistant = next(
                            (
                                item["content"] for item in reversed(self.conversation_store.load_all())
                                if item["role"] == "assistant"
                            ),
                            None,
                        )
                        quality_error = response_quality_error(
                            response,
                            previous_assistant=previous_assistant,
                            current_request=user_prompt,
                        )
                        if (
                            quality_error is not None
                            and not quality_retried
                            and (self._active_plan is None or self._active_plan.intent == "general task")
                            and not model_tool_activity(turn_messages)
                        ):
                            quality_retried = True
                            self._write_warning(
                                "The model echoed or looped; retrying the simple request "
                                "once without prior replies."
                            )
                            self.thread_config = new_thread_config()
                            payload = {"messages": [{
                                "role": "user",
                                "content": clean_response_retry_prompt(user_prompt),
                            }]}
                            turn_messages = []
                            response = None
                            recovery_attempted = False
                            empty_response_retried = False
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

            activity = model_tool_activity(turn_messages)
            async def approve_host_check(request: dict) -> bool:
                self._set_thinking_status(rotate=False, message="Waiting for test approval")
                approved = await self._ask_approval(request)
                if approved:
                    self._set_thinking_status(rotate=True, message="Running full test suite")
                return approved

            host_check = await recover_pending_full_suite(
                self._active_plan, activity, approve_host_check
            )
            if host_check is not None:
                verification, observed = host_check
                activity.extend(observed)
                # The model has not seen this newly observed result, so its
                # earlier completion claim cannot be the final report.
                response = verification

            if response is None:
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
                self._write_chat_block(
                    Text(
                        status,
                        style=WALLACE_THEME["error"],
                    )
                )
            else:
                previous_assistant = next(
                    (
                        item["content"] for item in reversed(self.conversation_store.load_all())
                        if item["role"] == "assistant"
                    ),
                    None,
                )
                quality_error = response_quality_error(
                    response,
                    previous_assistant=previous_assistant,
                    current_request=user_prompt or "",
                )
                if quality_error is not None:
                    raise RepetitiveResponseError(quality_error)
                if self._active_plan is not None:
                    response = verified_plan_response(
                        self._active_plan, response, activity
                    )
                    save_plan(SAFE_ROOT, self._active_plan)
                    self.thread_manager.touch(self.thread_id)
                    self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
                self._write_chat_block(
                    Text("AI:", style=f"bold {WALLACE_THEME['assistant']}")
                )
                self.chat_log.write(Markdown(response))
                if user_prompt is not None:
                    saved = self.conversation_store.load_all()
                    if not saved or saved[-1] != {"role": "user", "content": user_prompt}:
                        self.conversation_store.append("user", user_prompt)
                    self.conversation_store.append("assistant", response)
                    self._conversation_history = self.conversation_store.load()
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
                        self.credentials.redact(describe_error(error, self.model_name))
                        if is_malformed_tool_call_error(error)
                        else self.credentials.redact(str(error))
                    )
                    self._active_plan.record_progress(
                        model_tool_activity(turn_messages), error=recorded_error
                    )
                    save_plan(SAFE_ROOT, self._active_plan)
                except Exception:
                    pass
            self._write_warning(describe_error(error, self.model_name))
        finally:
            # Keep only the current turn's tool transcript in the graph.
            self.thread_config = new_thread_config()
            self._set_busy(False)

    async def _ask_approval(self, request: dict) -> bool:
        mode = self.thread_manager.resolve(self.thread_id).trust_mode
        if mode == "yes-trust":
            self.chat_log.write(Text(
                "  Auto-approved by yes-trust mode.", style=WALLACE_THEME["warning"]
            ))
            return True
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        detail = self.credentials.redact(format_action_request(request))
        if parse_model_selection(self.model_name)[0] != "ollama":
            detail = f"{CLOUD_TOOL_WARNING}\n\n{detail}"
        self.push_screen(
            ApprovalScreen(
                detail,
                countdown_seconds=10 if mode == "informed-trust" else None,
            ),
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
            failed = tool_result_failed(str(msg.content), getattr(msg, "status", None))
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
        message = self.credentials.redact(message)
        self.chat_log.write(Text(message, style=WALLACE_THEME["warning"]))
        if self.workspace_panel is not None:
            self.workspace_panel.refresh_view(message)

    # ---------- actions ----------

    def _update_header_datetime(self) -> None:
        """Keep local time and one theme-appropriate sky icon alongside Wallace."""
        now = datetime.now().astimezone()
        # Emoji-focused themes can opt in; current presets retain text symbols.
        emoji = self.current_theme.variables.get("sky-icon-style") == "emoji"
        icon, description = sky_icons(now, emoji=emoji)
        self.sub_title = f"{now:%A, %Y-%m-%d %H:%M}  {icon}"
        self.query_one(Header).tooltip = description

    def _switch_thread(self, thread_id: str) -> None:
        if self._busy:
            self._write_warning("Finish or cancel the current request before switching threads.")
            return
        record = self.thread_manager.resolve(thread_id)
        self.thread_manager.select_thread(record.thread_id)
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
        self._write_chat_block(Text(
            f"Thread: {record.thread_id} — {record.title} · trust: {record.trust_mode}",
            style=WALLACE_THEME["success"],
        ))
        self._announce_trust_risk(record.trust_mode)
        self._restore_conversation()
        if self.conversation_store.load_pending():
            self._write_chat_block(Text("This thread has an interrupted request; send it again to resume.", style=WALLACE_THEME["warning"]))

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
            if self._busy and action in {"/new", "/switch", "/rename", "/remove", "/delete", "/trust", "/edit"}:
                raise ValueError("finish or cancel the current request before changing threads or editing files")
            if action == "/key":
                if len(parts) != 1:
                    self._write_warning("Use /key with no argument; enter the secret in the masked dialog.")
                elif not self._busy:
                    self._set_busy(True)
                    self.change_cloud_key()
            elif action in {"/model", "/models"}:
                if action == "/models":
                    if len(parts) > 2 or (len(parts) == 2 and parts[1].lower() not in {"ollama", "openai", "gemini"}):
                        raise ValueError("use /models [ollama|openai|gemini]")
                    if not self._busy:
                        self._set_busy(True)
                        self.show_models(parts[1].lower() if len(parts) == 2 else None)
                elif len(parts) == 1:
                    self.action_show_models()
                else:
                    if not self._busy:
                        self._set_busy(True)
                        self.change_model(" ".join(parts[1:]))
            elif action in {"/mcp", "/mcps"}:
                operation, name, url, keywords = parse_mcp_command(parts[1:], SAFE_ROOT)
                if operation == "list":
                    self.action_show_mcp()
                else:
                    if self._busy:
                        self._write_warning("Finish the current request before changing MCP servers.")
                    else:
                        self.change_mcp(operation, name, url, keywords)
            elif action in {"/theme", "/themes"}:
                if len(parts) == 1:
                    self.action_change_theme()
                else:
                    reference = "-".join(parts[1:]).lower()
                    reference = {
                        "retro-terminal": "retro",
                        "sophistication": "elegant",
                        "sophisticated": "elegant",
                        "night-mode": "night",
                        "red-light": "red",
                        "redlight": "red",
                        "vapor": "vaporwave",
                        "grey": "grayscale",
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
                    self.chat_log.write(f"{marker} {record.thread_id}: {record.title} [{record.trust_mode}]")
            elif action == "/new":
                title, trust_mode = parse_new_thread_arguments(parts[1:])
                record = self.thread_manager.create(title, trust_mode=trust_mode)
                self._switch_thread(record.thread_id)
            elif action == "/trust":
                if len(parts) == 1:
                    current = self.thread_manager.resolve(self.thread_id).trust_mode
                    self._write_chat_block(
                        f"Trust: {current}. Modes: {', '.join(TRUST_MODES)}. "
                        "Use /trust <mode> to change this thread."
                    )
                elif len(parts) == 2:
                    record = self.thread_manager.set_trust_mode(
                        self.thread_id, validate_trust_mode(parts[1])
                    )
                    self.thread_panel.refresh_threads(self.thread_manager, self.thread_id)
                    self._write_chat_block(Text(
                        f"Trust mode for {record.title}: {record.trust_mode}. "
                        + (
                            "Approval-required tools now run without a prompt; they can modify files and access your OS account."
                            if record.trust_mode == "yes-trust" else
                            "Approval-required tools auto-run after 10 seconds unless you interact with the dialog."
                            if record.trust_mode == "informed-trust" else
                            "Approval-required tools need Y/N confirmation."
                        ),
                        style=WALLACE_THEME["warning"],
                    ))
                else:
                    raise ValueError("use /trust [no-trust|informed-trust|yes-trust]")
            elif action == "/edit":
                if len(parts) == 1:
                    self.action_edit_file()
                else:
                    self._open_workspace_editor(" ".join(parts[1:]))
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
                self._write_warning("Commands: /threads, /new [--trust MODE] [title], /trust [MODE], /edit [path], /switch <id or title>, /rename <title>, /remove <id or title>, /clear, /theme, /model, /mcp, /key")
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

    def action_edit_file(self) -> None:
        if self._busy:
            self._write_warning("Finish or cancel the current request before editing a file.")
            return
        paths = sorted(_workspace_file_state(), key=str.casefold)
        if not paths:
            self._write_warning("No editable workspace files were found.")
            return
        self.push_screen(
            WorkspaceFileScreen(paths),
            lambda path: self._open_workspace_editor(path) if path else None,
        )

    def _open_workspace_editor(self, relative_path: str) -> None:
        if self._busy:
            self._write_warning("Finish or cancel the current request before editing a file.")
            return
        path = (SAFE_ROOT / relative_path).resolve()
        if not path.is_file() or not _allowed_workspace_path(path):
            self._write_warning("Choose an existing, non-sensitive file inside the workspace.")
            return
        editor = shutil.which("nvim") or shutil.which("vim") or shutil.which("vi")
        if editor is None:
            self._write_warning("Neovim, Vim, or vi is needed for /edit.")
            return
        self._set_busy(True)
        try:
            with self.suspend():
                result = subprocess.run([editor, "--", str(path)], cwd=SAFE_ROOT, check=False)
            self.workspace_panel.refresh_view(f"Edited {path.relative_to(SAFE_ROOT)}")
            if result.returncode == 0:
                self._write_chat_block(f"Editor closed: {path.relative_to(SAFE_ROOT)}")
            else:
                self._write_warning(f"Editor exited with status {result.returncode}.")
        except (OSError, RuntimeError, SuspendNotSupported) as error:
            self._write_warning(f"Could not open editor: {error}")
        finally:
            self._set_busy(False)

    def _refresh_workspace(self) -> None:
        if self.workspace_panel is not None and self.workspace_panel.query("#workspace_tree"):
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
        self.credentials.clear()
        _write_reload_status("idle")


if __name__ == "__main__":
    AgentTUI().run()
