"""Textual TUI front end for the coding agent.

Run with:  python tui_agent.py
Keys:      Enter = send, Esc = cancel current request, Ctrl+L = clear log, Ctrl+Q = quit
Approvals: y = approve, n / Esc = reject (when the agent wants to run a command or change a file)
"""

import asyncio

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from main import (
    assistant_response_text,
    describe_error,
    extract_action_requests,
    format_action_request,
    make_decision,
    resume_command,
    setup_agent,
)


def _short(value: object, limit: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_tool_call(call: dict) -> str:
    args = ", ".join(f"{k}={_short(v, 40)!r}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('name', 'tool')}({args})"


def _styled_detail(detail: str) -> Text:
    """Colour added (+), removed (-) and hunk (@) lines of an approval preview."""
    styles = {"+": "green", "-": "red", "@": "cyan"}
    text = Text()
    for line in detail.splitlines():
        text.append(line + "\n", style=styles.get(line[:1], ""))
    return text


class ApprovalScreen(ModalScreen[bool]):
    """Ask the user to approve or reject a risky tool call."""

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #approval_title {
        text-style: bold;
        margin-bottom: 1;
    }
    #approval_detail {
        height: auto;
        max-height: 20;
        margin-bottom: 1;
    }
    #approval_buttons {
        height: auto;
        align-horizontal: right;
    }
    #approval_buttons Button {
        margin-left: 2;
    }
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
    TITLE = "Coding Agent"

    CSS = """
    #chat_log {
        height: 1fr;
        border: solid green;
        margin: 1;
    }
    #prompt {
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("escape", "cancel", "Cancel request"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.thread_config = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat_log", highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Starting agent...", id="prompt", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat_log", RichLog)
        self.prompt_input = self.query_one("#prompt", Input)
        self.chat_log.write("[bold green]Agent TUI started.[/bold green]")
        self.chat_log.write("[yellow]Initializing agent...[/yellow]")
        self.initialize_agent()

    # ---------- setup ----------

    @work(exclusive=True, group="setup")
    async def initialize_agent(self) -> None:
        try:
            self.agent, self.thread_config = await setup_agent(on_status=self._write_warning)
        except Exception as error:
            self.chat_log.write(Text(f"Failed to start agent: {error}", style="red"))
            self.prompt_input.placeholder = "Agent unavailable. Press Ctrl+Q to quit."
            return

        self.chat_log.write("[bold green]Agent ready.[/bold green]")
        self._set_busy(False)

    # ---------- input handling ----------

    def _set_busy(self, busy: bool) -> None:
        self.prompt_input.disabled = busy
        self.prompt_input.placeholder = (
            "Working... (Esc to cancel)" if busy else "Type a message and press Enter (Ctrl+Q to quit)"
        )
        if not busy:
            self.prompt_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        self.prompt_input.value = ""

        if not prompt or self.agent is None:
            return
        if prompt.lower() == "exit":
            self.exit()
            return

        user_line = Text()
        user_line.append("\nUser: ", style="bold blue")
        user_line.append(prompt)
        self.chat_log.write(user_line)

        self._set_busy(True)
        self.run_turn(prompt)

    # ---------- agent turn ----------

    @work(exclusive=True, group="turn")
    async def run_turn(self, prompt: str) -> None:
        turn_messages = []
        payload = {"messages": [{"role": "user", "content": prompt}]}
        try:
            while True:
                interrupts = []
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

                requests = extract_action_requests(interrupts)
                if not requests:
                    break

                # The agent paused for approval: ask, then resume the same run.
                decisions = []
                for request in requests:
                    approved = await self._ask_approval(request)
                    decisions.append(make_decision(approved))
                    self.chat_log.write(
                        Text("  ✓ approved", style="green")
                        if approved
                        else Text("  ✗ rejected", style="red")
                    )
                payload = resume_command(decisions)

            response = assistant_response_text(turn_messages)
            if response is None:
                self.chat_log.write("[red]The agent finished without a text response.[/red]")
            else:
                self.chat_log.write(Text("AI:", style="bold magenta"))
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
            self.chat_log.write(Text(f"  → {_format_tool_call(call)}", style="dim cyan"))
        if getattr(msg, "type", None) == "tool":
            name = getattr(msg, "name", None) or "tool"
            failed = getattr(msg, "status", None) == "error" or str(msg.content).startswith("Error")
            if failed:
                self.chat_log.write(Text(f"  ✗ {name} failed", style="dim red"))
            else:
                self.chat_log.write(Text(f"  ✓ {name} finished", style="dim green"))

    def _write_warning(self, message: str) -> None:
        self.chat_log.write(Text(message, style="yellow"))

    # ---------- actions ----------

    def action_clear_log(self) -> None:
        self.chat_log.clear()

    def action_cancel(self) -> None:
        if self.prompt_input.disabled and self.agent is not None:
            self.workers.cancel_group(self, "turn")
            self.chat_log.write(Text("Request cancelled.", style="yellow"))


if __name__ == "__main__":
    AgentTUI().run()
