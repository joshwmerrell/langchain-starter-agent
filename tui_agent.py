import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Vertical
from rich.markdown import Markdown
from main import setup_agent, _assistant_response_text

class AgentTUI(App):
    CSS = """
    RichLog {
        height: 1fr;
        border: solid green;
        margin: 1;
    }
    Input {
        margin: 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            RichLog(id="chat_log", highlight=True, markup=True),
        )
        yield Input(placeholder="Type your message here and press Enter...")
        yield Footer()

    async def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat_log", RichLog)
        self.input = self.query_one(Input)
        self.chat_log.write("[bold green]Agent TUI Started![/bold green]")
        self.chat_log.write("Type 'exit' to quit.")
        
        # Initialize the agent
        self.chat_log.write("[yellow]Initializing agent...[/yellow]")
        self.agent, self.thread_config = await setup_agent()
        self.chat_log.write("[bold green]Agent Ready![/bold green]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        self.input.value = ""

        if not prompt:
            return

        if prompt.lower() == "exit":
            self.exit()
            return

        self.chat_log.write(f"\n[bold blue]User:[/bold blue] {prompt}")

        try:
            self.chat_log.write("[yellow]Thinking...[/yellow]")
            # We need to run the agent invocation in a way that doesn't block the TUI
            # Since agent.ainvoke is an async function, we can await it directly in this async handler
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                self.thread_config,
            )
            
            response = _assistant_response_text(result["messages"])
            
            if response is None:
                self.chat_log.write("[red]The agent finished without a text response.[/red]")
            else:
                self.chat_log.write("[bold magenta]AI:[/bold magenta]")
                # Textual's RichLog can take Markdown if we use it correctly, 
                # but for simplicity, we'll just write the text or use Markdown if supported.
                # RichLog.write supports rich text.
                self.chat_log.write(Markdown(response))

        except Exception as e:
            self.chat_log.write(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    app = AgentTUI()
    app.run()
