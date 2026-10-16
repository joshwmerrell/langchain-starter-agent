"""Live Ollama model selection in the CLI and TUI."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from main import run_agent, setup_agent, summarize_thread_memory
from tui_agent import AgentTUI
from wallace_models import load_model_choice, list_installed_models, save_model_choice
from wallace_themes import DARK_WORKING_COLOR


class ModelSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_choice_is_persisted_and_invalid_data_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_model_choice(root, "env:latest"), "env:latest")
            save_model_choice(root, "local/coder:latest")
            self.assertEqual(load_model_choice(root, "env:latest"), "local/coder:latest")
            path = root / ".wallace" / "model.json"
            self.assertEqual(json.loads(path.read_text()), {"model": "local/coder:latest"})
            path.write_text('{"model": "bad\nname"}')
            self.assertEqual(load_model_choice(root, "env:latest"), "env:latest")
            with self.assertRaises(ValueError):
                save_model_choice(root, "bad name")

    async def test_installed_models_come_from_ollama_tags(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"models": [
                {"name": "zeta:latest"},
                {"model": "alpha:latest"},
                {"name": "zeta:latest"},
                {"name": "bad name"},
            ]})

        client_type = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        with patch("wallace_models.httpx.AsyncClient", side_effect=lambda **kwargs: client_type(transport=transport, **kwargs)):
            models = await list_installed_models("http://localhost:11434")
        self.assertEqual(models, ["alpha:latest", "zeta:latest"])
        self.assertEqual(requests[0].url.path, "/api/tags")

    async def test_selected_model_reaches_agent_and_memory_summarizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model_choice(root, "saved:latest")
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
                    with patch("main.ChatOllama") as model_type:
                        with patch("main.create_agent", return_value=object()):
                            await setup_agent()
                            self.assertEqual(model_type.call_args.kwargs["model"], "saved:latest")
                            await setup_agent(model_name="override:latest")
                            self.assertEqual(model_type.call_args.kwargs["model"], "override:latest")
                            model_type.return_value.ainvoke = AsyncMock(
                                return_value=SimpleNamespace(content="Earlier user preference is saved.")
                            )
                            await summarize_thread_memory("", [{"role": "user", "content": "Remember this"}], 200)
                            self.assertEqual(model_type.call_args.kwargs["model"], "saved:latest")
                            await summarize_thread_memory("", [{"role": "user", "content": "Remember this"}], 200, model_name="override:latest")
                            self.assertEqual(model_type.call_args.kwargs["model"], "override:latest")

    async def test_tui_switches_live_model_without_losing_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app._handle_thread_command("/new model-choice-thread")
                        old_agent = object()
                        new_agent = object()
                        app.agent = old_agent
                        app.thread_config = {"thread": "old"}
                        thread_id = app.thread_id
                        app.conversation_store.append("user", "Keep this question")
                        with patch("tui_agent.list_installed_models", new=AsyncMock(return_value=["new:latest"])):
                            with patch("tui_agent.setup_agent", new=AsyncMock(return_value=(new_agent, {"thread": "new"}))):
                                worker = app.change_model("new:latest")
                                await worker.wait()
                        self.assertIs(app.agent, new_agent)
                        self.assertEqual(app.model_name, "new:latest")
                        self.assertEqual(app.thread_config, {"thread": "new"})
                        self.assertEqual(app.thread_id, thread_id)
                        self.assertEqual(app.conversation_store.load_all()[0]["content"], "Keep this question")
                        self.assertEqual(load_model_choice(root, "env:latest"), "new:latest")
                        self.assertFalse(app._busy)
                    self.assertEqual(AgentTUI().model_name, "new:latest")

    async def test_tui_model_command_and_shortcut_are_discoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        with patch.object(app, "change_model") as change:
                            app._handle_thread_command("/model local/coder:latest")
                            change.assert_called_once_with("local/coder:latest")
                        app._set_busy(False)
                        with patch.object(app, "show_models") as show:
                            await pilot.press("f10")
                            show.assert_called_once_with()
                        app._set_busy(False)

    async def test_tui_rejects_unknown_or_failed_model_without_changing_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        old_agent = object()
                        app.agent = old_agent
                        old_name = app.model_name
                        with patch("tui_agent.list_installed_models", new=AsyncMock(return_value=["new:latest"])):
                            await app.change_model("missing:latest").wait()
                            with patch("tui_agent.setup_agent", new=AsyncMock(side_effect=RuntimeError("rebuild failed"))):
                                await app.change_model("new:latest").wait()
                        self.assertIs(app.agent, old_agent)
                        self.assertEqual(app.model_name, old_name)
                        self.assertFalse((root / ".wallace" / "model.json").exists())
                        self.assertFalse(app._busy)

    async def test_cli_model_command_rebuilds_and_remembers_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                setup = AsyncMock(side_effect=[(object(), {"old": 1}), (object(), {"new": 1})])
                with patch("main.setup_agent", new=setup):
                    with patch("main.list_installed_models", new=AsyncMock(return_value=["new:latest"])):
                        with patch("builtins.input", side_effect=["/model new:latest", "exit"]):
                            with patch("main.console"):
                                await run_agent()
                self.assertEqual(setup.await_count, 2)
                self.assertEqual(setup.await_args.kwargs["model_name"], "new:latest")
                self.assertEqual(load_model_choice(root, "env:latest"), "new:latest")

    async def test_cli_failed_rebuild_does_not_save_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                setup = AsyncMock(side_effect=[(object(), {"old": 1}), RuntimeError("rebuild failed")])
                with patch("main.setup_agent", new=setup):
                    with patch("main.list_installed_models", new=AsyncMock(return_value=["new:latest"])):
                        with patch("builtins.input", side_effect=["/model new:latest", "exit"]):
                            with patch("main.console"):
                                await run_agent()
            self.assertFalse((root / ".wallace" / "model.json").exists())

    async def test_dark_working_text_is_red_and_other_theme_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/theme dark")
                        app._set_busy(True)
                        await pilot.pause()
                        self.assertEqual(app.thinking_panel.styles.color.hex.lower(), DARK_WORKING_COLOR)
                        working_segments = [
                            segment
                            for row in range(app.thinking_panel.size.height)
                            for segment in app.thinking_panel.render_line(row)
                            if "WALLACE" in segment.text
                        ]
                        self.assertTrue(working_segments)
                        self.assertEqual(
                            working_segments[0].style.color.get_truecolor().red,
                            255,
                        )
                        app._handle_thread_command("/theme grayscale")
                        await pilot.pause()
                        self.assertNotEqual(app.thinking_panel.styles.color.hex.lower(), DARK_WORKING_COLOR)
                        app._set_busy(False)


if __name__ == "__main__":
    unittest.main()
