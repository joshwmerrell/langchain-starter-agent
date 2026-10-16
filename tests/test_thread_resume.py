"""The selected conversation should survive normal starts and TUI reloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from main import run_agent
from tui_agent import AgentTUI
from wallace_planning import ConversationThreadManager


class ThreadResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_thread_persists_and_removed_thread_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ConversationThreadManager(root)
            manager.ensure("main")
            second = manager.create("Second")
            self.assertEqual(manager.active_thread_id(), "main")
            manager.select_thread(second.thread_id)
            self.assertEqual(ConversationThreadManager(root).active_thread_id(), second.thread_id)
            with self.assertRaises(ValueError):
                manager.select_thread("missing")
            manager.remove(second.thread_id)
            self.assertEqual(ConversationThreadManager(root).active_thread_id(), "main")
            saved = json.loads(manager.active_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["thread_id"], "main")

    async def test_tui_reopens_the_last_selected_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    first = AgentTUI()
                    async with first.run_test():
                        first._handle_thread_command("/new Notes")
                        selected_id = first.thread_id
                        first.conversation_store.append("user", "Remember this")
                    reloaded = AgentTUI()
                    self.assertEqual(reloaded.thread_id, selected_id)
                    self.assertEqual(reloaded.conversation_store.load_all()[0]["content"], "Remember this")
                    async with reloaded.run_test():
                        reloaded._handle_thread_command("/switch main")
                    self.assertEqual(AgentTUI().thread_id, "main")

    async def test_cli_restores_selected_thread_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ConversationThreadManager(root, scope="workspace")
            manager.ensure("main")
            second = manager.create("Second")
            manager.select_thread(second.thread_id)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))):
                    with patch("builtins.input", side_effect=["exit"]):
                        with patch("main.console"):
                            await run_agent()
            self.assertEqual(ConversationThreadManager(root, scope="workspace").active_thread_id(), second.thread_id)


if __name__ == "__main__":
    unittest.main()
