"""User-owned trust modes and workspace editor behavior."""

import asyncio
import json
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main as wallace_main
from main import _ask_cli_approval, _handle_thread_command
from tui_agent import AgentTUI, ApprovalScreen, WorkspaceFileScreen
from wallace_planning import ConversationThreadManager, parse_new_thread_arguments


class TrustAndEditorTests(unittest.IsolatedAsyncioTestCase):
    async def test_trust_modes_default_persist_and_reject_invalid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ConversationThreadManager(root)
            main = manager.ensure("main")
            self.assertEqual(main.trust_mode, "no-trust")
            self.assertEqual(main.approval_timeout_seconds, 5)
            title, mode = parse_new_thread_arguments(["--trust", "informed-trust", "Build", "Wallace"])
            self.assertEqual((title, mode), ("Build Wallace", "informed-trust"))
            thread = manager.create(title, trust_mode=mode)
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).trust_mode, "informed-trust")
            self.assertEqual(thread.approval_timeout_seconds, 5)
            manager.set_trust_mode(thread.thread_id, "informed-trust", 7)
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).approval_timeout_seconds, 7)
            for invalid in (0, 301, "1.5", "bad", True):
                with self.assertRaises(ValueError):
                    manager.set_trust_mode(thread.thread_id, "informed-trust", invalid)
            with self.assertRaises(ValueError):
                manager.set_trust_mode(thread.thread_id, "yes-trust", 7)
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).approval_timeout_seconds, 7)
            manager.set_trust_mode(thread.thread_id, "yes-trust")
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).trust_mode, "yes-trust")
            with self.assertRaises(ValueError):
                manager.set_trust_mode(thread.thread_id, "unsafe")
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).trust_mode, "yes-trust")
            with self.assertRaises(ValueError):
                parse_new_thread_arguments(["--trust", "unsafe", "Bad"])

            index = json.loads(manager.index_path.read_text(encoding="utf-8"))
            for record in index["threads"]:
                if record["thread_id"] == thread.thread_id:
                    record["trust_mode"] = "unexpected-future-mode"
            manager.index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).trust_mode, "no-trust")
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).approval_timeout_seconds, 7)

            # Older index entries with no timeout receive the new default.
            for record in index["threads"]:
                record.pop("approval_timeout_seconds", None)
            manager.index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(ConversationThreadManager(root).resolve(thread.thread_id).approval_timeout_seconds, 5)

    async def test_tui_modes_use_user_controlled_thread_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root), patch.object(AgentTUI, "initialize_agent", lambda self: None):
                app = AgentTUI()
                async with app.run_test() as pilot:
                    self.assertIn("trust: no-trust", str(app.chat_log._theme_entries[2][0]))
                    app._handle_thread_command("/new --trust informed-trust Build Wallace")
                    self.assertEqual(app.thread_manager.resolve(app.thread_id).trust_mode, "informed-trust")
                    self.assertEqual(app.thread_manager.resolve(app.thread_id).approval_timeout_seconds, 5)
                    app._handle_thread_command("/trust informed-trust 7")
                    self.assertEqual(app.thread_manager.resolve(app.thread_id).approval_timeout_seconds, 7)
                    request = {"name": "execute_command", "args": {"command": "pwd"}}
                    pending = asyncio.create_task(app._ask_approval(request))
                    await pilot.pause()
                    self.assertIsInstance(app.screen, ApprovalScreen)
                    self.assertEqual(app.screen.countdown_seconds, 7)
                    self.assertIn("Auto-approve", app.screen.query_one("#approval_hint").render().plain)
                    await pilot.press("n")
                    self.assertFalse(await pending)

                    app._handle_thread_command("/trust yes-trust")
                    self.assertTrue(await app._ask_approval({"name": "configured_mcp_tool", "args": {}}))
                    self.assertNotIsInstance(app.screen, ApprovalScreen)
                    app._handle_thread_command("/switch main")
                    self.assertEqual(app.thread_manager.resolve(app.thread_id).trust_mode, "no-trust")
                    app._handle_thread_command("/trust no-trust")
                reopened = AgentTUI()
                self.assertEqual(reopened.thread_manager.resolve("Build Wallace").trust_mode, "yes-trust")
                self.assertEqual(reopened.thread_manager.resolve("Build Wallace").approval_timeout_seconds, 7)

    async def test_cli_can_set_informed_trust_countdown(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationThreadManager(Path(directory))
            manager.ensure("main")
            with patch("main.console") as console:
                _handle_thread_command("/trust informed-trust 9", manager, "main")
                self.assertEqual(manager.resolve("main").approval_timeout_seconds, 9)
                _handle_thread_command("/trust informed-trust", manager, "main")
                self.assertEqual(manager.resolve("main").approval_timeout_seconds, 5)
                _handle_thread_command("/trust yes-trust 9", manager, "main")
                self.assertEqual(manager.resolve("main").trust_mode, "informed-trust")
                self.assertTrue(any("timeout can only" in str(call.args[0])
                                    for call in console.print.call_args_list if call.args))

    async def test_informed_countdown_auto_approves_but_interaction_pauses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)), patch.object(AgentTUI, "initialize_agent", lambda self: None):
                app = AgentTUI()
                decisions = []
                async with app.run_test() as pilot:
                    app.push_screen(ApprovalScreen("execute_command\n$ pwd", countdown_seconds=10), decisions.append)
                    await pilot.pause()
                    auto_screen = app.screen
                    auto_screen._deadline = time.monotonic() - 1
                    auto_screen._update_countdown()
                    await pilot.pause()
                    self.assertEqual(decisions, [True])

                    app.push_screen(ApprovalScreen("execute_command\n$ pwd", countdown_seconds=10), decisions.append)
                    await pilot.pause()
                    await pilot.hover("#approval_detail")
                    self.assertIn("paused", app.screen.query_one("#approval_hint").render().plain.lower())
                    paused_screen = app.screen
                    paused_screen._deadline = time.monotonic() - 1
                    paused_screen._update_countdown()
                    await pilot.pause()
                    self.assertIs(app.screen, paused_screen)
                    self.assertEqual(decisions, [True])
                    await pilot.press("n")
                    await pilot.pause()
                    self.assertEqual(decisions, [True, False])

    async def test_file_picker_and_vim_only_open_safe_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / ".env").write_text("FAKE=hidden\n", encoding="utf-8")
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root), patch.object(AgentTUI, "initialize_agent", lambda self: None):
                app = AgentTUI()
                async with app.run_test() as pilot:
                    app.action_edit_file()
                    await pilot.pause()
                    self.assertIsInstance(app.screen, WorkspaceFileScreen)
                    self.assertIn("main.py", app.screen.paths)
                    self.assertNotIn(".env", app.screen.paths)
                    await pilot.press("escape")
                    await pilot.pause()

                    with patch("tui_agent.shutil.which", return_value="/usr/bin/nvim"), patch.object(app, "suspend", return_value=nullcontext()), patch("tui_agent.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
                        app._open_workspace_editor(".env")
                        app._open_workspace_editor("../elsewhere")
                        run.assert_not_called()
                        app._open_workspace_editor("main.py")
                        run.assert_called_once_with(
                            ["/usr/bin/nvim", "--", str(workspace / "main.py")], cwd=workspace, check=False,
                        )
                    self.assertFalse(app._busy)

    async def test_picker_selection_launches_editor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "main.py").write_text("pass\n", encoding="utf-8")
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root), patch.object(AgentTUI, "initialize_agent", lambda self: None):
                app = AgentTUI()
                async with app.run_test() as pilot:
                    with patch("tui_agent.shutil.which", return_value="/usr/bin/nvim"), patch.object(app, "suspend", return_value=nullcontext()), patch("tui_agent.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
                        app.action_edit_file()
                        await pilot.pause()
                        await pilot.press("enter")
                        await pilot.pause()
                        run.assert_called_once_with(
                            ["/usr/bin/nvim", "--", str(workspace / "main.py")], cwd=workspace, check=False,
                        )
                    self.assertFalse(app._busy)

    async def test_cli_yes_trust_approves_without_prompt_and_informed_non_tty_rejects(self):
        request = {"name": "execute_command", "args": {"command": "pwd"}}
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            self.assertTrue(_ask_cli_approval(request, trust_mode="yes-trust"))
            with patch("main.sys.stdin.isatty", return_value=False):
                self.assertFalse(_ask_cli_approval(request, trust_mode="informed-trust"))

    @unittest.skipIf(wallace_main.termios is None, "CLI countdown needs termios")
    async def test_cli_informed_trust_uses_five_second_default(self):
        request = {"name": "execute_command", "args": {"command": "pwd"}}
        with patch("main.sys.stdin.isatty", return_value=True), \
             patch("main.sys.stdin.fileno", return_value=0), \
             patch("main.termios.tcgetattr", return_value=[]), \
             patch("main.termios.tcsetattr"), \
             patch("main.tty.setcbreak"), \
             patch("main.time.monotonic", side_effect=[0, 0, 5]), \
             patch("main.select.select", return_value=([], [], [])) as select_call, \
             patch("main.console") as console:
            self.assertTrue(_ask_cli_approval(request, trust_mode="informed-trust"))
        self.assertEqual(select_call.call_args.args[-1], 1)
        self.assertTrue(any("Approved after 5 seconds" in str(call.args[0])
                            for call in console.print.call_args_list if call.args))


if __name__ == "__main__":
    unittest.main()
