import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from rich.markdown import Markdown
from textual.selection import Selection
from textual.theme import Theme
from textual.widgets import Button, Header

from tui_agent import AgentTUI, ApprovalScreen
from wallace_themes import DEFAULT_THEME, WALLACE_PRESETS, discover_omarchy_themes


class TUIThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_and_delete_thread_by_title(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app._handle_thread_command("/new build_wallace")
                        created_id = app.thread_id
                        self.assertNotEqual(created_id, "main")

                        app._handle_thread_command("/switch main")
                        self.assertEqual(app.thread_id, "main")

                        app._handle_thread_command("/switch build_wallace")
                        self.assertEqual(app.thread_id, created_id)
                        self.assertEqual(len(app.thread_manager.list_threads()), 2)

                        app._handle_thread_command("/delete build_wallace")
                        self.assertEqual(app.thread_id, "main")
                        self.assertEqual(
                            [record.thread_id for record in app.thread_manager.list_threads()],
                            ["main"],
                        )

    async def test_theme_switching_is_saved_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    omarchy_theme = Theme(name="omarchy-custom", primary="#aabbcc")
                    with patch("tui_agent.discover_omarchy_themes", return_value=[omarchy_theme]):
                        app = AgentTUI()
                        async with app.run_test() as pilot:
                            for name in ("hacker", "retro", "elegant", "future", "night-mode", "red-light"):
                                app._handle_thread_command(f"/theme {name}")
                                self.assertEqual(app.theme, f"wallace-{name}")
                            app._handle_thread_command("/theme night")
                            self.assertEqual(app.theme, "wallace-night-mode")
                            app._handle_thread_command("/theme red")
                            self.assertEqual(app.theme, "wallace-red-light")
                            app._handle_thread_command("/theme omarchy-custom")
                            self.assertEqual(app.theme, "omarchy-custom")
                            app._handle_thread_command("/theme ansi-dark")
                            self.assertEqual(app.theme, "ansi-dark")
                            self.assertIsNone(app.get_theme("wallace-simple"))
                            self.assertIsNone(app.get_theme("wallace-minimal"))
                            welcome_count = sum(
                                getattr(entry[0], "plain", "") == "Welcome to Wallace."
                                for entry in app.chat_log._theme_entries
                            )
                            self.assertEqual(welcome_count, 1)
                            app.chat_log.write("This long line should wrap inside the chat panel. " * 5)
                            await pilot.pause()
                            self.assertEqual(app.chat_log.max_scroll_x, 0)
                            await pilot.press("f9")
                            self.assertEqual(type(app.screen).__name__, "CommandPalette")
                        restored = AgentTUI()
                        self.assertEqual(restored.theme, "ansi-dark")

    async def test_removed_saved_theme_falls_back_to_classic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".wallace").mkdir()
            (root / ".wallace" / "ui.json").write_text(json.dumps({"theme": "wallace-minimal"}))
            with patch("tui_agent.SAFE_ROOT", root):
                app = AgentTUI()
                self.assertEqual(app.theme, DEFAULT_THEME)

    async def test_approval_hint_and_shortcuts_are_visible_and_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    decisions: list[bool] = []
                    async with app.run_test() as pilot:
                        app.push_screen(ApprovalScreen("execute_command\n$ pwd"), decisions.append)
                        await pilot.pause()
                        hint = app.screen.query_one("#approval_hint")
                        self.assertIn("Y = Yes", hint.render().plain)
                        self.assertIn("N = No", hint.render().plain)
                        approve = app.screen.query_one("#approve", Button)
                        reject = app.screen.query_one("#reject", Button)
                        self.assertIn("Yes (Y)", approve.label.plain)
                        self.assertIn("No (N)", reject.label.plain)
                        self.assertNotEqual(approve.styles.color, approve.styles.background)
                        self.assertNotEqual(reject.styles.color, reject.styles.background)
                        await pilot.press("y")
                        await pilot.pause()
                        self.assertEqual(decisions, [True])
                        app.push_screen(ApprovalScreen("replace_in_file"), decisions.append)
                        await pilot.pause()
                        await pilot.press("n")
                        await pilot.pause()
                        self.assertEqual(decisions, [True, False])

    async def test_header_shows_local_weekday_date_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        self.assertIsInstance(app.query_one(Header), Header)
                        fixed = datetime(2026, 10, 12, 12, 34).astimezone()
                        with patch("tui_agent.datetime") as clock:
                            clock.now.return_value = fixed
                            app._update_header_datetime()
                        self.assertEqual(app.sub_title, "Monday, 2026-10-12 12:34")
                        self.assertIn(app.sub_title, app.query_one(Header).format_title().plain)

    async def test_red_light_removes_markdown_syntax_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/theme red-light")
                        app.chat_log.write(Markdown("```python\nprint('hi')\n```"))
                        await pilot.pause()
                        segments = [
                            segment
                            for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                        ]
                        self.assertTrue(any("print" in segment.text for segment in segments))
                        for segment in segments:
                            if segment.style and segment.style.color and segment.style.color.triplet:
                                self.assertEqual(segment.style.color.triplet.green, 0)
                                self.assertEqual(segment.style.color.triplet.blue, 0)
                            if segment.style and segment.style.bgcolor and segment.style.bgcolor.triplet:
                                self.assertEqual(segment.style.bgcolor.triplet.green, 0)
                                self.assertEqual(segment.style.bgcolor.triplet.blue, 0)
                        app.screen.selections[app.chat_log] = Selection(None, None)
                        selected = [
                            segment
                            for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                            if "print" in segment.text
                        ]
                        self.assertTrue(selected)
                        self.assertTrue(any(
                            segment.style and segment.style.bgcolor
                            and segment.style.bgcolor.triplet.red > 5
                            for segment in selected
                        ))

    def test_red_light_palette_uses_only_red_and_black(self):
        theme = next(theme for theme in WALLACE_PRESETS if theme.name == "wallace-red-light")
        colors = {**theme.to_color_system().generate(), **theme.variables}
        for name, value in colors.items():
            if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                self.assertEqual(value[3:], "0000", f"{name} is not red-only: {value}")

    def test_installed_omarchy_colors_can_be_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            theme_dir = root / "system" / "custom"
            theme_dir.mkdir(parents=True)
            (theme_dir / "colors.toml").write_text(
                'mode = "dark"\naccent = "#aabbcc"\nbackground = "#101010"\n'
                'foreground = "#eeeeee"\ngreen = "#00ff00"\n'
            )
            themes = discover_omarchy_themes(root / "system", root / "user")
            self.assertEqual([theme.name for theme in themes], ["omarchy-custom"])
            self.assertEqual(themes[0].primary, "#aabbcc")


if __name__ == "__main__":
    unittest.main()
