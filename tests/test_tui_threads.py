import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.markdown import Markdown
from rich.text import Text
from textual.selection import Selection
from textual.theme import Theme
from textual.widgets import Button, Header

from tui_agent import AgentTUI, ApprovalScreen
from wallace_mcp import all_mcp_servers
from wallace_themes import (
    DARK_ERROR_COLOR,
    DARK_HEADING_COLOR,
    DARK_TEXT_ACCENTS,
    DARK_WARNING_COLOR,
    DARK_WORKING_COLOR,
    DEFAULT_THEME,
    WALLACE_PRESETS,
    discover_omarchy_themes,
    rich_palette,
)


class TUIThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_commands_rebuild_live_agent_and_keep_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/new mcp-work")
                        app.conversation_store.append("user", "Keep this conversation")
                        thread_id = app.thread_id
                        app.agent = object()
                        rebuilt = object()
                        with patch("tui_agent.setup_agent", new=AsyncMock(return_value=(rebuilt, {"thread": "rebuilt"}))) as setup:
                            await app.change_mcp("add", "files", "https://example.com/mcp", ("drive",)).wait()
                            self.assertIs(app.agent, rebuilt)
                            self.assertEqual(app.thread_id, thread_id)
                            self.assertEqual(app.conversation_store.load_all()[0]["content"], "Keep this conversation")
                            self.assertEqual(setup.await_count, 1)
                            await app.change_mcp("off", "files").wait()
                            self.assertFalse(all_mcp_servers(root)[1].enabled)
                            await app.change_mcp("off", "langchain-docs").wait()
                            self.assertFalse(all_mcp_servers(root)[0].enabled)
                            self.assertEqual(setup.await_count, 3)
                        with patch.object(app, "change_mcp") as change:
                            app._handle_thread_command("/mcp toggle files")
                            change.assert_called_once_with("on", "files", "", ())
                        await pilot.press("f11")
                        self.assertTrue(any(
                            "MCP servers:" in getattr(entry[0], "plain", str(entry[0]))
                            for entry in app.chat_log._theme_entries
                        ))
                    self.assertFalse(all_mcp_servers(root)[0].enabled)
                    self.assertFalse(all_mcp_servers(root)[1].enabled)

    async def test_failed_mcp_rebuild_rolls_back_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        old_agent = object()
                        app.agent = old_agent
                        with patch("tui_agent.setup_agent", new=AsyncMock(side_effect=RuntimeError("failed rebuild"))):
                            await app.change_mcp("add", "files", "https://example.com/mcp").wait()
                        self.assertIs(app.agent, old_agent)
                        self.assertEqual([item.name for item in all_mcp_servers(root)], ["langchain-docs"])
                        self.assertFalse(app._busy)

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
                            for name in ("hacker", "retro", "elegant", "dark", "vaporwave", "grayscale", "night", "red"):
                                app._handle_thread_command(f"/theme {name}")
                                self.assertEqual(app.theme, f"wallace-{name}")
                            app._handle_thread_command("/theme night-mode")
                            self.assertEqual(app.theme, "wallace-night")
                            app._handle_thread_command("/theme red-light")
                            self.assertEqual(app.theme, "wallace-red")
                            app._handle_thread_command("/theme omarchy-custom")
                            self.assertEqual(app.theme, "omarchy-custom")
                            app._handle_thread_command("/theme ansi-dark")
                            self.assertEqual(app.theme, "ansi-dark")
                            self.assertIsNone(app.get_theme("wallace-simple"))
                            self.assertIsNone(app.get_theme("wallace-minimal"))
                            self.assertIsNone(app.get_theme("wallace-future"))
                            self.assertIsNone(app.get_theme("wallace-night-mode"))
                            self.assertIsNone(app.get_theme("wallace-red-light"))
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

    async def test_renamed_saved_themes_keep_the_users_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".wallace").mkdir()
            preference = root / ".wallace" / "ui.json"
            with patch("tui_agent.SAFE_ROOT", root):
                for old, new in (
                    ("wallace-night-mode", "wallace-night"),
                    ("wallace-red-light", "wallace-red"),
                ):
                    preference.write_text(json.dumps({"theme": old}))
                    self.assertEqual(AgentTUI().theme, new)
                preference.write_text(json.dumps({"theme": "wallace-future"}))
                self.assertEqual(AgentTUI().theme, DEFAULT_THEME)

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
                        self.assertEqual(app.sub_title, "Monday, 2026-10-12 12:34  ☀")
                        self.assertIn("Waxing Crescent", app.query_one(Header).tooltip)
                        self.assertIn(app.sub_title, app.query_one(Header).format_title().plain)
                        with patch("tui_agent.datetime") as clock:
                            clock.now.return_value = fixed.replace(hour=22)
                            app._update_header_datetime()
                        self.assertEqual(app.sub_title, "Monday, 2026-10-12 22:34  ☽")

    async def test_emoji_focused_theme_opts_into_emoji_sky_icons(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    app.register_theme(Theme(
                        name="wallace-emoji-test", primary="#7dd3fc",
                        variables={"sky-icon-style": "emoji"},
                    ))
                    async with app.run_test():
                        fixed = datetime(2026, 10, 12, 22, 34).astimezone()
                        with patch("tui_agent.datetime") as clock:
                            clock.now.return_value = fixed
                            app.theme = "wallace-emoji-test"
                            self.assertEqual(app.sub_title, "Monday, 2026-10-12 22:34  🌒")
                            app.theme = DEFAULT_THEME
                            self.assertEqual(app.sub_title, "Monday, 2026-10-12 22:34  ☽")

    async def test_startup_and_conversation_turns_have_blank_line_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        def entries():
                            return [
                                content.plain if isinstance(content, Text) else content
                                for content, *_ in app.chat_log._theme_entries
                            ]

                        startup = entries()
                        self.assertEqual(startup[0], "Welcome to Wallace.")
                        self.assertEqual(startup[1], "")
                        self.assertTrue(startup[2].startswith("Thread:"))
                        self.assertEqual(startup[3], "")
                        self.assertEqual(startup[4], "Initializing Wallace...")

                        app.chat_log.clear()
                        app._conversation_history = [
                            {"role": "user", "content": "First question"},
                            {"role": "assistant", "content": "First answer"},
                            {"role": "user", "content": "Next question"},
                        ]
                        app._restore_conversation()
                        await pilot.pause()
                        history = entries()
                        self.assertEqual(history[0], "Restored 1 previous turn(s).")
                        for label in ("User: First question", "AI:", "User: Next question"):
                            self.assertEqual(history[history.index(label) - 1], "")
                        self.assertTrue(any(not line.text.strip() for line in app.chat_log.lines))

    async def test_prompt_has_editor_selection_editing_and_undo_shortcuts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        prompt = app.prompt_input
                        prompt.disabled = False
                        prompt.focus()
                        prompt.value = "alpha beta gamma"
                        prompt.cursor_position = len(prompt.value)

                        await pilot.press("ctrl+a")
                        self.assertEqual(prompt.selected_text, "alpha beta gamma")
                        await pilot.press("left")
                        self.assertEqual(prompt.cursor_position, 0)
                        await pilot.press("shift+right")
                        self.assertEqual(prompt.selected_text, "a")

                        prompt.cursor_position = len(prompt.value)
                        await pilot.press("ctrl+shift+left")
                        self.assertEqual(prompt.selected_text, "gamma")
                        await pilot.press("ctrl+backspace")
                        self.assertEqual(prompt.value, "alpha beta ")
                        await pilot.press("ctrl+z")
                        self.assertEqual(prompt.value, "alpha beta gamma")
                        self.assertEqual(prompt.selected_text, "gamma")
                        await pilot.press("ctrl+shift+z")
                        self.assertEqual(prompt.value, "alpha beta ")

                        prompt.cursor_position = 0
                        await pilot.press("ctrl+delete")
                        self.assertEqual(prompt.value, "beta ")
                        await pilot.press("ctrl+z")
                        self.assertEqual(prompt.value, "alpha beta ")
                        await pilot.press("ctrl+y")
                        self.assertEqual(prompt.value, "beta ")

                        prompt.load_history(["first command", "second command"])
                        prompt.value = "draft"
                        prompt.cursor_position = len(prompt.value)
                        await pilot.press("up")
                        self.assertEqual(prompt.value, "second command")
                        await pilot.press("down")
                        self.assertEqual(prompt.value, "draft")
                        await pilot.press("ctrl+z")
                        self.assertEqual(prompt.value, "draft")

                        prompt.value = "alpha beta"
                        prompt.cursor_position = 0
                        await pilot.press("ctrl+shift+right")
                        self.assertEqual(prompt.selected_text, "alpha ")
                        await pilot.press("ctrl+c")
                        self.assertEqual(app.clipboard, "alpha ")
                        await pilot.press("ctrl+x")
                        self.assertEqual(prompt.value, "beta")
                        await pilot.press("ctrl+v")
                        self.assertEqual(prompt.value, "alpha beta")
                        await pilot.press("ctrl+z")
                        self.assertEqual(prompt.value, "beta")

                        prompt.cursor_position = len(prompt.value)
                        await pilot.press("ctrl+shift+home")
                        self.assertEqual(prompt.selected_text, "beta")
                        await pilot.press("ctrl+end")
                        self.assertEqual(prompt.cursor_position, len(prompt.value))
                        await pilot.press("alt+backspace")
                        self.assertEqual(prompt.value, "")

                        selection_style = prompt.get_component_rich_style("input--selection")
                        self.assertNotEqual(selection_style.bgcolor, prompt.rich_style.bgcolor)

    async def test_red_removes_markdown_syntax_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/theme red")
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

    def test_red_palette_uses_only_red_and_black(self):
        theme = next(theme for theme in WALLACE_PRESETS if theme.name == "wallace-red")
        colors = {**theme.to_color_system().generate(), **theme.variables}
        for name, value in colors.items():
            if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                self.assertEqual(value[3:], "0000", f"{name} is not red-only: {value}")

    async def test_grayscale_removes_markdown_syntax_hues(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/theme grayscale")
                        app.chat_log.write(Markdown("```python\nprint('hi')\n```"))
                        await pilot.pause()
                        segments = [
                            segment
                            for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                        ]
                        self.assertTrue(any("print" in segment.text for segment in segments))
                        for segment in segments:
                            if segment.style:
                                for color in (segment.style.color, segment.style.bgcolor):
                                    if color is not None:
                                        channels = color.get_truecolor()
                                        self.assertEqual(channels.red, channels.green)
                                        self.assertEqual(channels.green, channels.blue)
                            if "print" in segment.text and segment.style and segment.style.color:
                                self.assertGreaterEqual(segment.style.color.get_truecolor().red, 176)

    async def test_dark_keeps_black_surfaces_and_uses_red_only_for_text_hierarchy(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/theme dark")
                        palette = rich_palette(app.current_theme)
                        app.chat_log.write(app._user_chat_line("My message is red"))
                        app.chat_log.write(Text("Warning:", style=palette["warning"]))
                        app.chat_log.write(Text("Error:", style=palette["error"]))
                        app.chat_log.write(Markdown("```python\nprint('hi')\n```"))
                        await pilot.pause()
                        for selector in ("#chat_log", "#prompt", "#thinking"):
                            self.assertEqual(app.query_one(selector).styles.background.hex, "#000000")
                        self.assertEqual(app.query_one("#prompt").styles.color.hex.lower(), DARK_WORKING_COLOR)
                        for selector in ("#thread_title", "#workspace_title", "#workspace_activity_title"):
                            self.assertEqual(app.query_one(selector).styles.color.hex.lower(), DARK_HEADING_COLOR)
                        current_thread = app.query_one("#thread_list").render()
                        self.assertTrue(any(
                            span.style.foreground.hex.lower() == DARK_WORKING_COLOR
                            for span in current_thread.spans
                            if span.style.foreground is not None
                        ))
                        segments = [
                            segment
                            for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                        ]
                        self.assertTrue(any("print" in segment.text for segment in segments))
                        colored = {}
                        for segment in segments:
                            if segment.style:
                                if segment.style.color is not None:
                                    channels = segment.style.color.get_truecolor()
                                    hex_color = f"#{channels.red:02x}{channels.green:02x}{channels.blue:02x}"
                                    if channels.red != channels.green or channels.green != channels.blue:
                                        self.assertIn(hex_color, DARK_TEXT_ACCENTS)
                                    if segment.text.strip() in {"User:", "My message is red", "Warning:", "Error:"}:
                                        colored[segment.text.strip()] = hex_color
                                    if "print" in segment.text:
                                        self.assertEqual(channels.red, channels.green)
                                        self.assertEqual(channels.green, channels.blue)
                                if segment.style.bgcolor is not None:
                                    channels = segment.style.bgcolor.get_truecolor()
                                    self.assertEqual((channels.red, channels.green, channels.blue), (0, 0, 0))
                        self.assertEqual(colored, {
                            "User:": DARK_WORKING_COLOR,
                            "My message is red": DARK_WORKING_COLOR,
                            "Warning:": DARK_WARNING_COLOR,
                            "Error:": DARK_ERROR_COLOR,
                        })
                        app._handle_thread_command("/theme grayscale")
                        await pilot.pause()
                        grayscale_labels = [
                            segment for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                            if segment.text.strip() in {"User:", "My message is red"}
                        ]
                        self.assertTrue(grayscale_labels)
                        for label in grayscale_labels:
                            channels = label.style.color.get_truecolor()
                            self.assertEqual(channels.red, channels.green)
                            self.assertEqual(channels.green, channels.blue)
                        app._handle_thread_command("/theme dark")
                        await pilot.pause()
                        dark_labels = [
                            segment for row in range(app.chat_log.size.height)
                            for segment in app.chat_log.render_line(row)
                            if segment.text.strip() == "User:"
                        ]
                        self.assertTrue(any(
                            f"#{label.style.color.get_truecolor().red:02x}"
                            f"{label.style.color.get_truecolor().green:02x}"
                            f"{label.style.color.get_truecolor().blue:02x}" == DARK_WORKING_COLOR
                            for label in dark_labels
                        ))
                        app.push_screen(ApprovalScreen("execute_command"))
                        await pilot.pause()
                        for button in app.screen.query(Button):
                            self.assertEqual(button.styles.background.hex, "#000000")
                            self.assertNotEqual(button.styles.color.hex, "#000000")

    def test_grayscale_and_dark_palettes_have_no_hues(self):
        for theme_name in ("wallace-grayscale", "wallace-dark"):
            theme = next(theme for theme in WALLACE_PRESETS if theme.name == theme_name)
            colors = {**theme.to_color_system().generate(), **theme.variables}
            for name, value in colors.items():
                if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                    self.assertEqual(value[1:3], value[3:5], f"{theme_name} {name} is not grayscale: {value}")
                    self.assertEqual(value[3:5], value[5:7], f"{theme_name} {name} is not grayscale: {value}")
        dark = next(theme for theme in WALLACE_PRESETS if theme.name == "wallace-dark")
        self.assertEqual((dark.background, dark.surface, dark.panel), ("#000000",) * 3)
        dark_text = rich_palette(dark)
        self.assertEqual(dark_text["primary"], DARK_HEADING_COLOR)
        self.assertEqual(dark_text["user"], DARK_WORKING_COLOR)
        self.assertEqual(dark_text["warning"], DARK_WARNING_COLOR)
        self.assertEqual(dark_text["error"], DARK_ERROR_COLOR)
        self.assertEqual(dark_text["assistant"].lower(), "#a8a8a8")
        vaporwave = next(theme for theme in WALLACE_PRESETS if theme.name == "wallace-vaporwave")
        self.assertEqual(
            (vaporwave.background, vaporwave.surface, vaporwave.primary, vaporwave.secondary),
            ("#101431", "#1b2350", "#ff4bd8", "#56eaff"),
        )

    def test_new_theme_text_colors_contrast_with_background(self):
        def luminance(value: str) -> float:
            channels = [int(value[index:index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
            return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

        for theme in WALLACE_PRESETS:
            if theme.name not in {"wallace-dark", "wallace-vaporwave", "wallace-grayscale"}:
                continue
            background = luminance(theme.background)
            for role in ("foreground", "primary", "secondary"):
                foreground = luminance(getattr(theme, role))
                ratio = (max(background, foreground) + 0.05) / (min(background, foreground) + 0.05)
                self.assertGreaterEqual(ratio, 4.5, f"{theme.name} {role} contrast is {ratio:.2f}")
            if theme.name == "wallace-dark":
                for role in ("primary", "user", "warning", "error"):
                    foreground = luminance(rich_palette(theme)[role])
                    ratio = (foreground + 0.05) / (background + 0.05)
                    self.assertGreaterEqual(ratio, 4.5, f"dark accent {role} contrast is {ratio:.2f}")

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
