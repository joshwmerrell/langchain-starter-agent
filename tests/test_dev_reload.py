"""The watcher should preserve a working TUI when new source is broken."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from watchfiles import Change

from dev_tui import MCP_CONFIG_PATH, _changes_smoke, _config_change, _wait_for_ready
from wallace_mcp import add_mcp_server


class DevReloadTests(unittest.TestCase):
    def test_failed_startup_preflight_blocks_reload(self):
        with patch("dev_tui.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "ImportError: missing module"
            run.return_value.stdout = ""
            self.assertFalse(_changes_smoke())
        with patch("dev_tui.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertTrue(_changes_smoke())

    def test_private_mcp_config_change_triggers_reload(self):
        with patch("dev_tui.tui_applied_current_config", return_value=False):
            self.assertTrue(_config_change({(Change.modified, str(MCP_CONFIG_PATH))}))
        self.assertFalse(_config_change({(Change.modified, str(Path("main.py").resolve()))}))

    def test_tui_applied_mcp_change_does_not_restart_running_tui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".wallace" / "mcp_servers.json"
            add_mcp_server(root, "files", "https://example.com/mcp", live_applied=True)
            changes = {(Change.modified, str(config))}
            with patch("dev_tui.ROOT", root), patch("dev_tui.MCP_CONFIG_PATH", config):
                self.assertFalse(_config_change(changes))
                config.write_text('{"servers": []}')
                self.assertTrue(_config_change(changes))
                add_mcp_server(root, "cli-added", "https://example.com/other")
                self.assertTrue(_config_change(changes))

    def test_crashed_child_is_not_considered_ready(self):
        class Child:
            returncode = 1

            def poll(self):
                return self.returncode

        self.assertFalse(_wait_for_ready(Child(), timeout=0.1))


if __name__ == "__main__":
    unittest.main()
