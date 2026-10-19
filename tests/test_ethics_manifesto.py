"""Wallace should load the owner's Markdown ethics guidance for every agent."""

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from watchfiles import Change

from dev_tui import _manifesto_change, _manifesto_valid
from main import ethics_manifesto_prompt, setup_agent


class EthicsManifestoTests(unittest.IsolatedAsyncioTestCase):
    def test_project_manifesto_has_markdown_sections(self):
        manifesto = ethics_manifesto_prompt()
        for section in ("# Artificial Intelligence Ethics Manifesto", "## Human Agency", "## Do Good and Avoid Harm", "## Truth and Discernment"):
            self.assertIn(section, manifesto)
        project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        self.assertIn(
            "ethics_manifesto.md",
            project["tool"]["hatch"]["build"]["targets"]["wheel"]["include"],
        )

    async def test_agent_system_prompt_contains_current_manifesto(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ethics_manifesto.md").write_text("# Ethics\n\nPreserve human agency.\n", encoding="utf-8")
            with patch("main.APP_ROOT", root), patch("main._chat_model", return_value=object()):
                with patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
                    with patch("main._load_configured_mcp_tools", new=AsyncMock(return_value=[])):
                        with patch("main.create_agent", return_value=object()) as create:
                            await setup_agent(model_name="local:test")
            self.assertIn("Preserve human agency.", create.call_args.kwargs["system_prompt"])
            self.assertIn("Standing ethics manifesto", create.call_args.kwargs["system_prompt"])

    async def test_missing_manifesto_warns_and_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warnings = []
            with patch("main.APP_ROOT", root), patch("main._chat_model", return_value=object()):
                with patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
                    with patch("main._load_configured_mcp_tools", new=AsyncMock(return_value=[])):
                        with patch("main.create_agent", return_value=object()) as create:
                            await setup_agent(on_status=warnings.append, model_name="local:test")
            self.assertIn("Ethics manifesto could not be loaded", warnings[0])
            self.assertIn("Preserve the user's agency", create.call_args.kwargs["system_prompt"])

    def test_dev_watcher_reloads_valid_manifesto_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ethics_manifesto.md"
            changes = {(Change.modified, str(path))}
            with patch("dev_tui.ROOT", root), patch("dev_tui.ETHICS_MANIFESTO_PATH", path):
                self.assertTrue(_manifesto_change(changes))
                self.assertFalse(_manifesto_valid())
                path.write_text("# Ethics\n\nPreserve agency.\n", encoding="utf-8")
                self.assertTrue(_manifesto_valid())
                path.write_text("", encoding="utf-8")
                self.assertFalse(_manifesto_valid())


if __name__ == "__main__":
    unittest.main()
