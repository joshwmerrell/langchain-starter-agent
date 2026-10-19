"""Bounded local-model IO, exact edits, and command diagnostics."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tools


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.scope = patch("tools.SAFE_ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_small_file_is_exact_and_ranged_read_has_a_useful_header(self):
        (self.root / "note.txt").write_text("first\nsecond\n")
        self.assertEqual(tools.read_safe_file.invoke({"path": "note.txt"}), "first\nsecond\n")
        self.assertEqual(
            tools.read_safe_file.invoke({"path": "note.txt", "start_line": 2, "end_line": 2}),
            "[lines 2-2 of 2]\nsecond\n",
        )
        self.assertIn("end_line", tools.read_safe_file.invoke({
            "path": "note.txt", "start_line": 2, "end_line": 1,
        }))
        self.assertIn("only 2 lines", tools.read_safe_file.invoke({
            "path": "note.txt", "start_line": 3,
        }))

    def test_giant_single_line_is_streamed_and_strictly_bounded(self):
        (self.root / "minified.json").write_text("[" + "1," * 250_000 + "0]")
        with patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
            result = tools.read_safe_file.invoke({"path": "minified.json"})
        self.assertLessEqual(len(result), tools.MAX_READ_CHARS)
        self.assertIn("line 1 truncated", result)
        self.assertIn("targeted command", result)

    def test_multiline_read_continues_at_an_unread_line(self):
        text = "".join(f"line {number}: " + "x" * 990 + "\n" for number in range(30))
        (self.root / "large.txt").write_text(text)
        result = tools.read_safe_file.invoke({"path": "large.txt"})
        self.assertLessEqual(len(result), tools.MAX_READ_CHARS)
        self.assertTrue("truncated" in result or "start_line=" in result)

    def test_tree_and_search_limit_output_without_exposing_private_paths(self):
        for number in range(350):
            (self.root / f"{number:03d}-".ljust(110, "x")).write_text("match " + "x" * 400)
        (self.root / ".env").write_text("private-match")
        (self.root / "loop").symlink_to("loop")
        listing = tools.get_project_tree.invoke({"max_depth": 10000})
        self.assertLessEqual(len(listing), tools.MAX_READ_CHARS)
        self.assertNotIn(".env", listing)
        self.assertIn("omitted", listing)
        result = tools.search_in_files.invoke({"pattern": "match"})
        self.assertLessEqual(len(result), tools.MAX_OUTPUT_CHARS)
        self.assertIn("narrow", result)
        self.assertNotIn("private-match", result)
        self.assertTrue(tools.read_safe_file.invoke({"path": "loop"}).startswith("Error:"))

    def test_search_reports_skipped_large_files(self):
        (self.root / "huge.txt").write_text("match" * (tools.MAX_SEARCH_FILE_BYTES // 5 + 1))
        result = tools.search_in_files.invoke({"pattern": "match"})
        self.assertIn("skipped 1 files", result)
        self.assertNotIn("No matches found", result)

    def test_exact_edits_reject_ambiguity_oversized_files_and_stale_contents(self):
        target = self.root / "note.txt"
        target.write_text("same\nsame\n")
        result = tools.replace_in_file.invoke({"path": "note.txt", "old_text": "same", "new_text": "new"})
        self.assertIn("matches 2 places", result)
        self.assertEqual(target.read_text(), "same\nsame\n")
        with patch("tools.MAX_EDIT_FILE_BYTES", 4):
            result = tools.replace_in_file.invoke({"path": "note.txt", "old_text": "same", "new_text": "new"})
        self.assertIn("edit/check limit", result)
        target.write_text("changed by another writer\n")
        with self.assertRaisesRegex(ValueError, "changed during"):
            tools._atomic_write_text(target, "new", expected_text="stale contents")
        self.assertEqual(target.read_text(), "changed by another writer\n")
        self.assertEqual(list(self.root.glob(".note.txt.*.tmp")), [])

    def test_exact_edit_preserves_mode_and_accepts_partial_read_header(self):
        target = self.root / "script.sh"
        target.write_text("first\nsecond\nthird\n")
        target.chmod(0o750)
        result = tools.replace_in_file.invoke({
            "path": "script.sh", "old_text": "[lines 2-2]\nsecond\n", "new_text": "updated\n",
        })
        self.assertEqual(result, "Edited script.sh.")
        self.assertEqual(target.read_text(), "first\nupdated\nthird\n")
        self.assertEqual(target.stat().st_mode & 0o777, 0o750)

    def test_approval_previews_are_bounded(self):
        (self.root / "note.txt").write_text("old\n" * 10_000)
        self.assertLessEqual(len(tools.preview_write("note.txt", "new\n" * 10_000)), tools.MAX_READ_CHARS)
        self.assertLessEqual(len(tools.preview_line_edit("note.txt", 1, 80, "new" * 5000)), tools.MAX_READ_CHARS)

    def test_zero_discovered_tests_is_not_success(self):
        with patch("tools._module_available", return_value=False), patch(
            "tools._run_command", return_value="Exit code: 0\nSTDERR:\nRan 0 tests in 0.000s\nOK",
        ):
            result = tools.run_tests.invoke({})
        self.assertTrue(result.startswith("Error: no tests were discovered"))

    def test_mcp_status_is_valid_bounded_json_with_explicit_tool_counts(self):
        server = SimpleNamespace(
            name="files", transport="http", url="https://example.com/mcp", enabled=True,
            auth_env=None, builtin=False, keywords=("files",),
        )
        with patch("tools.all_mcp_servers", return_value=[server]), patch(
            "tools.connection_status", return_value="connected",
        ), patch("tools.connection_details", return_value={
            "detail": "", "tools": [f"tool_{number}" for number in range(1000)],
        }):
            result = tools.list_mcp_connections.invoke({})
        self.assertLessEqual(len(result), tools.MAX_READ_CHARS)
        entry = json.loads(result)["servers"][0]
        self.assertEqual(entry["tool_count"], 1000)
        self.assertEqual(entry["tools_omitted"], 1000 - len(entry["tools"]))

    def test_external_python_environment_is_mounted_read_only(self):
        environment = self.root / "runtime"
        environment.mkdir()
        (environment / "pyvenv.cfg").write_text("home = /usr/bin\n")
        workspace = self.root / "workspace"
        workspace.mkdir()
        with patch("tools.APP_ROOT", self.root / "checkout"), patch("tools.sys.prefix", str(environment)), patch(
            "tools.shutil.which", return_value="/usr/bin/bwrap",
        ):
            argv = tools._sandbox_command([sys.executable, "-V"], shell=False, cwd=workspace, sandbox_root=workspace)
        triples = [argv[index:index + 3] for index in range(len(argv) - 2)]
        self.assertIn(["--ro-bind", str(environment), str(environment)], triples)
        self.assertNotIn(["--bind", str(environment), str(environment)], triples)
        self.assertNotIn(["--ro-bind", str(self.root), str(self.root)], triples)

    def test_generic_credentials_are_not_inherited_by_tool_commands(self):
        with patch.dict(os.environ, {"API_KEY": "test-fixture", "TOKEN": "test-fixture", "SAFE_MARKER": "yes"}):
            environment = tools._tool_environment()
        self.assertNotIn("API_KEY", environment)
        self.assertNotIn("TOKEN", environment)
        self.assertEqual(environment["SAFE_MARKER"], "yes")


class CommandIOTests(unittest.IsolatedAsyncioTestCase):
    def test_sync_output_is_bounded_and_keeps_error_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            script = "import sys; print('BEGIN'); print('x'*500000); print('FINAL FAILURE',file=sys.stderr); sys.exit(7)"
            with patch("tools.SAFE_ROOT", Path(directory)), patch(
                "tools._sandbox_command", return_value=[sys.executable, "-c", script],
            ):
                result = tools._run_command("controlled test")
        self.assertLessEqual(len(result), tools.MAX_OUTPUT_CHARS)
        self.assertTrue(result.startswith("Exit code: 7"))
        self.assertIn("BEGIN", result)
        self.assertIn("FINAL FAILURE", result)

    def test_sync_timeout_preserves_partial_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tools.SAFE_ROOT", Path(directory)), patch(
                "tools._sandbox_command", return_value=[
                    sys.executable, "-c", "import time; print('started check',flush=True); time.sleep(2)",
                ],
            ):
                result = tools._run_command("controlled test", timeout=0.1)
        self.assertIn("timed out", result)
        self.assertIn("started check", result)

    async def test_async_output_is_drained_without_losing_unicode_or_error_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            script = "import sys; print('BEGIN λ'); print('x'*500000); print('FINAL FAILURE λ',file=sys.stderr); sys.exit(9)"
            with patch("tools.SAFE_ROOT", Path(directory)), patch(
                "tools._sandbox_command", return_value=[sys.executable, "-c", script],
            ):
                result = await tools.execute_command_async("controlled test")
        self.assertLessEqual(len(result), tools.MAX_OUTPUT_CHARS)
        self.assertTrue(result.startswith("Exit code: 9"))
        self.assertIn("BEGIN λ", result)
        self.assertIn("FINAL FAILURE λ", result)

    async def test_async_timeout_keeps_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tools.SAFE_ROOT", Path(directory)), patch("tools.COMMAND_TIMEOUT", 0.1), patch(
                "tools._sandbox_command", return_value=[
                    sys.executable, "-c", "import time; print('started check',flush=True); time.sleep(2)",
                ],
            ):
                result = await tools.execute_command_async("controlled test")
        self.assertIn("timed out", result)
        self.assertIn("started check", result)

    def test_bounded_buffer_retains_both_ends(self):
        output = tools._OutputBuffer()
        for chunk in ["BEGIN\n", "x" * 100_000, "\nEND λ"]:
            output.append(chunk)
        self.assertLessEqual(len(output.text()), tools.MAX_OUTPUT_CHARS)
        self.assertTrue(output.text().startswith("BEGIN"))
        self.assertTrue(output.text().endswith("END λ"))
        self.assertLessEqual(len(output.head) + len(output.tail), 2 * tools.MAX_OUTPUT_CHARS)


if __name__ == "__main__":
    unittest.main()
