import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dependency_analysis import DependencyGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from main import (
    ConversationStore,
    MAX_HISTORY_MESSAGES,
    assistant_response_text,
    model_tool_activity,
    no_text_status,
    response_quality_error,
)
from tools import (
    SAFE_ROOT, _resolve, check_python_syntax, edit_file_lines, preview_line_edit,
    execute_command, read_safe_file, replace_in_file, run_tests, write_safe_file,
)


class CoreSafetyTests(unittest.TestCase):
    def test_syntax_checker_accepts_unambiguous_module_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.py").write_text("value = 1\n")
            with patch("tools.SAFE_ROOT", root):
                self.assertEqual(
                    check_python_syntax.invoke({"path": "example"}),
                    "OK: example.py has valid Python syntax.",
                )
                self.assertIn("not an accessible Python file", check_python_syntax.invoke({
                    "path": ".env",
                }))

    def test_absent_removal_is_not_run_or_reported_as_success(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tools.SAFE_ROOT", root), patch("tools._run_command") as run:
                result = execute_command.invoke({"command": "rm -rf test_site"})
                self.assertIn("already absent; removal was not run", result)
                run.assert_not_called()
                (root / "test_site").mkdir()
                run.return_value = "Exit code: 0"
                self.assertEqual(
                    execute_command.invoke({"command": "rm -rf test_site"}),
                    "Exit code: 0",
                )
                run.assert_called_once()

    def test_run_tests_discovers_unittest_suite_from_project_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            with patch("tools.SAFE_ROOT", root), patch("tools._module_available", return_value=False):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    self.assertEqual(run_tests.invoke({}), "Exit code: 0")
            self.assertEqual(run.call_args.args[0][0:6], [
                sys.executable, "-m", "unittest", "discover", "-s", "tests",
            ])

    def test_line_editor_replaces_range_and_rejects_invalid_python(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "example.py"
            target.write_text("def working():\n    return 1\n")
            with patch("tools.SAFE_ROOT", root):
                preview = preview_line_edit("example.py", 2, 2, "    return 2")
                self.assertIn("-     return 1", preview)
                self.assertIn("+     return 2", preview)
                result = edit_file_lines.invoke({
                    "path": "example.py", "start_line": 2, "end_line": 2,
                    "new_text": "    return (",
                })
                self.assertIn("Python syntax error", result)
                self.assertEqual(target.read_text(), "def working():\n    return 1\n")
                result = edit_file_lines.invoke({
                    "path": "example.py", "start_line": 2, "end_line": 2,
                    "new_text": "    return 2",
                })
                self.assertEqual(result, "Edited example.py lines 2-2.")
                self.assertEqual(target.read_text(), "def working():\n    return 2\n")
                self.assertIn("invalid", edit_file_lines.invoke({
                    "path": "example.py", "start_line": 4, "end_line": 4,
                    "new_text": "pass",
                }))

    def test_line_editor_keeps_private_files_blocked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("placeholder")
            with patch("tools.SAFE_ROOT", root):
                self.assertIn("Access denied", edit_file_lines.invoke({
                    "path": ".env", "start_line": 1, "end_line": 1,
                    "new_text": "changed",
                }))
                self.assertIn("not writable", preview_line_edit(".env", 1, 1, "changed"))

    def test_replace_accepts_display_header_from_ranged_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "example.py"
            target.write_text("def working():\n    return 1\n")
            with patch("tools.SAFE_ROOT", root):
                ranged = read_safe_file.invoke({
                    "path": "example.py", "start_line": 2, "end_line": 2,
                })
                self.assertTrue(ranged.startswith("[lines 2-2 of 2]"))
                result = replace_in_file.invoke({
                    "path": "example.py", "old_text": ranged,
                    "new_text": "    return 2\n",
                })
            self.assertEqual(result, "Edited example.py.")
            self.assertEqual(target.read_text(), "def working():\n    return 2\n")

    def test_write_safe_file_rejects_invalid_python_without_creating_or_overwriting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "example.py"
            with patch("tools.SAFE_ROOT", root):
                result = write_safe_file.invoke({"path": "example.py", "content": "def broken(:\n"})
                self.assertIn("Python syntax error", result)
                self.assertIn("File was not changed", result)
                self.assertFalse(target.exists())

                result = write_safe_file.invoke({"path": "example.py", "content": "return 1\n"})
                self.assertIn("'return' outside function", result)
                self.assertFalse(target.exists())

                valid = "def working():\n    return 1\n"
                self.assertIn("Created example.py", write_safe_file.invoke({
                    "path": "example.py", "content": valid,
                }))
                result = write_safe_file.invoke({
                    "path": "example.py", "content": "def broken(:\n",
                })
                self.assertIn("Python syntax error", result)
                self.assertEqual(target.read_text(), valid)

    def test_replace_in_file_rejects_invalid_python_without_changing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "example.py"
            target.write_text("def working():\n    return 1\n")
            with patch("tools.SAFE_ROOT", root):
                result = replace_in_file.invoke({
                    "path": "example.py", "old_text": "return 1", "new_text": "return (",
                })
                self.assertIn("Python syntax error", result)
                self.assertEqual(target.read_text(), "def working():\n    return 1\n")

                result = replace_in_file.invoke({
                    "path": "example.py", "old_text": "return 1", "new_text": "return 2",
                })
                self.assertEqual(result, "Edited example.py.")
                self.assertEqual(target.read_text(), "def working():\n    return 2\n")

    def test_syntax_failure_is_recorded_as_failed_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.py").write_text("return 1\n")
            with patch("tools.SAFE_ROOT", root):
                result = check_python_syntax.invoke({"path": "broken.py"})
            self.assertTrue(result.startswith("Error: Python syntax error"))
            activity = model_tool_activity([
                AIMessage(content="", tool_calls=[{
                    "id": "check-1", "name": "check_python_syntax",
                    "args": {"path": "broken.py"},
                }]),
                ToolMessage(content=result, tool_call_id="check-1", name="check_python_syntax"),
            ])
            self.assertFalse(activity[0]["ok"])
            self.assertEqual(activity[0]["outcome"], "failed")

    def test_skipped_directories_are_not_readable(self):
        self.assertIsNone(_resolve(".venv/pyvenv.cfg"))
        self.assertIsNone(_resolve("node_modules/example.js"))
        self.assertIsNotNone(_resolve("main.py"))

    def test_history_is_bounded_to_complete_turns(self):
        history = []
        for index in range(MAX_HISTORY_MESSAGES + 8):
            history.extend(
                [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            )
        trimmed = ConversationStore.trim(history)
        self.assertLessEqual(len(trimmed), MAX_HISTORY_MESSAGES)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed[-1]["role"], "assistant")

    def test_dependency_graph_uses_local_package_names(self):
        graph = DependencyGraph(SAFE_ROOT).build()
        self.assertIn("main", graph.module_paths)
        self.assertIn("tools", graph.module_paths)
        self.assertIn("tools", graph.modules["main"])
        self.assertNotIn("ast", graph.modules["tools"])

    def test_assistant_response_survives_cumulative_stream_updates(self):
        messages = [
            HumanMessage(content="run the command"),
            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "execute_command", "args": {}}]),
            ToolMessage(content="Exit code: 0", tool_call_id="call-1"),
            AIMessage(content=[{"type": "text", "text": "The command completed."}]),
            HumanMessage(content="replayed request"),
        ]
        self.assertEqual(assistant_response_text(messages), "The command completed.")

    def test_preliminary_text_and_tool_output_are_not_final_responses(self):
        messages = [
            HumanMessage(content="implement the change"),
            AIMessage(content="I'll start now.", tool_calls=[
                {"id": "read-1", "name": "read_safe_file", "args": {"path": "main.py"}}
            ]),
            ToolMessage(content="file body", tool_call_id="read-1", name="read_safe_file"),
        ]
        self.assertIsNone(assistant_response_text(messages))
        messages.append(AIMessage(content="The change is incomplete."))
        self.assertEqual(assistant_response_text(messages), "The change is incomplete.")

    def test_repetitive_model_output_is_rejected_without_echoing_it(self):
        line = "I'll review the relevant files before making this change."
        repeated = (f"{line}\n</think>\n\n" * 8).strip()
        issue = response_quality_error(repeated)
        self.assertIsNotNone(issue)
        self.assertIn("repetitive response", issue)
        self.assertNotIn(line, issue)
        self.assertIsNone(response_quality_error("I inspected main.py and found the issue."))

    def test_copied_prior_reply_and_single_line_word_salad_are_rejected(self):
        previous = (
            "I'm Wallace, your all-around assistant. Here are my capabilities: "
            + " ".join(f"capability{index}" for index in range(90))
        )
        copied = previous + "I will now answer the user's new question."
        self.assertIn(
            "copied an earlier reply",
            response_quality_error(
                copied,
                previous_assistant=previous,
                current_request="What is the date?",
            ),
        )
        word_salad = "Capabilities include " + (
            "sorting ranking ordering filtering grouping clustering partitioning " * 90
        )
        self.assertIn("repetitive response", response_quality_error(word_salad))
        self.assertIsNone(response_quality_error(
            previous,
            previous_assistant=previous,
            current_request="Create an HTML page about you.",
        ))
        short_nonanswer = "I can answer directly. I'll provide the information in the next response. " * 3
        self.assertIn(
            "copied an earlier reply",
            response_quality_error(
                short_nonanswer,
                previous_assistant=short_nonanswer,
                current_request="What is 2+2?",
            ),
        )
        self.assertIsNone(response_quality_error("The answer is four."))

    def test_no_text_status_reports_observed_work_without_claiming_completion(self):
        class Plan:
            def next_pending(self):
                return type("Step", (), {"title": "Implement detection"})()

        status = no_text_status(Plan(), [{
            "name": "read_safe_file", "target": "main.py", "outcome": "ok", "ok": True,
        }])
        self.assertIn("request is unfinished", status)
        self.assertIn("Implement detection", status)
        self.assertIn("Last observed tool: read_safe_file (ok)", status)
        self.assertIn("No file edit was observed", status)

    def test_tool_journal_does_not_store_file_contents(self):
        messages = [
            AIMessage(content="", tool_calls=[
                {"id": "write-1", "name": "write_safe_file", "args": {
                    "path": "new.py", "content": "SECRET FILE BODY"
                }}
            ]),
            ToolMessage(content="Created new.py (16 characters).", tool_call_id="write-1", name="write_safe_file"),
        ]
        activity = model_tool_activity(messages)
        self.assertEqual(activity[0]["target"], "new.py")
        self.assertNotIn("SECRET FILE BODY", str(activity))


if __name__ == "__main__":
    unittest.main()
