"""Regression checks for host execution, conversation scope, and configuration."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from main import (
    EMPTY_RESPONSE_RETRY_PROMPT, MALFORMED_TOOL_CALL_RETRY_PROMPT,
    is_malformed_tool_call_error, malformed_retry_payload,
)
from wallace_execution import TurnRunner, _repeated_call, assistant_response_text
from wallace_memory import ConversationStore, bind_conversation, reset_conversation, search_conversation
from wallace_planning import parse_request


class ExecutionAndMemoryTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, agent, **kwargs):
        return TurnRunner(
            agent, {}, approve=kwargs.pop("approve", AsyncMock(return_value=True)),
            malformed_retry=malformed_retry_payload, malformed_error=is_malformed_tool_call_error,
            malformed_prompt=MALFORMED_TOOL_CALL_RETRY_PROMPT,
            empty_prompt=EMPTY_RESPONSE_RETRY_PROMPT, **kwargs,
        )

    async def test_continuation_budget_is_finite(self):
        class Agent:
            calls = 0

            async def astream(self, payload, config, stream_mode):
                self.calls += 1
                yield {"model": {"messages": [AIMessage(content=f"Still pending {self.calls}.")]}}

        agent = Agent()
        result = await self.runner(agent, continue_work=lambda *_: "Finish the pending check.").run("Do the work")
        self.assertEqual(agent.calls, 3)
        self.assertEqual(result.response, "Still pending 3.")

    async def test_rejection_disables_automatic_work_continuation(self):
        class Agent:
            calls = 0

            async def astream(self, payload, config, stream_mode):
                self.calls += 1
                if self.calls == 1:
                    yield {"__interrupt__": [{"action_requests": [{"name": "write_safe_file", "args": {}}]}]}
                else:
                    self.decision = payload
                    yield {"model": {"messages": [AIMessage(content="The change was rejected.")]}}

        agent = Agent()
        followup = AsyncMock(return_value="Try again")
        runner = self.runner(agent, approve=AsyncMock(return_value=False), continue_work=followup)
        await runner.run("Make a change")
        self.assertEqual(agent.calls, 2)
        self.assertIsInstance(agent.decision, Command)
        self.assertEqual(agent.decision.resume["decisions"], [{"type": "reject"}])
        followup.assert_not_awaited()

    def test_final_text_must_follow_the_latest_instruction(self):
        self.assertIsNone(assistant_response_text([
            HumanMessage(content="Implement the feature"), AIMessage(content="I will start."),
            HumanMessage(content="Continue the unverified work."),
        ], after_instruction=True))

    def test_repeated_tool_calls_stop_only_without_an_intervening_change(self):
        messages = []
        for index in range(3):
            messages += [AIMessage(content="", tool_calls=[{
                "name": "read_safe_file", "args": {"path": "a.py"}, "id": str(index),
            }]), ToolMessage(content="text", tool_call_id=str(index))]
        call = {"name": "read_safe_file", "args": {"path": "a.py"}, "id": "next"}
        self.assertTrue(_repeated_call(call, messages))
        messages += [AIMessage(content="", tool_calls=[{
            "name": "replace_in_file", "args": {"path": "a.py"}, "id": "edit",
        }]), ToolMessage(content="updated", tool_call_id="edit")]
        self.assertFalse(_repeated_call(call, messages))

    async def test_search_is_bound_to_the_current_thread_and_marks_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ConversationStore(Path(directory) / "one.json")
            second = ConversationStore(Path(directory) / "two.json")
            first.append("user", "Use crimson for the moon")
            first.append("assistant", "The crimson moon was installed")
            second.append("user", "Crimson private second thread")
            token = bind_conversation(first)
            try:
                result = search_conversation.invoke({"query": "crimson"})
            finally:
                reset_conversation(token)
            self.assertIn("Use crimson", result)
            self.assertIn("Wallace (unverified)", result)
            self.assertNotIn("private second thread", result)
            self.assertIn("no conversation", search_conversation.invoke({"query": "crimson"}))

    async def test_compaction_uses_one_model_call_and_preserves_raw_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "thread.json")
            for index in range(100):
                store.append("user", f"fact {index}: " + "x" * 100)
            summarizer = AsyncMock(return_value="The user recorded numbered facts.")
            context, _ = await store.memory_context("current task", summarizer, max_chars=800)
            self.assertEqual(summarizer.await_count, 1)
            self.assertEqual(len(store.load_all()), 100)
            self.assertLessEqual(len(context), 800)
            self.assertIn("fact 99", context)

    def test_failed_command_side_effects_still_require_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = parse_request(Path(directory), "Implement the requested behavior")
            plan.record_progress([{
                "id": "command", "name": "execute_command", "target": "python change.py",
                "ok": False, "outcome": "Exit code: 1", "mutation": True,
                "changed_paths": ["app.py"],
            }], response="The command failed after writing a file.")
            self.assertEqual(plan.unchecked_targets(), ["app.py"])
            self.assertEqual(plan.status, "active")

    def test_dotenv_model_settings_load_before_defaults_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(Path(__file__).parents[1] / "wallace_config.py", root / "wallace_config.py")
            (root / "wallace_runtime.py").write_text("from pathlib import Path\nAPP_ROOT = Path(__file__).parent\n")
            (root / ".env").write_text("OLLAMA_MODEL=from-dotenv:test\nWALLACE_NUM_CTX=8192\n")
            environment = {key: value for key, value in os.environ.items()
                           if key not in {"OLLAMA_MODEL", "WALLACE_NUM_CTX", "PYTHONPATH"}}
            command = [sys.executable, "-B", "-c", "import wallace_config as c; print(c.OLLAMA_MODEL, c.OLLAMA_NUM_CTX)"]
            result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "from-dotenv:test 8192")
            environment["OLLAMA_MODEL"] = "from-environment:test"
            result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "from-environment:test 8192")
