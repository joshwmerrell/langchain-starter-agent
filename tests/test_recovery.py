import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual.widgets import Input

from main import (
    EMPTY_RESPONSE_RETRY_PROMPT,
    MALFORMED_TOOL_CALL_RETRY_PROMPT,
    _invoke_with_approvals,
    describe_error,
    malformed_retry_payload,
)
from tui_agent import AgentTUI
from wallace_planning import load_plan, parse_request


MALFORMED = "llama-server returned invalid tool call arguments: unexpected end of JSON input"


class FakeAgent:
    def __init__(self, *, fail_count, checkpointed=True):
        self.fail_count = fail_count
        self.calls = []
        self.state_messages = [HumanMessage(content="original request")] if checkpointed else []

    async def ainvoke(self, payload, config):
        self.calls.append(payload)
        if len(self.calls) <= self.fail_count:
            raise RuntimeError(MALFORMED)
        return {"messages": [*self.state_messages, AIMessage(content="Progress reported.")]}

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.state_messages})


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_uses_checkpoint_without_replaying_original_prompt(self):
        agent = FakeAgent(fail_count=1)
        result = await _invoke_with_approvals(agent, {}, "Build the feature")
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.calls[0]["messages"][0]["content"], "Build the feature")
        self.assertEqual(agent.calls[1]["messages"], [
            {"role": "user", "content": MALFORMED_TOOL_CALL_RETRY_PROMPT}
        ])
        self.assertNotIn("_wallace_recovery_summary", result)

    async def test_second_failure_stops_without_model_written_handoff(self):
        agent = FakeAgent(fail_count=2)
        with self.assertRaisesRegex(RuntimeError, "invalid tool call arguments"):
            await _invoke_with_approvals(agent, {}, "Build the feature")
        self.assertEqual(len(agent.calls), 2)

    def test_malformed_error_reports_only_observed_parser_detail(self):
        error = RuntimeError(
            'llama-server returned invalid tool call arguments for "write_safe_file": '
            'unexpected end of JSON input; content={"private": "SECRET"}'
        )
        detail = describe_error(error)
        self.assertIn("write_safe_file", detail)
        self.assertIn("unexpected end of JSON input", detail)
        self.assertIn("unfinished", detail)
        self.assertNotIn("SECRET", detail)
        self.assertNotIn("content=", detail)

    async def test_original_prompt_is_restored_if_no_checkpoint_exists(self):
        agent = FakeAgent(fail_count=0, checkpointed=False)
        payload = await malformed_retry_payload(agent, {}, "Build the feature")
        self.assertIn("Build the feature", payload["messages"][0]["content"])
        self.assertIn(MALFORMED_TOOL_CALL_RETRY_PROMPT, payload["messages"][0]["content"])

    async def test_completed_tool_result_survives_checkpoint_retry(self):
        agent = FakeAgent(fail_count=1)
        agent.state_messages.extend([
            AIMessage(content="", tool_calls=[{
                "id": "cmd-1", "name": "execute_command", "args": {"command": "pwd"}
            }]),
            ToolMessage(content="Exit code: 0", tool_call_id="cmd-1", name="execute_command"),
        ])
        await _invoke_with_approvals(agent, {}, "Build the feature")
        self.assertEqual(len(agent.calls[1]["messages"]), 1)
        self.assertNotIn("Build the feature", agent.calls[1]["messages"][0]["content"])

    async def test_tui_stream_retry_does_not_replay_original_request(self):
        class StreamAgent:
            def __init__(self):
                self.calls = []
                self.messages = []

            async def astream(self, payload, config, stream_mode):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    call = AIMessage(content="", tool_calls=[
                        {"id": "cmd-1", "name": "execute_command", "args": {"command": "pwd"}}
                    ])
                    result = ToolMessage(content="Exit code: 0", tool_call_id="cmd-1", name="execute_command")
                    self.messages.extend([HumanMessage(content="Build the feature"), call, result])
                    yield {"tools": {"messages": [result]}}
                    raise RuntimeError(MALFORMED)
                answer = AIMessage(content="Work remains, but the command ran successfully.")
                self.messages.extend([HumanMessage(content=MALFORMED_TOOL_CALL_RETRY_PROMPT), answer])
                yield {"agent": {"messages": [answer]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": self.messages})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        agent = StreamAgent()
                        app.agent = agent
                        app.thread_config = {}
                        worker = app.run_turn("Build the feature")
                        await worker.wait()
                        self.assertEqual(len(agent.calls), 2)
                        self.assertEqual(agent.calls[1]["messages"], [
                            {"role": "user", "content": MALFORMED_TOOL_CALL_RETRY_PROMPT}
                        ])

    async def test_tui_second_failure_preserves_plan_without_fake_answer(self):
        class StreamAgent:
            def __init__(self):
                self.calls = []
                self.messages = []

            async def astream(self, payload, config, stream_mode):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    call = AIMessage(content="", tool_calls=[{
                        "id": "read-1", "name": "read_safe_file", "args": {"path": "main.py"}
                    }])
                    result = ToolMessage(content="source read", tool_call_id="read-1", name="read_safe_file")
                    self.messages.extend([HumanMessage(content="Implement main.py"), call, result])
                    yield {"tools": {"messages": [result]}}
                raise RuntimeError(
                    'llama-server returned invalid tool call arguments for "write_safe_file": '
                    'unexpected end of JSON input; content={"private": "SECRET"}'
                )

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": self.messages})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        plan = parse_request(root, "Implement main.py", session_id=app.thread_id)
                        app._active_plan = plan
                        worker = app.run_turn("Implement main.py", user_prompt="Implement main.py")
                        await worker.wait()
                        self.assertEqual(len(app.agent.calls), 2)
                        self.assertEqual(app.conversation_store.load(), [])
                        saved = load_plan(root, plan.session_id, plan.plan_id)
                        self.assertEqual(saved.status, "active")
                        self.assertEqual(saved.tool_history[-1]["name"], "read_safe_file")
                        self.assertIn("write_safe_file", saved.last_error)
                        self.assertNotIn("SECRET", saved.last_error)

    async def test_tui_continuation_does_not_inject_prior_assistant_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = []

            def capture_turn(self, prompt, **kwargs):
                captured.append(prompt)

            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    with patch.object(AgentTUI, "run_turn", capture_turn):
                        app = AgentTUI()
                        async with app.run_test():
                            app.agent = object()
                            parse_request(root, "Implement main.py", session_id=app.thread_id)
                            app._conversation_history = [
                                {"role": "user", "content": "Implement main.py"},
                                {"role": "assistant", "content": "Missing step_id in /workspace."},
                            ]
                            app.on_input_submitted(Input.Submitted(app.prompt_input, "Continue with the active plan"))
                            self.assertEqual(len(captured), 1)
                            self.assertIn("Original request: Implement main.py", captured[0])
                            self.assertNotIn("step_id", captured[0])
                            self.assertNotIn("/workspace", captured[0])

    async def test_tui_repetitive_answer_is_not_saved_as_completion(self):
        repeated = ("I will review the files before editing them.\n</think>\n\n" * 20)

        class StreamAgent:
            def __init__(self):
                self.messages = []
                self.calls = 0

            async def astream(self, payload, config, stream_mode):
                self.calls += 1
                answer = AIMessage(content=repeated)
                self.messages = [HumanMessage(content="Continue"), answer]
                yield {"agent": {"messages": [answer]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": self.messages})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        plan = parse_request(root, "Implement main.py", session_id=app.thread_id)
                        app._active_plan = plan
                        worker = app.run_turn("Continue", user_prompt="Continue")
                        await worker.wait()
                        self.assertEqual(app.agent.calls, 1)
                        self.assertEqual(app.conversation_store.load(), [])
                        saved = load_plan(root, plan.session_id, plan.plan_id)
                        self.assertEqual(saved.status, "active")
                        self.assertIn("repetitive response", saved.last_error)
                        self.assertEqual(saved.steps[1].status, "pending")

                        app.chat_log.clear()
                        app._conversation_history = [
                            {"role": "user", "content": "Continue"},
                            {"role": "assistant", "content": repeated},
                        ]
                        app._restore_conversation()
                        rendered = [getattr(entry[0], "plain", "") for entry in app.chat_log._theme_entries]
                        self.assertTrue(any("omitted from display" in item for item in rendered))
                        self.assertFalse(any(repeated in item for item in rendered))

    async def test_tui_requests_final_text_after_tool_only_turn(self):
        class StreamAgent:
            def __init__(self):
                self.calls = []
                self.messages = []

            async def astream(self, payload, config, stream_mode):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    call = AIMessage(content="I'll begin.", tool_calls=[
                        {"id": "read-1", "name": "read_safe_file", "args": {"path": "main.py"}}
                    ])
                    result = ToolMessage(content="file content", tool_call_id="read-1", name="read_safe_file")
                    self.messages.extend([HumanMessage(content="Review main.py"), call, result])
                    yield {"tools": {"messages": [result]}}
                else:
                    answer = AIMessage(content="I reviewed the file.")
                    self.messages.extend([HumanMessage(content=EMPTY_RESPONSE_RETRY_PROMPT), answer])
                    yield {"agent": {"messages": [answer]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": self.messages})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        agent = StreamAgent()
                        app.agent = agent
                        app.thread_config = {}
                        worker = app.run_turn("Review main.py")
                        await worker.wait()
                        self.assertEqual(len(agent.calls), 2)
                        self.assertEqual(agent.calls[1]["messages"], [
                            {"role": "user", "content": EMPTY_RESPONSE_RETRY_PROMPT}
                        ])

    async def test_tui_keeps_streamed_answer_when_checkpoint_lags(self):
        class StreamAgent:
            def __init__(self):
                self.calls = 0

            async def astream(self, payload, config, stream_mode):
                self.calls += 1
                yield {"agent": {"messages": [AIMessage(content="I reviewed the file.")]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [HumanMessage(content="Review main.py")]})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        app._conversation_history = [{"role": "user", "content": "Review main.py"}]
                        worker = app.run_turn("Review main.py", user_prompt="Review main.py")
                        await worker.wait()
                        self.assertEqual(app.agent.calls, 1)
                        self.assertEqual(app.conversation_store.load()[-1]["content"], "I reviewed the file.")

    async def test_tui_no_text_after_retry_has_factual_handoff(self):
        class StreamAgent:
            def __init__(self):
                self.calls = 0
                self.messages = []

            async def astream(self, payload, config, stream_mode):
                self.calls += 1
                if self.calls == 1:
                    call = AIMessage(content="", tool_calls=[{
                        "id": "read-1", "name": "read_safe_file", "args": {"path": "main.py"}
                    }])
                    result = ToolMessage(content="file content", tool_call_id="read-1", name="read_safe_file")
                    self.messages = [HumanMessage(content="Continue"), call, result]
                    yield {"tools": {"messages": [result]}}
                else:
                    answer = AIMessage(content="")
                    self.messages.extend([HumanMessage(content=EMPTY_RESPONSE_RETRY_PROMPT), answer])
                    yield {"agent": {"messages": [answer]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": self.messages})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        plan = parse_request(root, "Implement main.py", session_id=app.thread_id)
                        app._active_plan = plan
                        worker = app.run_turn("Continue", user_prompt="Continue")
                        await worker.wait()
                        self.assertEqual(app.agent.calls, 2)
                        saved = load_plan(root, plan.session_id, plan.plan_id)
                        self.assertEqual(saved.status, "active")
                        self.assertIn("Next plan step: Implement the requested work", saved.last_error)
                        self.assertIn("Last observed tool: read_safe_file", saved.last_error)
                        self.assertIn("No file edit was observed", saved.last_error)
                        self.assertEqual(app.conversation_store.load(), [])


if __name__ == "__main__":
    unittest.main()
