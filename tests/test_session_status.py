"""Provider-neutral session usage and host-owned status command."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from main import _invoke_with_approvals, run_agent
from tools import READ_ONLY_TOOLS, TOOLS, get_session_status
from tui_agent import AgentTUI
from wallace_status import SessionUsage, StatusContext, bind_status, reset_status
from wallace_tool_selection import select_tools


class SessionStatusTests(unittest.TestCase):
    def test_generic_provider_usage_is_deduplicated_and_contains_no_keys(self):
        usage = SessionUsage()
        first = AIMessage(
            content="answer", id="response-1",
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )
        usage.record_message(first, "future-provider:future-model")
        usage.record_message(first, "future-provider:future-model")
        updated = AIMessage(
            content="answer", id="response-1",
            usage_metadata={"input_tokens": 14, "output_tokens": 4, "total_tokens": 18},
        )
        usage.record_message(updated, "future-provider:future-model")
        fallback = AIMessage(
            content="second", id="response-2",
            response_metadata={"token_usage": {
                "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                "api_key": "TEST_SECRET_MUST_NOT_APPEAR",
            }},
        )
        usage.record_message(fallback, "future-provider:future-model")
        totals = usage.by_model["future-provider:future-model"]
        self.assertEqual((totals.responses, totals.input_tokens, totals.output_tokens, totals.total_tokens),
                         (2, 19, 6, 25))
        context = StatusContext(usage, "test_thread", "future-provider:future-model",
                                "informed-trust", 7, Path("/safe/workspace"))
        token = bind_status(context)
        try:
            result = get_session_status.invoke({})
        finally:
            reset_status(token)
        self.assertIn("25 total tokens", result)
        self.assertIn("informed-trust (7s auto-approve)", result)
        self.assertIn("Remaining account quota/reset: unavailable", result)
        self.assertNotIn("TEST_SECRET_MUST_NOT_APPEAR", result)
        self.assertIn("get_session_status", READ_ONLY_TOOLS)
        self.assertIn("get_session_status", {
            tool.name for tool in select_tools("Show system status and session token usage", TOOLS)
        })


class SessionStatusIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_model_tool_receives_active_status_context(self):
        context = StatusContext(
            SessionUsage(), "thread_async", "future-provider:model", "no-trust", 5,
            Path("/safe/workspace"),
        )
        token = bind_status(context)
        try:
            result = await get_session_status.ainvoke({})
        finally:
            reset_status(token)
        self.assertIn("thread_async", result)
        self.assertIn("future-provider:model", result)

    async def test_cli_status_and_turn_count_provider_usage(self):
        class FakeAgent:
            async def ainvoke(self, payload, config):
                return {"messages": [AIMessage(
                    content="Done", id="cloud-response",
                    usage_metadata={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                )]}

        usage = SessionUsage()
        await _invoke_with_approvals(
            FakeAgent(), {}, "say done", model_name="another-provider:model", usage_tracker=usage,
        )
        self.assertEqual(usage.by_model["another-provider:model"].total_tokens, 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))):
                    with patch("builtins.input", side_effect=["/status", "exit"]):
                        with patch("main.console") as console:
                            await run_agent()
            printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
            self.assertIn("Wallace session status", printed)
            self.assertIn("Remaining account quota/reset: unavailable", printed)

    async def test_tui_status_tracks_streamed_usage_without_double_counting_state(self):
        message = AIMessage(
            content="Done", id="streamed-response",
            usage_metadata={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
        )

        class FakeAgent:
            async def astream(self, payload, config, stream_mode):
                yield {"agent": {"messages": [message]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [message]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.APP_ROOT", root), patch.object(AgentTUI, "initialize_agent", lambda self: None):
                app = AgentTUI()
                async with app.run_test():
                    app.agent = FakeAgent()
                    app.thread_config = {}
                    await app.run_turn("say done", user_prompt=None).wait()
                    app._handle_thread_command("/status")
                    visible = "\n".join(getattr(item[0], "plain", str(item[0])) for item in app.chat_log._theme_entries)
                    self.assertIn("24 total tokens", visible)
                    self.assertEqual(sum(item.responses for item in app.session_usage.by_model.values()), 1)


if __name__ == "__main__":
    unittest.main()
