"""Exercise the real agent graph offline, including middleware and host callbacks."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from main import (
    EMPTY_RESPONSE_RETRY_PROMPT, MALFORMED_TOOL_CALL_RETRY_PROMPT,
    focus_tools_on_request, is_malformed_tool_call_error, malformed_retry_payload,
    new_thread_config,
)
from tools import read_safe_file, reset_active_workspace, set_active_workspace, write_safe_file
from wallace_execution import TurnRunner, guard_tool_call, model_tool_activity
from wallace_planning import load_plan, parse_request, save_plan
from wallace_tool_selection import make_discovery_tool


class ScriptedChatModel(BaseChatModel):
    """Return prescribed messages while observing real schemas and tool results."""

    script: list[Any]
    seen_tools: list[list[str]] = Field(default_factory=list)
    seen_messages: list[list[Any]] = Field(default_factory=list)
    position: int = 0

    @property
    def _llm_type(self):
        return "wallace-offline-workflow-test"

    def bind_tools(self, tools, **kwargs):
        return self.bind(visible_tools=[item.name for item in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_tools.append(kwargs.get("visible_tools", []))
        self.seen_messages.append(list(messages))
        if self.position >= len(self.script):
            raise AssertionError("The graph requested an unexpected model response")
        response = self.script[self.position]
        self.position += 1
        if callable(response):
            response = response(messages)
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def agent_for(model, capabilities, *, interrupt_on=None):
    tools = [*capabilities, make_discovery_tool(capabilities)]
    return create_agent(
        model=model, tools=tools,
        middleware=[
            focus_tools_on_request,
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on or {}),
            guard_tool_call,
        ],
        checkpointer=InMemorySaver(),
    )


def runner_for(agent, *, approve=None, checkpoint=None):
    return TurnRunner(
        agent, new_thread_config(), approve=approve or AsyncMock(return_value=True),
        malformed_retry=malformed_retry_payload, malformed_error=is_malformed_tool_call_error,
        malformed_prompt=MALFORMED_TOOL_CALL_RETRY_PROMPT,
        empty_prompt=EMPTY_RESPONSE_RETRY_PROMPT, checkpoint=checkpoint,
    )


class AgentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_exposes_optional_tool_on_next_real_model_step(self):
        calls = []

        @tool
        def list_albums() -> str:
            """List photographs in the connected image library."""
            calls.append("listed")
            return "Album: red moon"

        model = ScriptedChatModel(script=[
            tool_call("discover_tools", {"query": "photographs"}, "discovery"),
            tool_call("list_albums", {}, "albums"),
            AIMessage(content="The image library contains the red moon album."),
        ])
        result = await runner_for(agent_for(model, [list_albums])).run("Find my pictures")
        self.assertNotIn("list_albums", model.seen_tools[0])
        self.assertIn("discover_tools", model.seen_tools[0])
        self.assertIn("list_albums", model.seen_tools[1])
        self.assertEqual(calls, ["listed"])
        self.assertIn("red moon", result.response)
        self.assertTrue(any(isinstance(message, ToolMessage) and message.name == "list_albums" for message in result.messages))

    async def test_rejected_actual_file_write_never_changes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "notes.txt"
            path.write_text("original", encoding="utf-8")
            model = ScriptedChatModel(script=[
                tool_call("write_safe_file", {"path": "notes.txt", "content": "replacement"}, "write"),
                AIMessage(content="The change was rejected, so notes.txt still contains its original text."),
            ])
            approve = AsyncMock(return_value=False)
            agent = agent_for(model, [write_safe_file], interrupt_on={
                "write_safe_file": {"allowed_decisions": ["approve", "reject"]},
            })
            token = set_active_workspace(root)
            try:
                result = await runner_for(agent, approve=approve).run("Replace notes.txt with replacement")
            finally:
                reset_active_workspace(token)
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            approve.assert_awaited_once()
            self.assertEqual(approve.await_args.args[0]["name"], "write_safe_file")
            rejected = [message for message in result.messages if isinstance(message, ToolMessage)]
            self.assertTrue(rejected)
            self.assertIn("reject", rejected[0].content.casefold())
            self.assertFalse(any(item["ok"] and item["name"] == "write_safe_file" for item in model_tool_activity(result.messages)))

    async def test_invalid_arguments_are_returned_to_model_and_corrected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("local evidence", encoding="utf-8")

            def correct_arguments(messages):
                failure = next(message for message in reversed(messages) if isinstance(message, ToolMessage))
                self.assertEqual(failure.status, "error")
                self.assertIn("invalid arguments", failure.content)
                self.assertIn("path", failure.content)
                return tool_call("read_safe_file", {"path": "notes.txt"}, "corrected-read")

            model = ScriptedChatModel(script=[
                tool_call("read_safe_file", {"start_line": 1}, "invalid-read"),
                correct_arguments,
                AIMessage(content="notes.txt contains local evidence."),
            ])
            token = set_active_workspace(root)
            try:
                result = await runner_for(agent_for(model, [read_safe_file])).run("Read notes.txt")
            finally:
                reset_active_workspace(token)
            activity = model_tool_activity(result.messages)
            self.assertEqual([(item["id"], item["ok"]) for item in activity], [("invalid-read", False), ("corrected-read", True)])
            self.assertIn("local evidence", result.response)

    async def test_edit_checkpoint_is_saved_before_the_next_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = parse_request(root, "Create answer.py with answer = 42", session_id="offline-workflow")
            save_plan(root, plan)
            observations = []

            def checkpoint(messages):
                activity = model_tool_activity(messages)
                plan.record_progress(activity)
                save_plan(root, plan)
                observations.extend(item["id"] for item in activity if item["ok"])

            def fail_after_observation(messages):
                saved = load_plan(root, plan.session_id, plan.plan_id)
                self.assertIn("answer.py", saved.edit_evidence.get("targets", {}), "Edit must be durable before asking the model to continue")
                self.assertIn("write-answer", observations)
                return RuntimeError("offline model unavailable after edit")

            model = ScriptedChatModel(script=[
                tool_call("write_safe_file", {"path": "answer.py", "content": "answer = 42\n"}, "write-answer"),
                fail_after_observation,
            ])
            agent = agent_for(model, [write_safe_file], interrupt_on={
                "write_safe_file": {"allowed_decisions": ["approve", "reject"]},
            })
            token = set_active_workspace(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "offline model unavailable after edit"):
                    await runner_for(agent, checkpoint=checkpoint).run(plan.request)
            finally:
                reset_active_workspace(token)
            self.assertEqual((root / "answer.py").read_text(encoding="utf-8"), "answer = 42\n")
            saved = load_plan(root, plan.session_id, plan.plan_id)
            self.assertEqual(saved.edit_evidence["targets"]["answer.py"]["id"], "write-answer")
            self.assertEqual(saved.status, "active")
            self.assertFalse(saved.edit_evidence["targets"]["answer.py"]["validations"])


if __name__ == "__main__":
    unittest.main()
