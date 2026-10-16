"""Conversation memory must survive reloads without bloating model prompts."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from main import (
    ConversationStore,
    MAX_HISTORY_MESSAGES,
    MAX_MEMORY_CHARS,
    local_datetime_response,
    memory_budget_for,
    run_agent,
    summarize_thread_memory,
    with_thread_memory,
)
from tui_agent import AgentTUI, WorkspacePanel
from wallace_planning import ConversationThreadManager


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_history_is_migrated_without_trimming_raw_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thread.json"
            legacy = [
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
                for index in range(60)
            ]
            path.write_text(json.dumps(legacy), encoding="utf-8")
            store = ConversationStore(path)
            self.assertEqual(store.load_all(), legacy)
            self.assertEqual(len(store.load()), MAX_HISTORY_MESSAGES)

            store.append("user", "next question")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["messages"][:-1], legacy)
            self.assertEqual(store.load_all()[-1]["content"], "next question")

    async def test_recent_context_is_thread_specific_and_excludes_current_request(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ConversationThreadManager(Path(directory))
            first = manager.store("main")
            second = manager.store("other")
            first.append("user", "Remember the blue button")
            first.append("assistant", "I painted it blue.")
            first.append("user", "What did I ask?")
            second.append("user", "Private other-thread request")

            async def should_not_summarize(*args):
                self.fail("short history should not be summarized")

            context, warning = await first.memory_context("What did I ask?", should_not_summarize)
            self.assertIsNone(warning)
            self.assertIn("Remember the blue button", context)
            self.assertIn("Previous Wallace reply (unverified): I painted it blue.", context)
            self.assertNotIn("What did I ask?", context)
            self.assertNotIn("Private other-thread request", context)

    async def test_bad_saved_replies_are_preserved_but_not_replayed_as_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "thread.json")
            first = "I'm Wallace and I can help with programming and creative tasks. " * 12
            store.append("user", "Who are you?")
            store.append("assistant", first)
            store.append("user", "What is the time?")
            store.append("assistant", first + "I'll answer shortly.")
            store.append("user", "What is the date?")
            store.append("assistant", "Capabilities include " + "sorting ranking ordering filtering " * 100)
            store.append("user", "What did I ask about?")

            async def should_not_summarize(*args):
                self.fail("short history should not be summarized")

            context, warning = await store.memory_context("What did I ask about?", should_not_summarize)
            self.assertIsNone(warning)
            self.assertIn("Who are you?", context)
            self.assertIn("Unusable earlier Wallace reply omitted", context)
            self.assertNotIn("sorting ranking ordering filtering", context)
            self.assertEqual(len(store.load_all()), 7)

    async def test_local_date_and_time_come_from_host_clock(self):
        fixed = datetime(2026, 10, 12, 14, 35, 38, tzinfo=timezone(timedelta(hours=-6), "MDT"))
        self.assertEqual(
            local_datetime_response("What is today's date?", fixed),
            "Today is Monday, 2026-10-12 in the computer's MDT (UTC-0600) time zone.",
        )
        self.assertIn("14:35:38 MDT", local_datetime_response("What is the time and date?", fixed))
        self.assertTrue(local_datetime_response("Can you access the time and date?", fixed).startswith("Yes."))
        self.assertIsNone(local_datetime_response("What was the date in Tokyo?", fixed))
        self.assertIn("Host local clock at turn start:", with_thread_memory("", "How late is it here?"))

    async def test_versioned_legacy_history_migrates_to_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / ".wallace" / "conversation.json"
            legacy.parent.mkdir()
            legacy.write_text(json.dumps({
                "version": 2,
                "messages": [{"role": "user", "content": "legacy goal"}],
                "summary": "",
                "summarized_through": 0,
            }), encoding="utf-8")
            store = ConversationThreadManager(root).store("main")
            self.assertEqual(store.load_all()[0]["content"], "legacy goal")

    async def test_older_turns_are_summarized_and_raw_history_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thread.json"
            store = ConversationStore(path)
            for index in range(15):
                store.append("user", f"request {index}: retain the roadmap")
                store.append("assistant", f"reply {index}")
            store.append("user", "unique-current-request-42")
            batches = []

            async def summarize(previous, messages, limit):
                batches.append((previous, messages, limit))
                return (previous + " " + " ".join(
                    message["content"] for message in messages if message["role"] == "user"
                )).strip()

            context, warning = await store.memory_context("unique-current-request-42", summarize, max_chars=1_200)
            self.assertIsNone(warning)
            self.assertTrue(batches)
            self.assertLessEqual(len(context), 1_200)
            self.assertIn("Older thread summary:", context)
            self.assertIn("request 14: retain the roadmap", context)
            self.assertNotIn("unique-current-request-42", context)
            self.assertEqual(len(store.load_all()), 31)

            reloaded = ConversationStore(path)
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreater(state["summarized_through"], 0)
            self.assertTrue(state["summary"])
            again, _ = await reloaded.memory_context("unique-current-request-42", summarize, max_chars=1_200)
            self.assertEqual(context, again)
            self.assertEqual(len(reloaded.load_all()), 31)
            reloaded.append("assistant", "acknowledged the current request")
            reloaded.append("user", "later request")
            later, _ = await reloaded.memory_context("later request", summarize, max_chars=1_200)
            self.assertIn("Older thread summary:", later)
            self.assertIn("acknowledged the current request", later)
            self.assertEqual(len(reloaded.load_all()), 33)

    async def test_summary_failure_keeps_transcript_and_recent_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "thread.json")
            for index in range(15):
                store.append("user", f"request {index}")
                store.append("assistant", f"reply {index}")
            store.append("user", "new request")

            async def fail(*args):
                raise RuntimeError("model unavailable")

            context, warning = await store.memory_context("new request", fail, max_chars=800)
            self.assertIn("Could not summarize", warning)
            self.assertLessEqual(len(context), 800)
            self.assertIn("request 14", context)
            self.assertIn("temporarily unavailable", context)
            self.assertEqual(len(store.load_all()), 31)
            state = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(state["summarized_through"], 0)

    async def test_clear_removes_summary_and_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "thread.json")
            store.append("user", "old context")
            store.save_pending("unfinished")
            store.clear()
            self.assertEqual(store.load_all(), [])
            self.assertIsNone(store.load_pending())

    async def test_long_current_task_reduces_memory_budget(self):
        self.assertEqual(memory_budget_for("short"), MAX_MEMORY_CHARS)
        self.assertEqual(memory_budget_for("x" * 100_000), 400)

    async def test_oversized_summary_keeps_newest_information(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "thread.json")
            for index in range(11):
                store.append("user", f"request {index}")
                store.append("assistant", f"reply {index}")
            store.append("user", "new task")

            async def long_summary(previous, messages, limit):
                return "old fact " + "x" * (limit + 100) + " latest fact"

            await store.memory_context("new task", long_summary, max_chars=600)
            state = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertLessEqual(len(state["summary"]), MAX_MEMORY_CHARS * 2 // 5)
            self.assertIn("old fact", state["summary"])
            self.assertIn("latest fact", state["summary"])
            self.assertEqual(len(store.load_all()), 23)

    async def test_summary_request_uses_local_model_without_tools(self):
        with patch("main.ChatOllama") as model_type:
            model_type.return_value.ainvoke = AsyncMock(
                return_value=SimpleNamespace(content="User wants the blue button; Wallace claimed it was done.")
            )
            summary = await summarize_thread_memory(
                "User asked for a button.",
                [{"role": "assistant", "content": "I made the button blue."}],
                400,
                model_name="local:test",
            )
            self.assertIn("blue button", summary)
            self.assertEqual(model_type.return_value.ainvoke.await_count, 1)
            messages = model_type.return_value.ainvoke.await_args.args[0]
            self.assertIn("assistant claims", messages[0]["content"])
            self.assertIn("Previous Wallace reply (unverified)", messages[1]["content"])

    async def test_tui_passes_memory_to_model_and_saves_each_turn_once(self):
        class StreamAgent:
            def __init__(self):
                self.payloads = []

            async def astream(self, payload, config, stream_mode):
                self.payloads.append(payload)
                yield {"agent": {"messages": [AIMessage(content="You asked for blue.")]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": []})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        app.conversation_store.append("user", "Remember the blue button")
                        app.conversation_store.append("assistant", "I painted it blue.")
                        app.conversation_store.append("user", "What color?")
                        worker = app.run_turn("User request:\nWhat color?", user_prompt="What color?")
                        await worker.wait()
                        payload = app.agent.payloads[0]["messages"][-1]["content"]
                        self.assertIn("Remember the blue button", payload)
                        self.assertIn("Previous Wallace reply (unverified)", payload)
                        self.assertEqual(payload.count("What color?"), 1)
                        history = app.conversation_store.load_all()
                        self.assertEqual(len(history), 4)
                        self.assertEqual(history[-1]["content"], "You asked for blue.")

    async def test_tui_answers_local_date_without_asking_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app.agent = object()
                        app.prompt_input.disabled = False
                        app.prompt_input.focus()
                        app.prompt_input.value = "What is today's date?"
                        await pilot.press("enter")
                        await pilot.pause()
                        history = app.conversation_store.load_all()
                        self.assertEqual(len(history), 2)
                        self.assertEqual(history[0]["content"], "What is today's date?")
                        self.assertIn("Today is ", history[1]["content"])
                        self.assertIsNone(app.conversation_store.load_pending())
                        self.assertFalse(app._busy)
                    app.workspace_panel = WorkspacePanel()
                    app._refresh_workspace()  # A late timer tick after unmount is harmless.

    async def test_tui_retries_copied_basic_answer_without_memory(self):
        previous = "Wallace can help with " + " ".join(
            f"capability{index}" for index in range(90)
        )

        class StreamAgent:
            def __init__(self):
                self.payloads = []

            async def astream(self, payload, config, stream_mode):
                self.payloads.append(payload)
                answer = previous if len(self.payloads) == 1 else "2 + 2 = 4."
                yield {"agent": {"messages": [AIMessage(content=answer)]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": []})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        app.conversation_store.append("user", "Who are you?")
                        app.conversation_store.append("assistant", previous)
                        app.conversation_store.append("user", "What is 2+2?")
                        worker = app.run_turn("User request:\nWhat is 2+2?", user_prompt="What is 2+2?")
                        await worker.wait()
                        self.assertEqual(len(app.agent.payloads), 2)
                        self.assertIn("thread_memory", app.agent.payloads[0]["messages"][-1]["content"])
                        self.assertNotIn("thread_memory", app.agent.payloads[1]["messages"][-1]["content"])
                        self.assertEqual(app.conversation_store.load_all()[-1]["content"], "2 + 2 = 4.")

    async def test_tui_stops_after_one_failed_clean_retry(self):
        previous = "Wallace can help with " + " ".join(
            f"capability{index}" for index in range(90)
        )

        class StreamAgent:
            def __init__(self):
                self.payloads = []

            async def astream(self, payload, config, stream_mode):
                self.payloads.append(payload)
                yield {"agent": {"messages": [AIMessage(content=previous)]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": []})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        app.conversation_store.append("user", "Who are you?")
                        app.conversation_store.append("assistant", previous)
                        app.conversation_store.append("user", "What is 2+2?")
                        worker = app.run_turn("User request:\nWhat is 2+2?", user_prompt="What is 2+2?")
                        await worker.wait()
                        self.assertEqual(len(app.agent.payloads), 2)
                        self.assertEqual(len(app.conversation_store.load_all()), 3)
                        self.assertFalse(app._busy)

    async def test_cli_answers_date_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))):
                    with patch("builtins.input", side_effect=["What is today's date?", "exit"]):
                        with patch("main.console"):
                            await run_agent()
            history = ConversationThreadManager(root, scope="workspace").store("main").load_all()
            self.assertEqual(len(history), 2)
            self.assertIn("Today is ", history[1]["content"])

    async def test_cli_retries_copied_basic_answer_without_memory(self):
        previous = "Wallace can help with " + " ".join(
            f"capability{index}" for index in range(90)
        )

        class FakeAgent:
            def __init__(self):
                self.prompts = []

            async def ainvoke(self, payload, config):
                self.prompts.append(payload["messages"][-1]["content"])
                answer = previous if len(self.prompts) == 1 else "2 + 2 = 4."
                return {"messages": [AIMessage(content=answer)]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ConversationThreadManager(root, scope="workspace").store("main")
            store.append("user", "Who are you?")
            store.append("assistant", previous)
            agent = FakeAgent()
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(agent, {}))):
                    with patch("builtins.input", side_effect=["What is 2+2?", "exit"]):
                        with patch("main.console"):
                            await run_agent()
            self.assertEqual(len(agent.prompts), 2)
            self.assertIn("thread_memory", agent.prompts[0])
            self.assertNotIn("thread_memory", agent.prompts[1])
            self.assertEqual(store.load_all()[-1]["content"], "2 + 2 = 4.")

    async def test_tui_retry_without_checkpoint_keeps_thread_memory(self):
        class StreamAgent:
            def __init__(self):
                self.payloads = []

            async def astream(self, payload, config, stream_mode):
                self.payloads.append(payload)
                if len(self.payloads) == 1:
                    raise RuntimeError("invalid tool call arguments: unexpected end of JSON input")
                yield {"agent": {"messages": [AIMessage(content="The button was blue.")]}}

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": []})

        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test():
                        app.agent = StreamAgent()
                        app.thread_config = {}
                        app.conversation_store.append("user", "Remember the blue button")
                        app.conversation_store.append("assistant", "I painted it blue.")
                        app.conversation_store.append("user", "What color?")
                        worker = app.run_turn("User request:\nWhat color?", user_prompt="What color?")
                        await worker.wait()
                        self.assertEqual(len(app.agent.payloads), 2)
                        retry_prompt = app.agent.payloads[1]["messages"][0]["content"]
                        self.assertIn("Remember the blue button", retry_prompt)
                        self.assertIn("What color?", retry_prompt)


if __name__ == "__main__":
    unittest.main()
