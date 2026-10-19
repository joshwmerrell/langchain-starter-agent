"""Retrieve retained thread details locally without extra model requests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from wallace_memory import (
    ConversationStore, bind_conversation, reset_conversation, search_conversation,
)


class MemoryRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_tracks_current_binding_and_labels_unverified_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ConversationStore(Path(directory) / "first.json")
            second = ConversationStore(Path(directory) / "second.json")
            first.append("user", "Choose crimson for the raven project.")
            first.append("assistant", "The raven logo was painted crimson.")
            second.append("user", "The raven project in this other thread uses blue.")
            token = bind_conversation(first)
            try:
                result = search_conversation.invoke({"query": "raven"})
                self.assertIn("crimson", result)
                self.assertIn("Wallace (unverified)", result)
                self.assertNotIn("blue", result)
                nested = bind_conversation(second)
                try:
                    switched = search_conversation.invoke({"query": "raven"})
                    self.assertIn("blue", switched)
                    self.assertNotIn("crimson", switched)
                finally:
                    reset_conversation(nested)
                self.assertIn("crimson", search_conversation.invoke({"query": "raven"}))
            finally:
                reset_conversation(token)
            self.assertIn("no conversation is bound", search_conversation.invoke({"query": "raven"}))

    async def test_large_backlog_makes_one_compaction_call_and_keeps_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "large-thread.json")
            history = [
                {"role": "user", "content": f"Decision {index}: retain project detail {index}. " + "Additional context. " * 15}
                for index in range(240)
            ]
            current = "What decisions did I make?"
            history.append({"role": "user", "content": current})
            store.save(history)
            summarize = AsyncMock(return_value="Earlier project decisions remain available in the saved thread.")
            context, warning = await store.memory_context(current, summarize, max_chars=1_000)
            summarize.assert_awaited_once()
            self.assertLessEqual(len(summarize.await_args.args[1]), 40)
            self.assertNotIn(current, [message["content"] for message in summarize.await_args.args[1]])
            self.assertIsNone(warning)
            self.assertLessEqual(len(context), 1_000)
            self.assertIn("Decision 239", context)
            self.assertEqual(ConversationStore(store.path).load_all(), history)
            token = bind_conversation(store)
            try:
                # A detail outside the bounded prompt is still locally searchable.
                found = search_conversation.invoke({"query": "Decision 9:", "limit": 1})
            finally:
                reset_conversation(token)
            self.assertIn("Decision 9:", found)
            summarize.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
