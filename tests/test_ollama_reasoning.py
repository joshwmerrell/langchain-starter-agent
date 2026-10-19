"""Thinking models must keep private reasoning separate from final text."""

import os
import unittest
from unittest.mock import Mock, patch

import httpx

from main import _chat_model
from wallace_models import _ollama_thinking_supported, ollama_reasoning


class OllamaReasoningTests(unittest.TestCase):
    def setUp(self):
        _ollama_thinking_supported.cache_clear()
        self.addCleanup(_ollama_thinking_supported.cache_clear)

    def test_auto_detects_supported_thinking_and_caches_metadata(self):
        response = Mock()
        response.json.return_value = {"capabilities": ["completion", "tools", "thinking"]}
        with patch.dict(os.environ, {"WALLACE_REASONING": "auto"}):
            with patch("wallace_models.httpx.post", return_value=response) as post:
                self.assertIs(ollama_reasoning("http://localhost:11434", "local:test"), True)
                self.assertIs(ollama_reasoning("http://localhost:11434", "local:test"), True)
                post.assert_called_once()

    def test_unavailable_or_non_thinking_model_keeps_provider_default(self):
        with patch.dict(os.environ, {"WALLACE_REASONING": "auto"}):
            with patch("wallace_models.httpx.post", side_effect=httpx.ConnectError("offline")):
                self.assertIsNone(ollama_reasoning("http://localhost:11434", "offline:test"))
            response = Mock()
            response.json.return_value = {"capabilities": ["completion", "tools"]}
            with patch("wallace_models.httpx.post", return_value=response):
                self.assertIsNone(ollama_reasoning("http://localhost:11434", "plain:test"))

    def test_explicit_choices_do_not_probe_the_server(self):
        with patch("wallace_models.httpx.post") as post:
            for setting, expected in (("true", True), ("false", False), ("default", None)):
                with patch.dict(os.environ, {"WALLACE_REASONING": setting}):
                    self.assertIs(ollama_reasoning("http://localhost:11434", "local:test"), expected)
            post.assert_not_called()

    def test_summary_uses_same_model_without_spending_its_budget_on_thinking(self):
        with patch("main.ollama_reasoning", return_value=True):
            with patch("main.ChatOllama") as model:
                _chat_model("local:test")
                self.assertIs(model.call_args.kwargs["reasoning"], True)
                _chat_model("local:test", summary_chars=600)
                self.assertIs(model.call_args.kwargs["reasoning"], False)
                self.assertEqual(model.call_args.kwargs["model"], "local:test")
