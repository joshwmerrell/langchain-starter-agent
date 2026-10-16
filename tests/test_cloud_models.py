"""Cloud credentials stay out of model context, saved state, and chat input."""

import json
import os
import asyncio
import tempfile
import unittest
import httpx
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import SecretStr
from textual.widgets import Input

from main import _ask_cli_approval, _chat_model, run_agent, setup_agent, summarize_thread_memory
from tools import _run_command
from tui_agent import AgentTUI, ApprovalScreen, CloudKeyScreen
from wallace_models import (
    CloudCredentials,
    MissingCredentialError,
    environment_key,
    list_cloud_models,
    load_model_choice,
    parse_model_selection,
    save_model_choice,
    use_environment_key,
)


class CloudModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        configured_keys = patch.dict(os.environ, {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""})
        configured_keys.start()
        self.addCleanup(configured_keys.stop)

    async def test_configured_keys_use_exact_variable_names(self):
        credentials = CloudCredentials()
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-configured",
            "GOOGLE_API_KEY": "AIzaTestConfigured",
        }):
            self.assertTrue(use_environment_key(credentials, "openai"))
            self.assertEqual(credentials.get("openai").get_secret_value(), "sk-test-configured")
            self.assertEqual(environment_key("gemini").get_secret_value(), "AIzaTestConfigured")
        self.assertIsNone(credentials.get("gemini"))

    async def test_cloud_model_lists_authenticate_in_headers_and_filter_gemini(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertNotIn("test-list-key", str(request.url))
            if request.url.host == "api.openai.com":
                self.assertEqual(request.headers["Authorization"], "Bearer test-list-key")
                return httpx.Response(200, json={"data": [{"id": "gpt-example"}]})
            self.assertEqual(request.headers["x-goog-api-key"], "test-list-key")
            if "pageToken" not in request.url.params:
                return httpx.Response(200, json={
                    "models": [
                        {"name": "models/gemini-example", "supportedGenerationMethods": ["generateContent"]},
                        {"name": "models/text-embedding-example", "supportedGenerationMethods": ["embedContent"]},
                    ],
                    "nextPageToken": "next-page",
                })
            self.assertEqual(request.url.params["pageToken"], "next-page")
            return httpx.Response(200, json={
                "models": [{"name": "models/gemini-example-2", "supportedGenerationMethods": ["generateContent"]}],
            })

        secret = SecretStr("test-list-key")
        for provider, expected in (
            ("openai", ["gpt-example"]),
            ("gemini", ["gemini-example", "gemini-example-2"]),
        ):
            client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            with patch("wallace_models.httpx.AsyncClient", return_value=client):
                self.assertEqual(await list_cloud_models(provider, secret), expected)
        self.assertEqual(len(requests), 3)

    async def test_cloud_model_selection_rejects_key_like_names(self):
        self.assertEqual(parse_model_selection("openai:gpt-test"), ("openai", "gpt-test"))
        self.assertEqual(parse_model_selection("gemini:gemini-test"), ("gemini", "gemini-test"))
        for value in ("openai:", "openai:sk-proj-example", "gemini:AIzaExample"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_model_selection(value)

    async def test_provider_client_gets_secret_but_model_context_does_not(self):
        credentials = CloudCredentials()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-environment-secret"}):
            with self.assertRaises(MissingCredentialError):
                _chat_model("openai:gpt-test")
        credentials.set("openai", "sk-test-cloud-secret")
        with patch("langchain_openai.ChatOpenAI") as model_type:
            model = _chat_model("openai:gpt-test", credentials)
            self.assertIs(model, model_type.return_value)
            kwargs = model_type.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-test")
            self.assertEqual(kwargs["base_url"], "https://api.openai.com/v1")
            self.assertIsInstance(kwargs["api_key"], SecretStr)
            self.assertEqual(kwargs["api_key"].get_secret_value(), "sk-test-cloud-secret")
        with patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
            with patch("langchain_openai.ChatOpenAI"):
                with patch("main.create_agent", return_value=object()) as create:
                    await setup_agent(model_name="openai:gpt-test", credentials=credentials)
                    self.assertNotIn("sk-test-cloud-secret", create.call_args.kwargs["system_prompt"])
                    self.assertNotIn("sk-test-cloud-secret", repr(create.call_args.kwargs["tools"]))
        with patch("langchain_openai.ChatOpenAI") as model_type:
            model_type.return_value.ainvoke = AsyncMock(
                return_value=SimpleNamespace(content="A concise memory summary.")
            )
            await summarize_thread_memory(
                "", [{"role": "user", "content": "Remember the project"}], 200,
                model_name="openai:gpt-test", credentials=credentials,
            )
            self.assertNotIn(
                "sk-test-cloud-secret",
                repr(model_type.return_value.ainvoke.await_args.args[0]),
            )
        credentials.set("gemini", "AIzaExampleSecret")
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as model_type:
            _chat_model("gemini:gemini-test", credentials)
            self.assertEqual(model_type.call_args.kwargs["model"], "gemini-test")
            self.assertFalse(model_type.call_args.kwargs["vertexai"])
            self.assertIsInstance(model_type.call_args.kwargs["api_key"], SecretStr)

    async def test_model_choice_is_saved_without_runtime_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = CloudCredentials()
            credentials.set("openai", "sk-test-cloud-secret")
            save_model_choice(root, "openai:gpt-test")
            self.assertEqual(load_model_choice(root, "local:latest"), "openai:gpt-test")
            saved = (root / ".wallace" / "model.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-cloud-secret", saved)
            self.assertEqual(json.loads(saved)["model"], "openai:gpt-test")
            self.assertNotIn("sk-test-cloud-secret", repr(credentials))
            self.assertEqual(
                credentials.redact("error: sk-test-cloud-secret"),
                "error: [REDACTED API KEY]",
            )

    async def test_tui_cloud_key_uses_masked_dialog_and_not_chat_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    with patch("tui_agent.setup_agent", new=AsyncMock(return_value=(object(), {}))):
                        async with app.run_test() as pilot:
                            worker = app.change_model("openai:gpt-test")
                            for _ in range(10):
                                await pilot.pause()
                                if isinstance(app.screen, CloudKeyScreen):
                                    break
                            self.assertIsInstance(app.screen, CloudKeyScreen)
                            field = app.screen.query_one("#cloud_key_input", Input)
                            self.assertTrue(field.password)
                            field.value = "sk-test-cloud-secret"
                            await pilot.press("enter")
                            await worker.wait()
                            self.assertEqual(app.model_name, "openai:gpt-test")
                            self.assertEqual(
                                app.credentials.get("openai").get_secret_value(),
                                "sk-test-cloud-secret",
                            )
                            self.assertEqual(app.conversation_store.load_all(), [])
                            visible = " ".join(str(item[0]) for item in app.chat_log._theme_entries)
                            self.assertNotIn("sk-test-cloud-secret", visible)
                            with patch("tui_agent.list_installed_models", new=AsyncMock(return_value=["local:latest"])):
                                await app.change_model("local:latest").wait()
                            self.assertIsNone(app.credentials.get("openai"))
                    self.assertNotIn(
                        "sk-test-cloud-secret",
                        (root / ".wallace" / "model.json").read_text(encoding="utf-8"),
                    )

    async def test_saved_cloud_choice_reprompts_after_tui_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model_choice(root, "gemini:gemini-test")
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                app = AgentTUI()
                async with app.run_test() as pilot:
                    for _ in range(10):
                        await pilot.pause()
                        if isinstance(app.screen, CloudKeyScreen):
                            break
                    self.assertIsInstance(app.screen, CloudKeyScreen)
                    self.assertIsNone(app.credentials.get("gemini"))
                    await pilot.press("escape")
                    await pilot.pause()
                    self.assertIsNone(app.agent)
                    self.assertFalse(app.prompt_input.disabled)

    async def test_saved_cloud_choice_uses_configured_key_without_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model_choice(root, "gemini:gemini-test")
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzaTestConfigured"}):
                with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root):
                    with patch("tui_agent.setup_agent", new=AsyncMock(return_value=(object(), {}))) as setup:
                        app = AgentTUI()
                        async with app.run_test() as pilot:
                            for _ in range(10):
                                await pilot.pause()
                                if app.agent is not None:
                                    break
                            self.assertIsNotNone(app.agent)
                            self.assertEqual(app.credentials.get("gemini").get_secret_value(), "AIzaTestConfigured")
                            self.assertFalse(isinstance(app.screen, CloudKeyScreen))
                        self.assertEqual(setup.await_args.kwargs["model_name"], "gemini:gemini-test")

    async def test_tui_cloud_approval_displays_residual_risk_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tui_agent.SAFE_ROOT", Path(directory)), patch("tui_agent.APP_ROOT", Path(directory)):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app.model_name = "openai:gpt-test"
                        approval = asyncio.create_task(app._ask_approval({
                            "name": "execute_command", "args": {"command": "pwd"},
                        }))
                        await pilot.pause()
                        self.assertIsInstance(app.screen, ApprovalScreen)
                        self.assertIn("Cloud mode warning", app.screen.detail)
                        await pilot.press("n")
                        self.assertFalse(await approval)

    async def test_cli_cloud_switch_uses_hidden_key_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))) as setup:
                    with patch("main.sys.stdin", SimpleNamespace(isatty=lambda: True)):
                        with patch("main.getpass.getpass", return_value="sk-test-cloud-secret") as hidden:
                            with patch("builtins.input", side_effect=["/model openai:gpt-test", "exit"]):
                                with patch("main.console"):
                                    await run_agent()
                self.assertEqual(setup.await_count, 2)
                self.assertEqual(setup.await_args.kwargs["model_name"], "openai:gpt-test")
                hidden.assert_called_once()
                self.assertEqual(load_model_choice(root, "local:latest"), "openai:gpt-test")
                for path in (root / ".wallace").rglob("*.json"):
                    self.assertNotIn("sk-test-cloud-secret", path.read_text(encoding="utf-8"))

    async def test_cli_cloud_switch_uses_configured_key_without_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-configured"}):
                with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                    with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))) as setup:
                        with patch("main.getpass.getpass", side_effect=AssertionError("unexpected key prompt")):
                            with patch("builtins.input", side_effect=["/model openai:gpt-test", "exit"]):
                                with patch("main.console"):
                                    await run_agent()
                self.assertEqual(setup.await_args.kwargs["model_name"], "openai:gpt-test")
                self.assertEqual(
                    setup.await_args.kwargs["credentials"].get("openai").get_secret_value(),
                    "sk-test-configured",
                )

    async def test_cloud_approval_warns_and_redacts(self):
        credentials = CloudCredentials()
        credentials.set("openai", "sk-test-cloud-secret")
        request = {"name": "execute_command", "args": {"command": "echo sk-test-cloud-secret"}}
        with patch("builtins.input", return_value="n"):
            with patch("main.console") as console:
                self.assertFalse(_ask_cli_approval(request, "openai:gpt-test", credentials))
        displayed = console.print.call_args.args[0].renderable.plain
        self.assertIn("Cloud mode warning", displayed)
        self.assertNotIn("sk-test-cloud-secret", displayed)

    async def test_tool_subprocess_drops_common_credential_environment(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test-cloud-secret",
            "GOOGLE_API_KEY": "AIzaExampleSecret",
            "WALLACE_TEST_TOKEN": "test-token",
            "WALLACE_SAFE_MARKER": "okay",
        }):
            with patch("tools.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout="okay", stderr="",
            )) as run:
                _run_command("echo okay", shell=True)
                child_env = run.call_args.kwargs["env"]
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("GOOGLE_API_KEY", child_env)
        self.assertNotIn("WALLACE_TEST_TOKEN", child_env)
        self.assertEqual(child_env["WALLACE_SAFE_MARKER"], "okay")


if __name__ == "__main__":
    unittest.main()
