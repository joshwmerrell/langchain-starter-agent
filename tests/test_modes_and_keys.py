"""Normal/development boundaries and host-only cloud key management."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from textual.widgets import Input

from tools import (
    _run_command, current_workspace_root, execute_command, execute_command_async,
    execute_direct_command, get_git_diff,
    get_git_status, get_project_tree, read_safe_file,
    read_safe_file, replace_in_file, reset_active_workspace, run_tests,
    set_active_workspace, verify_tool_contract, write_safe_file,
)
from main import direct_command_text, run_agent, setup_agent
from tui_agent import AgentTUI, CloudKeyScreen
from wallace_keys import active_key, add_key, list_keys, remove_key, set_key_active
from wallace_mcp import (
    MCPServer, add_mcp_server, all_mcp_servers, apply_mcp_command, mcp_target,
    parse_mcp_command, read_mcp_configuration, save_mcp_configuration,
)
from wallace_planning.threads import ConversationThreadManager, DEV_THREAD_ID
from wallace_runtime import plan_session_id, thread_workspace


class FakeKeyring:
    def __init__(self):
        self.values = {}

    def set_password(self, service, name, secret):
        self.values[service, name] = secret

    def get_password(self, service, name):
        return self.values.get((service, name))

    def delete_password(self, service, name):
        del self.values[service, name]


class ModeAndKeyTests(unittest.TestCase):
    def test_direct_command_parser_preserves_shell_quoting(self):
        self.assertEqual(direct_command_text('/run printf "%s" "a b"'), 'printf "%s" "a b"')
        self.assertEqual(direct_command_text('/RUN git status'), 'git status')
        self.assertIsNone(direct_command_text('/runner git status'))
        with self.assertRaisesRegex(ValueError, "use /run"):
            direct_command_text('/run')

    def test_stdio_mcp_configuration_uses_host_env_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_mcp_configuration(root, {
                "langchain_docs_enabled": True,
                "servers": [{
                    "name": "github-local",
                    "transport": "stdio",
                    "command": "docker",
                    "args": ["run", "-i", "--rm"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
                    "enabled": True,
                    "keywords": ["github"],
                }],
            })
            with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}, clear=False):
                server = read_mcp_configuration(root)["servers"][0]
                target = mcp_target(
                    MCPServer(
                        server["name"], "", tuple(server["keywords"]), True, False,
                        "stdio", server["command"], tuple(server["args"]),
                        tuple(server["env"].items()), None,
                    ),
                    root,
                )
            self.assertEqual(target["mcpServers"]["github-local"]["command"], "docker")
            self.assertEqual(
                target["mcpServers"]["github-local"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"],
                "test-token",
            )
            self.assertEqual(
                parse_mcp_command(["add", "github", "--stdio", "docker", "run", "-i"], root),
                ("add_stdio", "github", "docker", ("run", "-i")),
            )

    def test_stdio_mcp_configuration_rejects_literal_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_mcp_configuration(Path(directory), {
                    "servers": [{
                        "name": "unsafe",
                        "transport": "stdio",
                        "command": "server",
                        "env": {"TOKEN": "literal-secret"},
                    }],
                })

    def test_http_mcp_auth_uses_only_a_host_environment_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_mcp_server(root, "github", "https://api.githubcopilot.com/mcp/")
            self.assertEqual(
                parse_mcp_command(["auth", "github", "GITHUB_MCP_TOKEN"], root),
                ("auth", "github", "GITHUB_MCP_TOKEN", ()),
            )
            self.assertTrue(apply_mcp_command(root, "auth", "github", "GITHUB_MCP_TOKEN"))
            self.assertFalse(apply_mcp_command(root, "auth", "github", "GITHUB_MCP_TOKEN"))
            self.assertEqual(read_mcp_configuration(root)["servers"][0]["auth_env"], "GITHUB_MCP_TOKEN")
            server = all_mcp_servers(root)[1]
            with patch.dict(os.environ, {"GITHUB_MCP_TOKEN": "test-secret"}):
                target = mcp_target(server, root)
            self.assertEqual(
                target["mcpServers"]["github"]["headers"]["Authorization"],
                "Bearer test-secret",
            )
            from fastmcp.mcp_config import MCPConfig, RemoteMCPServer
            self.assertIsInstance(MCPConfig.model_validate(target).mcpServers["github"], RemoteMCPServer)
            self.assertNotIn("test-secret", (root / ".wallace" / "mcp_servers.json").read_text())
            with patch.dict(os.environ, {"GITHUB_MCP_TOKEN": ""}):
                with self.assertRaisesRegex(ValueError, "requires host environment variable"):
                    mcp_target(server, root)
            with patch.dict(os.environ, {"GITHUB_MCP_TOKEN": "bad\nheader"}):
                with self.assertRaisesRegex(ValueError, "invalid control character"):
                    mcp_target(server, root)
            with self.assertRaisesRegex(ValueError, "auth_env must name"):
                apply_mcp_command(root, "auth", "github", "literal-secret-value!")

    def test_workspace_files_do_not_require_git_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("Workspace instructions\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root), patch("tools.DEV_MODE", True):
                token = set_active_workspace(root)
                try:
                    tree = get_project_tree.invoke({})
                    readme = read_safe_file.invoke({"path": "README.md"})
                    missing = read_safe_file.invoke({"path": "roadmap.md"})
                    denied = read_safe_file.invoke({"path": ".aws/credentials"})
                    status = get_git_status.invoke({})
                finally:
                    reset_active_workspace(token)
            self.assertIn("README.md", tree)
            self.assertIn("Workspace instructions", readme)
            self.assertEqual(missing, "Error: file not found: roadmap.md")
            self.assertIn("Access denied", denied)
            self.assertIn("Git metadata is not available", status)

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_git_status_is_repository_wide_and_lists_untracked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
            (root / "tracked.py").write_text("before\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=Test", "-c",
                "user.email=test@example.com", "commit", "-q", "-m", "initial",
            ], check=True)
            (root / "tracked.py").write_text("after\n")
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("new workspace file\n")
            with patch("tools.APP_ROOT", root), patch("tools.SAFE_ROOT", workspace), patch("tools.DEV_MODE", False):
                token = set_active_workspace(workspace)
                try:
                    status = get_git_status.invoke({})
                finally:
                    reset_active_workspace(token)
            self.assertIn("tracked.py", status)
            self.assertIn("workspace/README.md", status)

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_user_directed_git_status_reads_config_but_model_command_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
            (root / "note.txt").write_text("changed\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root), patch("tools.DEV_MODE", True):
                token = set_active_workspace(root)
                try:
                    status = get_git_status.invoke({})
                    self.assertIn("##", status)
                    self.assertIn("note.txt", status)
                    command = _run_command("test -s .git/config", shell=True)
                    direct_status = execute_direct_command("git status --short --untracked-files=all")
                    direct_config = execute_direct_command("test -s .git/config")
                finally:
                    reset_active_workspace(token)
            self.assertIn("##", status)
            self.assertIn("note.txt", status)
            self.assertIn("Exit code: 1", command)
            self.assertIn("note.txt", direct_status)
            self.assertIn("Exit code: 0", direct_config)

    def test_regular_threads_are_shared_and_dev_thread_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal = ConversationThreadManager(root, scope="workspace")
            normal.ensure("main")
            legacy = ConversationThreadManager(root, scope="legacy")
            legacy.ensure("main")
            legacy.store("main").append("user", "Old development main")
            legacy.create("test_website", thread_id="thread_site")
            legacy.store("thread_site").append("user", "Inspect the site")
            dev = ConversationThreadManager(root, scope="development")
            dev.ensure(DEV_THREAD_ID)
            self.assertEqual(dev.active_thread_id(), DEV_THREAD_ID)
            self.assertIn("thread_site", {item.thread_id for item in normal.list_threads()})
            self.assertIn("legacy_main", {item.thread_id for item in normal.list_threads()})
            self.assertEqual(normal.store("thread_site").load_all()[0]["content"], "Inspect the site")
            self.assertEqual(normal.store("legacy_main").load_all()[0]["content"], "Old development main")
            self.assertEqual(legacy.store("thread_site").load_all()[0]["content"], "Inspect the site")
            dev.store("thread_site").append("assistant", "Site inspected")
            self.assertEqual(normal.store("thread_site").load_all()[-1]["content"], "Site inspected")
            created = normal.create("Shared new")
            self.assertEqual(dev.resolve(created.thread_id).title, "Shared new")
            dev.rename(created.thread_id, "Renamed elsewhere")
            self.assertEqual(normal.resolve(created.thread_id).title, "Renamed elsewhere")
            dev.remove(created.thread_id)
            with self.assertRaises(ValueError):
                normal.resolve(created.thread_id)
            with self.assertRaises(ValueError):
                normal.resolve(DEV_THREAD_ID)
            with self.assertRaises(ValueError):
                normal.ensure(DEV_THREAD_ID)
            with self.assertRaises(ValueError):
                dev.remove(DEV_THREAD_ID)
            with self.assertRaises(ValueError):
                dev.rename(DEV_THREAD_ID, "Disposable")
            self.assertEqual(dev.active_thread_id(), DEV_THREAD_ID)
            shared_index = json.loads(normal.index_path.read_text(encoding="utf-8"))
            shared_index["threads"].append(dev.resolve(DEV_THREAD_ID).to_dict())
            normal.index_path.write_text(json.dumps(shared_index), encoding="utf-8")
            reopened = ConversationThreadManager(root, scope="workspace")
            self.assertNotIn(DEV_THREAD_ID, {record.thread_id for record in reopened.list_threads()})
            with self.assertRaises(ValueError):
                reopened.resolve(DEV_THREAD_ID)
            self.assertNotIn(DEV_THREAD_ID, {
                record["thread_id"] for record in json.loads(reopened.index_path.read_text())["threads"]
            })
            self.assertEqual(dev.resolve(DEV_THREAD_ID).thread_id, DEV_THREAD_ID)

    def test_concurrent_windows_preserve_appended_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ConversationThreadManager(root, scope="workspace")
            first.ensure("main")
            second = ConversationThreadManager(root, scope="development")
            second.ensure(DEV_THREAD_ID)
            stores = [first.store("main"), second.store("main")]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(stores[index % 2].append, "user", f"turn {index}")
                           for index in range(20)]
                for future in futures:
                    future.result()
            self.assertEqual(len(first.store("main").load_all()), 20)

    def test_workspace_is_selected_by_thread_not_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "workspace"
            work.mkdir()
            (root / "main.py").write_text("source fixture")
            (work / "note.txt").write_text("workspace fixture")
            self.assertEqual(thread_workspace("thread_site", project_root=root, development=True), work)
            self.assertEqual(thread_workspace(DEV_THREAD_ID, project_root=root, development=True), root)
            self.assertEqual(thread_workspace(DEV_THREAD_ID, project_root=root, development=False), work)
            self.assertEqual(plan_session_id("thread_site", development=True), "workspace.thread_site")
            with patch("tools.APP_ROOT", root), patch("tools.DEV_MODE", True):
                token = set_active_workspace(work)
                try:
                    self.assertEqual(current_workspace_root(), work)
                    self.assertIn("Access denied", read_safe_file.invoke({"path": "../main.py"}))
                    self.assertIn("workspace fixture", read_safe_file.invoke({"path": "note.txt"}))
                finally:
                    reset_active_workspace(token)
                token = set_active_workspace(root)
                try:
                    self.assertIn("source fixture", read_safe_file.invoke({"path": "main.py"}))
                finally:
                    reset_active_workspace(token)

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_development_launcher_regular_thread_keeps_edit_shell_and_test_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "workspace"
            work.mkdir()
            (work / "tests").mkdir()
            (work / "tests" / "test_sample.py").write_text(
                "import unittest\n\nclass SampleTests(unittest.TestCase):\n"
                "    def test_example(self):\n        self.assertEqual(2 + 2, 4)\n"
            )
            with patch("tools.DEV_MODE", True):
                token = set_active_workspace(work)
                try:
                    self.assertIn("Created", write_safe_file.invoke({"path": "note.txt", "content": "first"}))
                    self.assertIn("Edited", replace_in_file.invoke({
                        "path": "note.txt", "old_text": "first", "new_text": "second",
                    }))
                    self.assertIn("second", read_safe_file.invoke({"path": "note.txt"}))
                    self.assertIn("second", execute_command.invoke({"command": "cat note.txt"}))
                    self.assertIn("Exit code: 0", run_tests.invoke({"test_path": "tests"}))
                finally:
                    reset_active_workspace(token)

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_workspace_tool_contract_without_source_registry_access(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "workspace"
            work.mkdir()
            (work / "tests").mkdir()
            (work / "demo_tool.py").write_text(
                "from langchain.tools import tool\n"
                "@tool\ndef double(value: int) -> int:\n"
                "    '''Double a value.'''\n    return value * 2\n"
            )
            (work / "tests" / "test_contract.py").write_text(
                "import unittest\nfrom demo_tool import double\n"
                "class ContractTests(unittest.TestCase):\n"
                "    def test_double(self):\n"
                "        self.assertEqual(double.invoke({'value': 2}), 4)\n"
            )
            token = set_active_workspace(work)
            try:
                result = verify_tool_contract.invoke({
                    "tool_name": "double", "test_path": "tests/test_contract.py",
                })
            finally:
                reset_active_workspace(token)
            self.assertIn("passing contract tests", result)

    def test_keyring_values_never_enter_metadata_and_toggle_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ring = FakeKeyring()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""}), patch("wallace_keys._ring", return_value=ring):
                add_key(root, "primary", "openai", "secret-for-test")
                self.assertEqual(active_key(root, "openai").get_secret_value(), "secret-for-test")
                metadata = (root / ".wallace" / "key_registry.json").read_text()
                self.assertNotIn("secret-for-test", metadata)
                self.assertEqual([entry.name for entry in list_keys(root) if entry.enabled and entry.source == "keyring"], ["primary"])
                add_key(root, "secondary", "openai", "another-test-secret")
                set_key_active(root, "primary", False)
                self.assertEqual(active_key(root, "openai").get_secret_value(), "another-test-secret")
                remove_key(root, "secondary")
                set_key_active(root, "primary", False)
                self.assertIsNone(active_key(root, "openai"))
                set_key_active(root, "primary", True)
                self.assertIsNotNone(active_key(root, "openai"))
                remove_key(root, "primary")
                self.assertIsNone(active_key(root, "openai"))
                with self.assertRaises(ValueError):
                    remove_key(root, "environment-openai")

    def test_environment_key_can_be_disabled_without_editing_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-environment-key"}):
                self.assertIsNotNone(active_key(root, "gemini"))
                set_key_active(root, "environment-gemini", False)
                self.assertIsNone(active_key(root, "gemini"))
                set_key_active(root, "environment-gemini", True)
                self.assertEqual(active_key(root, "gemini").get_secret_value(), "test-environment-key")
            self.assertNotIn("test-environment-key", (root / ".wallace" / "key_registry.json").read_text())

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_normal_command_cannot_read_source_or_key_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "workspace"
            work.mkdir()
            (root / "main.py").write_text("source fixture")
            (root / ".env").write_text("secret fixture")
            (work / "note.txt").write_text("allowed fixture")
            (work / "nested").mkdir()
            (work / "nested" / ".env").write_text("nested secret fixture")
            (work / ".aws").mkdir()
            (work / ".aws" / "credentials").write_text("private fixture")
            (work / ".git").mkdir()
            (work / ".git" / "config").write_text("git credential fixture")
            with patch("tools.SAFE_ROOT", work), patch("tools.APP_ROOT", root), patch("tools.DEV_MODE", False):
                self.assertIn("Access denied", read_safe_file.invoke({"path": "../main.py"}))
                self.assertIn("Access denied", read_safe_file.invoke({"path": ".aws/credentials"}))
                output = _run_command(
                    "test -e ../main.py; echo source:$?; test -e ../.env; echo env:$?; "
                    "test -s nested/.env; echo nested_env_has_content:$?; "
                    "test -e .aws/credentials; echo aws_visible:$?; "
                    "test -s .git/config; echo git_config_has_content:$?; cat note.txt",
                    shell=True,
                )
                self.assertIn("source:1", output)
                self.assertIn("env:1", output)
                self.assertIn("nested_env_has_content:1", output)
                self.assertIn("aws_visible:1", output)
                self.assertIn("git_config_has_content:1", output)
                self.assertIn("allowed fixture", output)

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_dev_command_sees_source_but_not_private_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("source fixture")
            (root / ".env").write_text("secret fixture")
            (root / ".aws").mkdir()
            (root / ".aws" / "credentials").write_text("private fixture")
            private = root / ".wallace"
            private.mkdir()
            (private / "key_registry.json").write_text(json.dumps({"secret": "secret fixture"}))
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root), patch("tools.DEV_MODE", True):
                output = _run_command(
                    "cat main.py; test -s .env; echo env_has_content:$?; "
                    "test -e .wallace/key_registry.json; echo registry_visible:$?; "
                    "test -e .aws/credentials; echo aws_visible:$?",
                    shell=True,
                )
                self.assertIn("source fixture", output)
                self.assertIn("env_has_content:1", output)
                self.assertIn("registry_visible:1", output)
                self.assertIn("aws_visible:1", output)
                self.assertNotIn("secret fixture", output)

    def test_missing_sandbox_fails_closed(self):
        with patch("tools.shutil.which", return_value=None), patch("tools.subprocess.run") as run:
            output = _run_command("echo not-run", shell=True)
        self.assertIn("Bubblewrap", output)
        run.assert_not_called()

    def test_git_diff_excludes_private_directories(self):
        with patch("tools._git_available", return_value=True), patch("tools._run_command", return_value="Exit code: 0") as run:
            get_git_diff.invoke({"staged": False})
        arguments = run.call_args.args[0]
        self.assertIn(":(glob,exclude)**/.aws/**", arguments)
        self.assertIn(":(glob,exclude)**/.wallace/**", arguments)


class TUIKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_masked_add_toggle_and_remove_do_not_enter_chat_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ring = FakeKeyring()
            with patch("tui_agent.SAFE_ROOT", root), patch("tui_agent.APP_ROOT", root), patch("wallace_keys._ring", return_value=ring):
                with patch.object(AgentTUI, "initialize_agent", lambda self: None):
                    app = AgentTUI()
                    async with app.run_test() as pilot:
                        app._handle_thread_command("/key add primary openai")
                        for _ in range(10):
                            await pilot.pause()
                            if isinstance(app.screen, CloudKeyScreen):
                                break
                        self.assertIsInstance(app.screen, CloudKeyScreen)
                        field = app.screen.query_one("#cloud_key_input", Input)
                        self.assertTrue(field.password)
                        field.value = "secret-for-test"
                        await pilot.press("enter")
                        await pilot.pause()
                        self.assertTrue(any(item.name == "primary" and item.enabled for item in list_keys(root)))
                        app._handle_thread_command("/key off primary")
                        await pilot.pause()
                        self.assertIsNone(active_key(root, "openai"))
                        app._handle_thread_command("/key on primary")
                        await pilot.pause()
                        app._handle_thread_command("/key remove primary")
                        await pilot.pause()
                        self.assertFalse(any(item.name == "primary" for item in list_keys(root)))
                        self.assertEqual(app.conversation_store.load_all(), [])
                        visible = " ".join(str(item[0]) for item in app.chat_log._theme_entries)
                        self.assertNotIn("secret-for-test", visible)


class CLIKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_command_uses_sandboxed_argv_and_reports_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = set_active_workspace(root)
            try:
                with patch("tools._sandbox_command", return_value=[
                    "/bin/sh", "-c", "printf '%s' 'a b'",
                ]) as sandbox:
                    result = await execute_command_async("printf '%s' 'a b'")
                sandbox.assert_called_once_with(
                    "printf '%s' 'a b'", shell=True, cwd=root, expose_git_config=False,
                )
                self.assertIn("Exit code: 0", result)
                self.assertIn("STDOUT:\na b", result)
                with patch("tools._sandbox_command", return_value=[
                    "/bin/sh", "-c", "printf failed >&2; exit 7",
                ]):
                    failure = await execute_command_async("printf failed >&2; exit 7")
                self.assertIn("Exit code: 7", failure)
                self.assertIn("STDERR:\nfailed", failure)
                with patch("tools._sandbox_command", return_value=[
                    "/bin/sh", "-c", "printf direct",
                ]) as sandbox:
                    direct = await execute_command_async("printf direct", user_directed=True)
                sandbox.assert_called_once_with(
                    "printf direct", shell=True, cwd=root, expose_git_config=True,
                )
                self.assertIn("STDOUT:\ndirect", direct)
            finally:
                reset_active_workspace(token)

    async def test_direct_cli_command_runs_without_model_turn_or_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))) as setup:
                    with patch("builtins.input", side_effect=[
                        '/run pwd', '/trust informed-trust 2', '/run pwd',
                        '/trust yes-trust', '/run printf "%s" "a b"', 'exit',
                    ]):
                        with patch("main._ask_cli_approval", side_effect=AssertionError("unexpected approval")) as approval:
                            with patch("main.execute_direct_command", return_value="Exit code: 0\nSTDOUT:\na b") as command_tool, patch("main.console"):
                                await run_agent()
            setup.assert_awaited_once()
            approval.assert_not_called()
            self.assertEqual(
                [call.args[0] for call in command_tool.call_args_list],
                ["pwd", "pwd", 'printf "%s" "a b"'],
            )

    async def test_normal_agent_loads_user_mcp_tools(self):
        with patch("main.DEV_MODE", False), patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
            with patch("main._load_configured_mcp_tools", new=AsyncMock(return_value=[])) as configured:
                with patch("main.ChatOllama"), patch("main.create_agent", return_value=object()):
                    await setup_agent(model_name="local:test")
            configured.assert_awaited_once()

    async def test_cli_add_toggle_and_remove_key_without_chat_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ring = FakeKeyring()
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root), patch("wallace_keys._ring", return_value=ring):
                with patch("main.setup_agent", new=AsyncMock(return_value=(object(), {}))):
                    with patch("main.sys.stdin", SimpleNamespace(isatty=lambda: True)), patch("main.getpass.getpass", return_value="secret-for-test"):
                        with patch("builtins.input", side_effect=[
                            "/key add primary openai", "/key off primary",
                            "/key on primary", "/key remove primary", "exit",
                        ]), patch("main.console"):
                            await run_agent()
            self.assertFalse(any(item.name == "primary" for item in list_keys(root)))
            self.assertNotIn("secret-for-test", (root / ".wallace" / "key_registry.json").read_text())
