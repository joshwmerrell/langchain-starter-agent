"""Tool authoring, non-Python validation, MCP policy, and task routing."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from main import (
    _load_configured_mcp_tools, approval_policy, focus_tools_on_request,
    format_action_request, model_tool_activity, run_agent, setup_agent,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tools import (
    READ_ONLY_TOOLS, TOOLS, TOOLS_REQUIRING_APPROVAL,
    check_source_file, list_mcp_connections, read_safe_file,
    run_language_tests, verify_tool_contract, write_safe_file,
)
from wallace_mcp import (
    add_mcp_server, all_mcp_servers, clear_connection_statuses,
    load_mcp_servers, remove_mcp_server, set_connection_status,
    set_mcp_server_active, tui_applied_current_config,
)
from wallace_tool_selection import select_tools


class ToolBuildingTests(unittest.TestCase):
    def test_only_explicit_read_only_tools_skip_approval(self):
        self.assertEqual(
            TOOLS_REQUIRING_APPROVAL,
            {tool.name for tool in TOOLS if tool.name not in READ_ONLY_TOOLS},
        )
        self.assertIn("verify_tool_contract", TOOLS_REQUIRING_APPROVAL)
        self.assertIn("edit_file_lines", TOOLS_REQUIRING_APPROVAL)

    def test_line_edit_approval_shows_current_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.py").write_text("one\ntwo\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root):
                prompt = format_action_request({
                    "name": "edit_file_lines",
                    "args": {"path": "example.py", "start_line": 2, "end_line": 2, "new_text": "three"},
                })
            self.assertIn("- two", prompt)
            self.assertIn("+ three", prompt)

    def test_default_test_target_is_recorded_for_plan(self):
        activity = model_tool_activity([
            AIMessage(content="", tool_calls=[{
                "id": "suite", "name": "run_tests", "args": {},
            }]),
            ToolMessage(content="Exit code: 0", tool_call_id="suite", name="run_tests"),
        ])
        self.assertEqual(activity[0]["target"], ".")

    def test_suffixless_syntax_success_is_recorded_with_actual_python_path(self):
        activity = model_tool_activity([
            AIMessage(content="", tool_calls=[{
                "id": "check", "name": "check_python_syntax", "args": {"path": "tests/example"},
            }]),
            ToolMessage(
                content="OK: tests/example.py has valid Python syntax.",
                tool_call_id="check", name="check_python_syntax",
            ),
        ])
        self.assertEqual(activity[0]["target"], "tests/example.py")

    def test_no_tests_diagnostic_is_saved_without_raw_output(self):
        activity = model_tool_activity([
            AIMessage(content="", tool_calls=[{
                "id": "suite", "name": "run_tests", "args": {},
            }]),
            ToolMessage(
                content="Exit code: 5\nSTDERR:\nRan 0 tests in 0.000s\nNO TESTS RAN",
                tool_call_id="suite", name="run_tests",
            ),
        ])
        self.assertFalse(activity[0]["ok"])
        self.assertIn("no tests were discovered", activity[0]["diagnostic"])

    def test_source_checker_uses_fixed_language_command_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.js").write_text("const value = ;\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", return_value="/usr/bin/node"):
                with patch("tools._run_command", return_value="Exit code: 1\nSTDERR:\nSyntaxError") as run:
                    result = check_source_file.invoke({"path": "example.js"})
            self.assertTrue(result.startswith("Exit code: 1"))
            self.assertEqual(run.call_args.args[0][:2], ["node", "--check"])
            activity = model_tool_activity([
                AIMessage(content="", tool_calls=[{
                    "id": "check", "name": "check_source_file", "args": {"path": "example.js"},
                }]),
                ToolMessage(content=result, tool_call_id="check", name="check_source_file"),
            ])
            self.assertFalse(activity[0]["ok"])

    def test_deno_type_error_code_survives_plan_summary_without_source_text(self):
        output = (
            "Exit code: 1\nSTDERR:\nTS2584 [ERROR]: Cannot find name 'SECRET_IDENTIFIER'.\n"
            "    at file:///project/app.ts:2:1"
        )
        activity = model_tool_activity([
            AIMessage(content="", tool_calls=[{
                "id": "check", "name": "check_source_file", "args": {"path": "app.ts"},
            }]),
            ToolMessage(content=output, tool_call_id="check", name="check_source_file"),
        ])
        self.assertFalse(activity[0]["ok"])
        self.assertIn("TS2584", activity[0]["diagnostic"])
        self.assertNotIn("SECRET_IDENTIFIER", str(activity))

    def test_failed_edit_records_safe_reason_without_file_contents(self):
        activity = model_tool_activity([
            AIMessage(content="", tool_calls=[{
                "id": "edit", "name": "replace_in_file",
                "args": {"path": "site/app.ts", "old_text": "SECRET", "new_text": "fixed"},
            }]),
            ToolMessage(
                content="Error: old_text was not found. Re-read the file and copy the text exactly. SECRET",
                tool_call_id="edit", name="replace_in_file",
            ),
        ])
        self.assertFalse(activity[0]["ok"])
        self.assertIn("anchor not found", activity[0]["diagnostic"])
        self.assertNotIn("SECRET", str(activity))

    def test_source_checker_rejects_private_file_and_missing_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("not-a-secret")
            (root / "example.rs").write_text("fn main() {}\n")
            with patch("tools.SAFE_ROOT", root):
                self.assertIn("Access denied", check_source_file.invoke({"path": ".env"}))
                with patch("tools.shutil.which", return_value=None):
                    self.assertIn("rustc is not installed", check_source_file.invoke({"path": "example.rs"}))

    def test_typescript_checker_uses_cached_deno_for_standalone_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.ts").write_text("const count: number = 1;\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", side_effect=lambda name: "/usr/bin/deno" if name == "deno" else None):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    result = check_source_file.invoke({"path": "app.ts"})
            self.assertIn("Checker: deno", result)
            self.assertEqual(run.call_args.args[0][:5], [
                "deno", "check", "--cached-only", "--no-lock", "--node-modules-dir=none",
            ])

    def test_web_and_shell_source_checks_do_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("<main>Hi</main>\n")
            (root / "styles.css").write_text("main { color: red; }\n")
            (root / "build.sh").write_text("#!/bin/sh\necho hi\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", return_value="/usr/bin/checker"):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    html = check_source_file.invoke({"path": "index.html"})
                    self.assertIn("browser behavior was not checked", html)
                    self.assertIn("CSS property validity", html)
                    self.assertEqual(run.call_args.args[0][:2], ["deno", "fmt"])
                    self.assertEqual(run.call_args.kwargs["input_text"], "<main>Hi</main>\n")
                    check_source_file.invoke({"path": "styles.css"})
                    self.assertEqual(run.call_args.args[0][-2:], ["css", "-"])
                    check_source_file.invoke({"path": "build.sh"})
                    self.assertEqual(run.call_args.args[0][:2], ["bash", "-n"])

    def test_language_test_runner_uses_manifest_and_fixed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "go.mod").write_text("module example.com/test\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", return_value="/usr/bin/go"):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    self.assertEqual(run_language_tests.invoke({"language": "go"}), "Exit code: 0")
            self.assertEqual(run.call_args.args[0], ["go", "test", "./..."])
            (root / "go.mod").unlink()
            with patch("tools.SAFE_ROOT", root):
                self.assertIn("go.mod is missing", run_language_tests.invoke({"language": "go"}))

    def test_deno_project_test_runner_uses_offline_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "deno.json").write_text("{}")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", return_value="/usr/bin/deno"):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    self.assertEqual(run_language_tests.invoke({"language": "typescript"}), "Exit code: 0")
            self.assertEqual(run.call_args.args[0], [
                "deno", "test", "--cached-only", "--no-lock", "--node-modules-dir=none",
            ])

    def test_standalone_deno_tests_run_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "test_site"
            site.mkdir()
            (site / "app_test.ts").write_text("Deno.test('smoke', () => {});\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.shutil.which", return_value="/usr/bin/deno"):
                with patch("tools._run_command", return_value="Exit code: 0") as run:
                    self.assertEqual(run_language_tests.invoke({
                        "language": "typescript", "path": "test_site",
                    }), "Exit code: 0")
            self.assertEqual(run.call_args.args[0], [
                "deno", "test", "--cached-only", "--no-config", "--no-lock",
                "--node-modules-dir=none",
            ])
            self.assertEqual(run.call_args.kwargs["cwd"], site)
            (site / "app_test.ts").unlink()
            with patch("tools.SAFE_ROOT", root):
                self.assertIn("no test manifest or standalone Deno tests", run_language_tests.invoke({
                    "language": "typescript", "path": "test_site",
                }))

    def test_tool_contract_requires_explicit_invocation_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = root / "test_new_tool.py"
            test.write_text("def test_unrelated():\n    assert True\n")
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root):
                self.assertIn("must call new_tool.invoke", verify_tool_contract.invoke({
                    "tool_name": "new_tool", "test_path": test.name,
                }))
                test.write_text("def test_new_tool():\n    new_tool.invoke({'path': 'x'})\n")
                with patch("tools._run_command", return_value="Exit code: 1"):
                    self.assertIn("not registered", verify_tool_contract.invoke({
                        "tool_name": "new_tool", "test_path": test.name,
                    }))
                with patch("tools._run_command", return_value="Exit code: 0"):
                    with patch("tools.run_tests", SimpleNamespace(invoke=lambda args: "Exit code: 0")):
                        result = verify_tool_contract.invoke({
                            "tool_name": "new_tool", "test_path": test.name,
                        })
            self.assertIn("registered", result)


class MCPAndSelectionTests(unittest.TestCase):
    def test_mcp_registry_persists_choices_and_protects_builtin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual([server.name for server in all_mcp_servers(root)], ["langchain-docs"])
            add_mcp_server(root, "files", "https://example.com/mcp", ("drive",), live_applied=True)
            self.assertTrue(tui_applied_current_config(root))
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE((root / ".wallace" / "mcp_servers.json").stat().st_mode), 0o600)
            self.assertEqual([server.name for server in load_mcp_servers(root)], ["files"])
            self.assertTrue(set_mcp_server_active(root, "files", False))
            self.assertEqual(load_mcp_servers(root), [])
            self.assertFalse(set_mcp_server_active(root, "files", False))
            self.assertTrue(set_mcp_server_active(root, "langchain-docs", False))
            self.assertEqual(
                [(server.name, server.enabled) for server in all_mcp_servers(root)],
                [("langchain-docs", False), ("files", False)],
            )
            with self.assertRaisesRegex(ValueError, "cannot be removed"):
                remove_mcp_server(root, "langchain-docs")
            with self.assertRaisesRegex(ValueError, "reserved"):
                add_mcp_server(root, "langchain-docs", "https://example.com/mcp")
            remove_mcp_server(root, "files")
            self.assertEqual([server.name for server in all_mcp_servers(root)], ["langchain-docs"])

    def test_read_only_mcp_tool_lists_links_and_connection_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clear_connection_statuses()
            add_mcp_server(root, "files", "https://example.com/mcp", ("drive",))
            set_connection_status("files", True)
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root):
                listed = json.loads(list_mcp_connections.invoke({}))
            self.assertEqual(listed["servers"][0]["name"], "langchain-docs")
            self.assertEqual(listed["servers"][1]["url"], "https://example.com/mcp")
            self.assertEqual(listed["servers"][1]["connection"], "connected")
            self.assertIn("list_mcp_connections", READ_ONLY_TOOLS)
            self.assertNotIn("add_mcp_server", {tool.name for tool in TOOLS})
            with patch("tools.SAFE_ROOT", root):
                self.assertIn("Access denied", read_safe_file.invoke({
                    "path": ".wallace/mcp_servers.json",
                }))
                self.assertIn("Access denied", write_safe_file.invoke({
                    "path": ".wallace/mcp_servers.json", "content": "{}",
                }))
            set_mcp_server_active(root, "files", False)
            with patch("tools.SAFE_ROOT", root), patch("tools.APP_ROOT", root):
                listed = json.loads(list_mcp_connections.invoke({}))
            self.assertEqual(listed["servers"][1]["connection"], "inactive")
            clear_connection_statuses()

    def test_mcp_registry_rejects_unsafe_urls_even_when_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for url in (
                "https://token@example.com/mcp",
                "http://example.com/mcp",
                "https://example.com/mcp?token=secret",
                "https://example.com/mcp\nother",
            ):
                with self.assertRaises(ValueError):
                    add_mcp_server(root, "unsafe", url)
            config = root / ".wallace" / "mcp_servers.json"
            config.parent.mkdir(exist_ok=True)
            config.write_text(json.dumps({"servers": [{
                "name": "unsafe", "url": "http://example.com/mcp", "enabled": False,
            }]}))
            with self.assertRaises(ValueError):
                all_mcp_servers(root)

    def test_private_mcp_config_requires_safe_url_and_explicit_enable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".wallace" / "mcp_servers.json"
            config.parent.mkdir()
            config.write_text(json.dumps({"servers": [
                {"name": "disabled", "url": "https://example.com/mcp"},
                {"name": "workspace", "url": "https://example.com/mcp", "enabled": True, "keywords": ["drive"]},
            ]}))
            servers = load_mcp_servers(root)
            self.assertEqual([server.name for server in servers], ["workspace"])
            self.assertEqual(servers[0].keywords, ("drive",))
            config.write_text(json.dumps({"servers": [{
                "name": "bad", "url": "https://token@example.com/mcp", "enabled": True,
            }]}))
            with self.assertRaisesRegex(ValueError, "without credentials"):
                load_mcp_servers(root)

    def test_configured_mcp_tools_always_need_approval(self):
        policy = approval_policy({"remote_read", "remote_write"})
        self.assertIn("remote_read", policy)
        self.assertIn("remote_write", policy)
        self.assertIn("execute_command", policy)

    def test_routing_uses_current_request_and_relevant_mcp_keywords(self):
        tools = [SimpleNamespace(name=name, metadata=None) for name in (
            "get_project_tree", "read_safe_file", "check_source_file",
            "get_system_info", "execute_command",
        )]
        tools.append(SimpleNamespace(name="list_drive", metadata={
            "wallace_mcp": {"server": "workspace", "keywords": ("drive", "workspace")},
        }))
        prompt = "Prior thread context: fix Python bugs.\nUser request:\nShow my Drive files"
        selected = {tool.name for tool in select_tools(prompt, tools)}
        self.assertIn("list_drive", selected)
        self.assertNotIn("check_source_file", selected)
        self.assertNotIn("execute_command", selected)
        code_selected = {tool.name for tool in select_tools("Fix the Python tool", tools)}
        self.assertIn("check_source_file", code_selected)
        self.assertNotIn("list_drive", code_selected)
        self.assertIn(
            "check_source_file",
            {tool.name for tool in select_tools("Create a website", tools)},
        )
        status_tools = tools + [SimpleNamespace(name="list_mcp_connections", metadata=None)]
        self.assertIn(
            "list_mcp_connections",
            {tool.name for tool in select_tools("Which MCP servers are connected?", status_tools)},
        )


class MCPSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_mcp_commands_rebuild_and_persist_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root):
                setup = AsyncMock(return_value=(object(), {"thread": "new"}))
                with patch("main.setup_agent", new=setup):
                    with patch("builtins.input", side_effect=[
                        "/mcp add files https://example.com/mcp drive",
                        "/mcp off files", "/mcp", "exit",
                    ]):
                        with patch("main.console"):
                            await run_agent()
            self.assertEqual(setup.await_count, 3)
            self.assertFalse(all_mcp_servers(root)[1].enabled)
            self.assertEqual(all_mcp_servers(root)[1].url, "https://example.com/mcp")

    async def test_builtin_docs_can_be_disabled_without_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_mcp_server_active(root, "langchain-docs", False)
            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root), patch("main.MCPAdapter") as adapter:
                from main import _load_docs_tools

                self.assertEqual(await _load_docs_tools(lambda warning: self.fail(warning)), [])
                adapter.assert_not_called()

    async def test_configured_mcp_loader_annotates_discovered_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".wallace" / "mcp_servers.json"
            config.parent.mkdir()
            config.write_text(json.dumps({"servers": [{
                "name": "files", "url": "https://example.com/mcp",
                "enabled": True, "keywords": ["drive"],
            }]}))
            seen = []

            class Adapter:
                def __init__(self, url):
                    seen.append(url)

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return None

                async def list_tools(self):
                    return [SimpleNamespace(name="list_files", metadata={})]

            with patch("main.__file__", str(root / "main.py")), patch("main.APP_ROOT", root), patch("main.MCPAdapter", Adapter):
                tools = await _load_configured_mcp_tools(lambda warning: self.fail(warning))
            self.assertEqual(seen, ["https://example.com/mcp"])
            self.assertEqual(tools[0].metadata["wallace_mcp"]["keywords"], ("drive", "files"))

    async def test_async_middleware_filters_tools_before_model_call(self):
        class Request:
            def __init__(self, tools):
                self.tools = tools
                self.state = {"messages": [HumanMessage(content="Fix Python code")]}

            def override(self, *, tools):
                return Request(tools)

        available = [SimpleNamespace(name=name, metadata=None) for name in (
            "get_project_tree", "check_source_file", "get_system_info",
        )]

        async def handler(request):
            return [tool.name for tool in request.tools]

        chosen = await focus_tools_on_request.awrap_model_call(Request(available), handler)
        self.assertIn("check_source_file", chosen)
        self.assertNotIn("get_system_info", chosen)

    async def test_configured_server_tool_is_registered_with_approval(self):
        remote = SimpleNamespace(name="list_remote_files", metadata={
            "wallace_mcp": {"server": "remote", "keywords": ("remote",)},
        })
        with patch("main.DEV_MODE", True), patch("main._load_docs_tools", new=AsyncMock(return_value=[])):
            with patch("main._load_configured_mcp_tools", new=AsyncMock(return_value=[remote])):
                with patch("main.ChatOllama"):
                    with patch("main.create_agent", return_value=object()) as create:
                        await setup_agent(model_name="example:latest")
        kwargs = create.call_args.kwargs
        self.assertIn(remote, kwargs["tools"])
        self.assertIn("list_remote_files", kwargs["middleware"][1].interrupt_on)


if __name__ == "__main__":
    unittest.main()
