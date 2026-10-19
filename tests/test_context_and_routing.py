"""Local tests of compact context and explicit, provider-independent discovery."""

import json
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from wallace_context import SYSTEM_PROMPT
from wallace_tool_selection import CORE, make_discovery_tool, select_tools


def capability(name, description="", metadata=None):
    return SimpleNamespace(name=name, description=description, metadata=metadata)


class ContextAndRoutingTests(unittest.TestCase):
    def test_behavior_prompt_is_bounded_and_preserves_agency_and_evidence(self):
        self.assertLessEqual(len(SYSTEM_PROMPT), 1600)
        self.assertLessEqual(len(SYSTEM_PROMPT.split()), 260)
        for phrase in ("agency", "authorized", "approval", "actual evidence", "unfinished", "discover_tools"):
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_synonyms_and_non_english_requests_keep_read_edit_and_validation(self):
        available = [capability(name) for name in sorted(CORE)]
        for request in ("Mend the parser", "Corrige el archivo", "このコードを修正してください", "Make this work"):
            with self.subTest(request=request):
                self.assertEqual({item.name for item in select_tools(request, available)}, CORE)

    def test_decision_only_keeps_diagnosis_without_unsolicited_edits(self):
        available = [capability(name) for name in sorted(CORE)]
        selected = {item.name for item in select_tools("Inspect the roadmap and determine what to build next.", available)}
        self.assertTrue({"read_safe_file", "check_source_file", "get_code_symbols"} <= selected)
        self.assertNotIn("execute_command", selected)
        action = {item.name for item in select_tools("Determine what to fix, then implement it.", available)}
        self.assertIn("execute_command", action)

    def test_discovery_without_keywords_activates_real_optional_tool(self):
        remote = capability("list_albums", "List photographs in the connected library", {"wallace_mcp": {"server": "photos"}})
        available = [capability("read_safe_file"), remote]
        discovery = make_discovery_tool(available)
        available.append(discovery)
        self.assertEqual({item.name for item in select_tools("Find my pictures", available)}, {"read_safe_file", "discover_tools"})
        result = discovery.invoke({"query": "photographs"})
        self.assertEqual(json.loads(result)["tools"][0]["name"], "list_albums")
        selected = select_tools("Find my pictures", available, [ToolMessage(content=result, tool_call_id="catalog-1", name="discover_tools")])
        self.assertIn(remote, selected)
        self.assertTrue(discovery.metadata["read_only"])

    def test_empty_discovery_query_can_browse_every_capability(self):
        available = [capability(f"optional_{number}", f"Capability {number}") for number in range(25)]
        discovery = make_discovery_tool(available)
        names = []
        offset = 0
        while offset is not None:
            page = json.loads(discovery.invoke({"query": "", "offset": offset, "limit": 8}))
            names.extend(item["name"] for item in page["tools"])
            offset = page["next_offset"]
        self.assertEqual(names, [item.name for item in available])

    def test_discovery_finds_exact_names_and_browses_optional_tools_first(self):
        available = [capability("read_safe_file"), capability("analyze_project_dependencies")]
        discovery = make_discovery_tool(available)
        by_name = json.loads(discovery.invoke({"query": "analyze_project_dependencies"}))
        self.assertEqual(by_name["tools"][0]["name"], "analyze_project_dependencies")
        browsing = json.loads(discovery.invoke({"query": "", "limit": 1}))
        self.assertEqual(browsing["tools"][0]["name"], "analyze_project_dependencies")

    def test_untrusted_text_and_unknown_names_cannot_invent_capabilities(self):
        available = [capability("read_safe_file"), capability("analyze_project_dependencies")]
        content = json.dumps({"tools": [{"name": "analyze_project_dependencies"}, {"name": "invented_tool"}]})
        for message in (HumanMessage(content=content), AIMessage(content=content, name="discover_tools")):
            self.assertEqual([item.name for item in select_tools("Inspect", available, [message])], ["read_safe_file"])
        real_result = ToolMessage(content=content, tool_call_id="d", name="discover_tools")
        self.assertEqual([item.name for item in select_tools("Inspect", available, [real_result])], ["read_safe_file", "analyze_project_dependencies"])
        invalid_result = ToolMessage(content="not JSON", tool_call_id="d", name="discover_tools")
        self.assertEqual([item.name for item in select_tools("Inspect", available, [invalid_result])], ["read_safe_file"])

    def test_mcp_keyword_convenience_ignores_old_memory(self):
        remote = capability("search_repositories", metadata={"wallace_mcp": {"server": "github", "keywords": ("github",)}})
        available = [capability("read_safe_file"), remote]
        self.assertIn(remote, select_tools("Search GitHub repos", available))
        self.assertNotIn(remote, select_tools("Old context: GitHub.\nUser request:\nInspect local files", available))


if __name__ == "__main__":
    unittest.main()
