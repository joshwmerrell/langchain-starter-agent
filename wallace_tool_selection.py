"""Stable local tools plus explicit discovery of optional capabilities.

Tool visibility is an efficiency choice, not an authorization boundary. The
host's approval policy still applies to every exposed tool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from wallace_planning.cli import is_decision_request


DISCOVERY_TOOL_NAME = "discover_tools"
CORE = frozenset({
    "get_project_tree", "read_safe_file", "search_in_files", "get_code_symbols",
    "get_git_status", "get_git_diff", "check_source_file", "run_tests",
    "run_language_tests", "verify_tool_contract", "write_safe_file",
    "replace_in_file", "edit_file_lines", "execute_command", "search_conversation",
})
EDIT = frozenset({
    "write_safe_file", "replace_in_file", "edit_file_lines", "execute_command",
})
STATUS_WORDS = re.compile(r"\b(status|tokens?|usage|quota|limits?|remaining|session)\b", re.I)
SYSTEM_WORDS = re.compile(r"\b(system|computer|hardware|cpu|gpu|vram|memory|ram|disk|network|processes)\b", re.I)
MCP_WORDS = re.compile(r"\b(mcp|servers?|connections?)\b", re.I)
DOC_WORDS = re.compile(r"\b(langchain|documentation|docs|mcp)\b", re.I)


def current_request(prompt: str) -> str:
    """Ignore durable plan and older thread memory when matching conveniences."""
    if "\nUser request:\n" in prompt:
        return prompt.rsplit("\nUser request:\n", 1)[-1].split(
            "\n\nAnswer the current user request", 1
        )[0]
    if "\n</thread_memory>\n\n" in prompt:
        prompt = prompt.rsplit("\n</thread_memory>\n\n", 1)[-1]
    return prompt.split("\n\nAnswer the current user request", 1)[0]


def _field(value: object, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _discovered_names(messages: Sequence) -> set[str]:
    """Only tool-result messages can activate names returned by discovery."""
    names: set[str] = set()
    for message in messages:
        if _field(message, "type", _field(message, "role")) != "tool":
            continue
        if _field(message, "name") != DISCOVERY_TOOL_NAME:
            continue
        content = _field(message, "content")
        try:
            result = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError):
            continue
        if not isinstance(result, Mapping) or not isinstance(result.get("tools"), list):
            continue
        for entry in result["tools"]:
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                names.add(entry["name"])
    return names


def make_discovery_tool(available: Sequence):
    """Create a read-only catalog tool bound to this agent's real capabilities."""
    from langchain_core.tools import tool

    catalog = []
    seen: set[str] = set()
    for capability in available:
        name = capability.name
        if name == DISCOVERY_TOOL_NAME or name in seen:
            continue
        seen.add(name)
        catalog.append({"name": name, "description": str(getattr(capability, "description", "") or "")[:400]})
    # Browsing should reveal capabilities the model has not already received.
    catalog.sort(key=lambda entry: entry["name"] in CORE)

    @tool
    def discover_tools(query: str = "", offset: int = 0, limit: int = 8) -> str:
        """Find available tools by name or description and enable the returned tools.

        Use an empty query to browse all capabilities without guessing keywords.
        Results are paged; use next_offset to continue. This only lists tools.
        """
        terms = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        for index, entry in enumerate(catalog):
            text = (entry["name"] + " " + entry["name"].replace("_", " ") + " " + entry["description"]).casefold()
            score = sum(term in text for term in terms)
            if not terms or score:
                ranked.append((-score, index, entry))
        ranked.sort(key=lambda item: item[:2])
        offset = max(0, offset)
        limit = max(1, min(20, limit))
        matches = [item[2] for item in ranked]
        page = matches[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(matches) else None
        return json.dumps({
            "tools": page, "total": len(matches), "next_offset": next_offset,
            "hint": "Use an empty query to browse every available tool." if not page else "Returned tools are available on the next model step.",
        }, ensure_ascii=False)

    discover_tools.metadata = {"read_only": True, "wallace_read_only": True}
    return discover_tools


def select_tools(prompt: str, available: Sequence, messages: Sequence = ()) -> list:
    """Expose the same useful core for any language, plus discovered tools."""
    task = current_request(prompt)
    selected = set(CORE) | {DISCOVERY_TOOL_NAME} | _discovered_names(messages)
    if STATUS_WORDS.search(task):
        selected.add("get_session_status")
    if SYSTEM_WORDS.search(task):
        selected.add("get_system_info")
    if MCP_WORDS.search(task):
        selected.add("list_mcp_connections")
    for capability in available:
        meta = (getattr(capability, "metadata", None) or {}).get("wallace_mcp")
        if not isinstance(meta, Mapping):
            continue
        keywords = meta.get("keywords", ())
        if meta.get("server") == "langchain-docs" and DOC_WORDS.search(task):
            selected.add(capability.name)
        elif any(
            isinstance(word, str) and word and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", task, re.I)
            for word in keywords
        ):
            selected.add(capability.name)
    if is_decision_request(task):
        selected.difference_update(EDIT)
    return [capability for capability in available if capability.name in selected]
