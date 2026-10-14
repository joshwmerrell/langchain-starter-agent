"""Conservative, task-relevant tool filtering for the local model.

Ambiguous requests keep the normal tool set. Only clearly irrelevant tool
families are omitted, so a user can still ask Wallace to perform new tasks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


BASE = {
    "get_project_tree", "read_safe_file", "search_in_files",
    "get_git_status", "get_git_diff",
}
CODE = {
    "get_code_symbols", "check_python_syntax", "check_source_file",
    "summarize_project_file", "analyze_project_dependencies",
    "analyze_git_change_impact", "run_tests", "run_language_tests",
    "verify_tool_contract",
}
EDIT = {"write_safe_file", "replace_in_file", "execute_command"}
CODE_WORDS = re.compile(
    r"\b(code|coding|program|python|javascript|typescript|rust|golang|go module|"
    r"compile|syntax|test|bug|fix|implement|refactor|build|tool|roadmap|dependency|"
    r"html|css|website|web\s+page|web\s+app|webapp|front.?end|\.py|\.js|\.ts|\.tsx|\.go|\.rs)\b", re.IGNORECASE,
)
EDIT_WORDS = re.compile(
    r"\b(write|edit|change|modify|create|add|remove|delete|fix|implement|build|"
    r"run|execute|test|install|update|apply|refactor)\b", re.IGNORECASE,
)
SYSTEM_WORDS = re.compile(
    r"\b(system|computer|hardware|cpu|gpu|vram|memory|ram|disk|network|processes)\b",
    re.IGNORECASE,
)
DOC_WORDS = re.compile(r"\b(langchain|documentation|docs|mcp)\b", re.IGNORECASE)
MCP_WORDS = re.compile(r"\b(mcp|servers?|connections?)\b", re.IGNORECASE)


def current_request(prompt: str) -> str:
    """Ignore the durable plan and older thread memory when routing tools."""
    if "\nUser request:\n" in prompt:
        return prompt.rsplit("\nUser request:\n", 1)[-1].split(
            "\n\nAnswer the current user request", 1
        )[0]
    if "\n</thread_memory>\n\n" in prompt:
        prompt = prompt.rsplit("\n</thread_memory>\n\n", 1)[-1]
    if "\n\nAnswer the current user request" in prompt:
        prompt = prompt.split("\n\nAnswer the current user request", 1)[0]
    return prompt


def select_tools(prompt: str, available: Sequence) -> list:
    """Select relevant tools while preserving their original order."""
    task = current_request(prompt)
    if re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you|good morning)[.!?\s]*", task, re.IGNORECASE):
        return []
    names = {tool.name for tool in available}
    selected = names & BASE
    code_requested = CODE_WORDS.search(task) is not None
    edit_requested = EDIT_WORDS.search(task) is not None
    system_requested = SYSTEM_WORDS.search(task) is not None
    external_requested = False
    if code_requested:
        selected |= names & CODE
    if edit_requested:
        selected |= names & EDIT
    if system_requested:
        selected.add("get_system_info")
    if MCP_WORDS.search(task):
        selected.add("list_mcp_connections")
    for tool in available:
        meta = (getattr(tool, "metadata", None) or {}).get("wallace_mcp")
        if not meta:
            continue
        keywords = meta.get("keywords", ())
        if meta.get("server") == "langchain-docs":
            relevant = DOC_WORDS.search(task) is not None
        else:
            relevant = any(
                re.search(rf"\b{re.escape(word)}\b", task, re.IGNORECASE)
                for word in keywords
            )
        if relevant:
            external_requested = True
            selected.add(tool.name)
    # A novel or ambiguous request must not lose capabilities by accident.
    if not (code_requested or edit_requested or system_requested or external_requested):
        return list(available)
    return [tool for tool in available if tool.name in selected]
