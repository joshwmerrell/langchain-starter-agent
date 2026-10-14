"""Small host-side checks for unusable model text."""

from collections import Counter
from difflib import SequenceMatcher
import re


def response_quality_error(
    response: str,
    *,
    previous_assistant: str | None = None,
    current_request: str = "",
) -> str | None:
    """Detect output loops and copied replies before displaying or saving them."""
    issue = (
        "Ollama generated a repetitive response instead of a usable answer. "
        "Wallace did not accept it as completion; the request remains unfinished. "
        "If it recurs, try a smaller step or a different tool-calling model."
    )
    if previous_assistant and not re.search(
        r"\b(?:repeat|quote|verbatim|create|write|draft|generate|rewrite|"
        r"summarize|translate|transform|convert|build|implement|edit|produce)\b"
        r"|say that again",
        current_request.casefold(),
    ):
        previous = re.sub(r"\s+", " ", previous_assistant.casefold())[:1200]
        current = re.sub(r"\s+", " ", response.casefold())[:1200]
        if min(len(previous), len(current)) >= 160:
            threshold = max(150, min(200, int(min(len(previous), len(current)) * 0.6)))
            if SequenceMatcher(None, previous, current, autojunk=False).find_longest_match().size >= threshold:
                return (
                    "Ollama copied an earlier reply instead of answering the current request. "
                    "Wallace did not accept it as completion; the request remains unfinished."
                )
    if len(response) < 500:
        return None
    long_lines = Counter(
        line.strip() for line in response.splitlines() if len(line.strip()) >= 40
    )
    repeated_line = any(
        count >= 4 and count * len(line) >= len(response) * 0.5
        for line, count in long_lines.items()
    )
    stray_thinking = (
        response.count("</think>") >= 3
        and response.count("<think>") < response.count("</think>")
    )
    if repeated_line or stray_thinking:
        return issue
    words = re.findall(r"\b\w+\b", response.casefold())
    if len(words) >= 200:
        windows = Counter(tuple(words[index:index + 5]) for index in range(len(words) - 4))
        repeated_windows = sum(count for count in windows.values() if count >= 3)
        if repeated_windows >= (len(words) - 4) * 0.55:
            return issue
    return None
