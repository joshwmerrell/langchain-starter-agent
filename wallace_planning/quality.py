"""Small host-side checks for unusable model text."""

from collections import Counter


def response_quality_error(response: str) -> str | None:
    """Detect obvious model-output loops before displaying or saving them."""
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
        return (
            "Ollama generated a repetitive response instead of a usable answer. "
            "Wallace did not accept it as completion; the request remains unfinished. "
            "If it recurs, try a smaller step or a different tool-calling model."
        )
    return None
