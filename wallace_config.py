"""Runtime model settings and context budgets loaded once at startup."""

import os

from dotenv import load_dotenv
from wallace_runtime import APP_ROOT

load_dotenv(APP_ROOT / ".env")

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer configuration value with a safe fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "SetneufPT/Qwopus3.5-9B-Coder_Q3_64k_8GB-GPU:latest",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = _env_int("WALLACE_NUM_CTX", 32768)
OLLAMA_NUM_PREDICT = _env_int("WALLACE_NUM_PREDICT", 4096)
OLLAMA_KEEP_ALIVE = os.getenv("WALLACE_KEEP_ALIVE", "15m")

# Keep the visible TUI log bounded, but retain the full thread transcript on
# disk. A rolling summary plus recent turns provides bounded model context.
MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_CHARS = 60_000
MAX_MEMORY_CHARS = min(8_000, max(600, OLLAMA_NUM_CTX // 4))
MEMORY_RECENT_MESSAGES = 10
MEMORY_COMPACT_AFTER_MESSAGES = 20


