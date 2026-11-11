import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AGY_RUNTIME_CWD = BASE_DIR / "agy_runtime_cwd"
AGY_RUNTIME_CWD.mkdir(parents=True, exist_ok=True)

# Server settings
HOST = os.getenv("AGY_WRAPPER_HOST", "127.0.0.1")
PORT = int(os.getenv("AGY_WRAPPER_PORT", "8000"))

# AGY CLI settings
AGY_BIN = os.getenv("AGY_BIN", "agy")
# Optional model override (e.g. "gemini-4-argon-high", "gemini-3.8-flash-high").
# Empty string uses the default model configured in ~/.gemini/antigravity-cli/settings.json
DEFAULT_AGY_MODEL = os.getenv("AGY_MODEL", "")
# Leave effort unset unless the backend operator explicitly configures it.  AGY
# rejects --effort for some default/custom models, so a wrapper-side default is
# not safe.
DEFAULT_AGY_EFFORT = os.getenv("AGY_EFFORT", "")
AGY_TIMEOUT_SECONDS = int(os.getenv("AGY_TIMEOUT_SECONDS", "300"))

# Responses API state is process-local. Bound it so abandoned conversations cannot
# retain an unbounded amount of Python memory.
RESPONSE_STATE_MAX_ENTRIES = int(os.getenv("RESPONSE_STATE_MAX_ENTRIES", "100"))
RESPONSE_STATE_TTL_SECONDS = int(os.getenv("RESPONSE_STATE_TTL_SECONDS", "3600"))

# Character guardrail for malformed or repeated client transcripts. This is not a
# tokenizer limit; it rejects oversized requests before AGY is launched.
AGY_MAX_PROMPT_CHARS = int(os.getenv("AGY_MAX_PROMPT_CHARS", "900000"))

# HTTP and subprocess resource limits. These defaults bound local single-host use.
HTTP_READ_TIMEOUT_SECONDS = int(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "30"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "10000000"))
MAX_CONCURRENT_AGY_PROCESSES = int(os.getenv("MAX_CONCURRENT_AGY_PROCESSES", "2"))

# Deterministic rolling context window. A model-generated summary is intentionally
# not fabricated here; overflow is marked so callers know prior detail was dropped.
CONTEXT_WINDOW_MAX_ITEMS = int(os.getenv("CONTEXT_WINDOW_MAX_ITEMS", "64"))
CONTEXT_WINDOW_MAX_BYTES = int(os.getenv("CONTEXT_WINDOW_MAX_BYTES", "450000"))
