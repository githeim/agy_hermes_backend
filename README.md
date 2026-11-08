# AGY Brain-Only Wrapper Server for Hermes Runtime (`/v1/responses`)

[한국어](README.ko.md)

This project is an **OpenAI Responses API (`POST /v1/responses`)** compatible wrapper server that uses the **Antigravity CLI (`agy`)** as the reasoning-only backend (brain) for the **Hermes Agent Runtime**.

The server itself runs with only the **Python 3 standard library (`asyncio`)** and has no external Python package dependency.

---

## How It Works

### Brain and Body Separation

- **Wrapper / AGY (Brain):** Built-in AGY tool execution is disabled through `--settings '{"permissions":{"allow":[],"deny":["*"]}}'` and the rules in `agy_runtime_cwd/GEMINI.md`. AGY returns only its reasoning result and tool-call decisions (`content` + `tool_calls`) in one turn.
- **Hermes Runtime (Body):** Hermes receives each `type: "function_call"` item returned by the wrapper, executes the real local tool (`terminal`, `clarify`, `memory`, `patch`, and so on), then sends the result back as `type: "function_call_output"`.

### Supported APIs and Features

- OpenAI Responses API: `POST /v1/responses`
  - `instructions`
  - `input` as a string or an array of `message`, `function_call`, and `function_call_output` items
  - In-memory state chaining with `previous_response_id`
  - `tools` and `tool_choice` (`auto`, `none`, `required`, or a forced function)
  - Typed SSE events for `stream: true` requests, including `response.created`, `response.output_item.added`, `response.function_call_arguments.delta`, and `response.completed`
- OpenAI Chat Completions-compatible API: `POST /v1/chat/completions`
- Model listing: `GET /v1/models`
- Health check: `GET /health`

---

## Repository Layout

- `server.py`: Async HTTP/1.1 + SSE server and API endpoints
- `agy_runner.py`: Runs the `agy` CLI in non-interactive brain-only mode and extracts responses and token usage
- `prompt_builder.py`: Serializes Responses API requests, conversation history, and tool schemas into one AGY prompt; parses AGY JSON output
- `schemas.py`: Internal data models (`dataclass`)
- `config.py`: Environment-based configuration for host, port, default model (`AGY_MODEL`), reasoning effort (`AGY_EFFORT`), and limits
- `agy_runtime_cwd/GEMINI.md`: Isolated AGY subprocess working directory and brain-only instructions
- `test_responses_api.py`: Verification script for normal tool calls, multi-turn `function_call_output`, and SSE streaming with `clarify`
- `test_server_safety.py`: pytest test for context and request-size guards

---

## Running the Server

### 1. Start the server

Install and authenticate the `agy` CLI, then start the server from the cloned repository.

```bash
cd /path/to/agy_backend
python3 server.py
```

By default, the server listens on `127.0.0.1:8000` and uses the default AGY model configured in `~/.gemini/antigravity-cli/settings.json`.

To explicitly set a model and reasoning effort:

```bash
AGY_MODEL="gemini-4-argon-high" AGY_EFFORT="high" AGY_WRAPPER_PORT=8000 python3 server.py
```

### 2. Configure Hermes

Add the backend to `~/.hermes/config.yaml`.

```yaml
model:
  default: "gemini-4-argon-high"   # Or "agy-hermes-backend", "gemini-3.8-flash-high", etc.
  provider: "custom"
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "agy-local"             # Placeholder value for the local wrapper
  api_mode: "codex_responses"
```

### 3. Configure Resource Limits

The defaults allow at most two concurrent AGY processes, a 10 MB HTTP request body, and a 30-second HTTP request read timeout.

```bash
MAX_CONCURRENT_AGY_PROCESSES=2 MAX_REQUEST_BODY_BYTES=10000000 \
HTTP_READ_TIMEOUT_SECONDS=30 AGY_MAX_PROMPT_CHARS=900000 python3 server.py
```

`/v1/responses` and `/v1/chat/completions` enforce the same prompt-size limit. Requests over that limit receive HTTP 413.

---

## Verification

Run the standard-library integration verification script:

```bash
python3 test_responses_api.py
```

`test_server_safety.py` is optional and requires `pytest`:

```bash
python3 -m pytest -q test_server_safety.py
```