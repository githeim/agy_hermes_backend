import asyncio
import json
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional, Union

from agy_runner import VALID_AGY_MODELS, run_agy_brain_turn
from config import (
    AGY_MAX_PROMPT_CHARS,
    CONTEXT_WINDOW_MAX_BYTES,
    CONTEXT_WINDOW_MAX_ITEMS,
    HOST,
    HTTP_READ_TIMEOUT_SECONDS,
    MAX_CONCURRENT_AGY_PROCESSES,
    MAX_REQUEST_BODY_BYTES,
    PORT,
    RESPONSE_STATE_MAX_ENTRIES,
    RESPONSE_STATE_TTL_SECONDS,
)
from prompt_builder import build_agy_prompt, normalize_input_items

# In-memory, bounded store for `previous_response_id` state chaining.
# OrderedDict keeps least-recently-used responses at the front.
RESPONSE_STATE_STORE: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
AGY_PROCESS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_AGY_PROCESSES)


class RequestTooLargeError(ValueError):
    """Raised before an unbounded HTTP request body is accumulated."""


def _item_fingerprint(item: dict[str, Any]) -> str:
    """Return a stable representation used only to find transcript overlap."""
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _longest_overlap(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> int:
    """Find the largest previous suffix that is already the current prefix."""
    limit = min(len(previous), len(current))
    for length in range(limit, 0, -1):
        if all(
            _item_fingerprint(previous[-length + index]) == _item_fingerprint(current[index])
            for index in range(length)
        ):
            return length
    return 0


def _load_response_state(response_id: Optional[str]) -> list[dict[str, Any]]:
    """Load a non-expired state and refresh its LRU position."""
    if not response_id:
        return []
    stored = RESPONSE_STATE_STORE.get(response_id)
    if stored is None:
        return []
    created_at, items = stored
    if time.monotonic() - created_at > RESPONSE_STATE_TTL_SECONDS:
        del RESPONSE_STATE_STORE[response_id]
        return []
    RESPONSE_STATE_STORE.move_to_end(response_id)
    return items


def _store_response_state(response_id: str, items: list[dict[str, Any]]) -> None:
    """Store state with TTL/LRU eviction to prevent a process-memory leak."""
    now = time.monotonic()
    expired_ids = [
        state_id
        for state_id, (created_at, _state_items) in RESPONSE_STATE_STORE.items()
        if now - created_at > RESPONSE_STATE_TTL_SECONDS
    ]
    for expired_id in expired_ids:
        del RESPONSE_STATE_STORE[expired_id]

    RESPONSE_STATE_STORE[response_id] = (now, items)
    RESPONSE_STATE_STORE.move_to_end(response_id)
    while len(RESPONSE_STATE_STORE) > RESPONSE_STATE_MAX_ENTRIES:
        RESPONSE_STATE_STORE.popitem(last=False)


def _merge_response_input(
    previous_items: list[dict[str, Any]], current_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge incremental input without duplicating a client-resubmitted transcript."""
    if not previous_items:
        return current_items
    overlap = _longest_overlap(previous_items, current_items)
    return previous_items + current_items[overlap:]


def _compact_context_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep system instructions and the newest bounded transcript items."""
    if len(items) <= CONTEXT_WINDOW_MAX_ITEMS:
        encoded = len(json.dumps(items, ensure_ascii=False).encode("utf-8"))
        if encoded <= CONTEXT_WINDOW_MAX_BYTES:
            return items

    pinned = [item for item in items if item.get("role") in ("system", "developer")]
    recent: list[dict[str, Any]] = []
    used = sum(len(json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in pinned)
    for item in reversed(items):
        if item.get("role") in ("system", "developer"):
            continue
        item_size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
        if recent and (
            used + item_size > CONTEXT_WINDOW_MAX_BYTES
            or len(pinned) + len(recent) >= CONTEXT_WINDOW_MAX_ITEMS - 1
        ):
            break
        recent.append(item)
        used += item_size
    marker = {"type": "message", "role": "system", "content": "Earlier conversation was compacted by the backend; do not assume omitted details."}
    return pinned + [marker] + list(reversed(recent))


def _validate_prompt_size(prompt: str) -> None:
    """Reject a prompt before it is handed to an AGY subprocess."""
    prompt_size = len(prompt.encode("utf-8"))
    if prompt_size > AGY_MAX_PROMPT_CHARS:
        raise ValueError(
            "Request context exceeds the configured AGY_MAX_PROMPT_CHARS "
            f"limit ({prompt_size} > {AGY_MAX_PROMPT_CHARS} bytes). "
            "Start a new conversation or reduce the client-supplied history."
        )


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    """Format a Server-Sent Event for the OpenAI Responses API."""
    data = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_health_payload() -> dict[str, Any]:
    return {"status": "ok", "backend": "agy", "mode": "brain-only-responses-api"}


def get_models_payload() -> dict[str, Any]:
    now = int(time.time())
    models = ["agy-hermes-backend"] + sorted(VALID_AGY_MODELS)
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "antigravity",
            }
            for m in models
        ],
    }


async def process_responses_api(
    body: dict[str, Any],
) -> Union[dict[str, Any], AsyncIterator[str]]:
    """
    Core handler for OpenAI Responses API (`POST /v1/responses`).
    Returns a dict when `stream=False`, or an async iterator of SSE strings when `stream=True`.
    """
    model_name: str = body.get("model", "agy-hermes-backend")
    instructions: Optional[str] = body.get("instructions")
    raw_input: Any = body.get("input", [])
    previous_response_id: Optional[str] = body.get("previous_response_id")
    tools: Optional[list[dict[str, Any]]] = body.get("tools")
    tool_choice: Any = body.get("tool_choice")
    stream: bool = bool(body.get("stream", False))

    reasoning_cfg = body.get("reasoning")
    effort: Optional[str] = None
    if isinstance(reasoning_cfg, dict) and reasoning_cfg.get("effort"):
        effort = str(reasoning_cfg["effort"])

    current_items = normalize_input_items(raw_input)
    previous_items = _load_response_state(previous_response_id)
    full_input_items = _merge_response_input(previous_items, current_items)
    full_input_items = _compact_context_items(full_input_items)

    prompt = build_agy_prompt(
        instructions=instructions,
        input_items=full_input_items,
        tools=tools,
        tool_choice=tool_choice,
    )
    _validate_prompt_size(prompt)

    async with AGY_PROCESS_SEMAPHORE:
        turn_result = await run_agy_brain_turn(
            prompt=prompt,
            requested_model=model_name,
            effort=effort,
        )

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    output_items: list[dict[str, Any]] = []

    # 1. Assistant message item (if content is present OR no tool calls were emitted)
    if turn_result.content or not turn_result.tool_calls:
        msg_item = {
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": turn_result.content or "",
                    "annotations": [],
                }
            ],
        }
        output_items.append(msg_item)

    # 2. Function call items (to be executed by Hermes runtime)
    for tc in turn_result.tool_calls:
        fc_item = {
            "id": f"fc_{uuid.uuid4().hex[:16]}",
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:16]}",
            "name": tc.name,
            "arguments": tc.arguments_json,
        }
        output_items.append(fc_item)

    # Save the de-duplicated conversation chain for follow-up requests.
    _store_response_state(response_id, full_input_items + output_items)

    completed_response_obj = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model_name,
        "output": output_items,
        "usage": turn_result.usage,
    }

    if not stream:
        return completed_response_obj

    async def responses_sse_generator() -> AsyncIterator[str]:
        seq = 0

        def next_seq() -> int:
            nonlocal seq
            val = seq
            seq += 1
            return val

        in_progress_resp = {
            **completed_response_obj,
            "status": "in_progress",
            "output": [],
        }

        yield sse_event(
            "response.created",
            {"sequence_number": next_seq(), "response": in_progress_resp},
        )
        yield sse_event(
            "response.in_progress",
            {"sequence_number": next_seq(), "response": in_progress_resp},
        )

        for out_idx, item in enumerate(output_items):
            if item["type"] == "message":
                text_val = item["content"][0]["text"]
                in_prog_msg = {**item, "status": "in_progress", "content": []}
                yield sse_event(
                    "response.output_item.added",
                    {
                        "sequence_number": next_seq(),
                        "output_index": out_idx,
                        "item": in_prog_msg,
                    },
                )
                part_obj = {"type": "output_text", "text": "", "annotations": []}
                yield sse_event(
                    "response.content_part.added",
                    {
                        "sequence_number": next_seq(),
                        "item_id": item["id"],
                        "output_index": out_idx,
                        "content_index": 0,
                        "part": part_obj,
                    },
                )
                if text_val:
                    yield sse_event(
                        "response.output_text.delta",
                        {
                            "sequence_number": next_seq(),
                            "item_id": item["id"],
                            "output_index": out_idx,
                            "content_index": 0,
                            "delta": text_val,
                        },
                    )
                yield sse_event(
                    "response.output_text.done",
                    {
                        "sequence_number": next_seq(),
                        "item_id": item["id"],
                        "output_index": out_idx,
                        "content_index": 0,
                        "text": text_val,
                    },
                )
                yield sse_event(
                    "response.content_part.done",
                    {
                        "sequence_number": next_seq(),
                        "item_id": item["id"],
                        "output_index": out_idx,
                        "content_index": 0,
                        "part": {**part_obj, "text": text_val},
                    },
                )
                yield sse_event(
                    "response.output_item.done",
                    {
                        "sequence_number": next_seq(),
                        "output_index": out_idx,
                        "item": item,
                    },
                )

            elif item["type"] == "function_call":
                args_val = item["arguments"]
                in_prog_fc = {**item, "status": "in_progress", "arguments": ""}
                yield sse_event(
                    "response.output_item.added",
                    {
                        "sequence_number": next_seq(),
                        "output_index": out_idx,
                        "item": in_prog_fc,
                    },
                )
                yield sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "sequence_number": next_seq(),
                        "item_id": item["id"],
                        "output_index": out_idx,
                        "delta": args_val,
                    },
                )
                yield sse_event(
                    "response.function_call_arguments.done",
                    {
                        "sequence_number": next_seq(),
                        "item_id": item["id"],
                        "output_index": out_idx,
                        "arguments": args_val,
                    },
                )
                yield sse_event(
                    "response.output_item.done",
                    {
                        "sequence_number": next_seq(),
                        "output_index": out_idx,
                        "item": item,
                    },
                )

        yield sse_event(
            "response.completed",
            {"sequence_number": next_seq(), "response": completed_response_obj},
        )

    return responses_sse_generator()


async def process_chat_completions_api(
    body: dict[str, Any],
) -> Union[dict[str, Any], AsyncIterator[str]]:
    """Fallback handler for OpenAI Chat Completions API (`POST /v1/chat/completions`)."""
    messages = body.get("messages", [])
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    stream = bool(body.get("stream", False))
    model_name = body.get("model", "agy-hermes-backend")

    prompt = build_agy_prompt(
        instructions=None,
        input_items=_compact_context_items(messages),
        tools=tools,
        tool_choice=tool_choice,
    )
    _validate_prompt_size(prompt)

    async with AGY_PROCESS_SEMAPHORE:
        turn_result = await run_agy_brain_turn(
            prompt=prompt,
            requested_model=model_name,
        )

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    openai_tool_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments_json,
            },
        }
        for tc in turn_result.tool_calls
    ]
    has_tools = len(openai_tool_calls) > 0
    finish_reason = "tool_calls" if has_tools else "stop"

    if not stream:
        msg_obj: dict[str, Any] = {
            "role": "assistant",
            "content": turn_result.content,
        }
        if has_tools:
            msg_obj["tool_calls"] = openai_tool_calls
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": turn_result.usage.get("input_tokens", 0),
                "completion_tokens": turn_result.usage.get("output_tokens", 0),
                "total_tokens": turn_result.usage.get("total_tokens", 0),
            },
        }

    async def chat_sse_generator() -> AsyncIterator[str]:
        first_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

        if turn_result.content:
            c_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": turn_result.content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n"

        if has_tools:
            tc_deltas = [
                {
                    "index": idx,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for idx, tc in enumerate(openai_tool_calls)
            ]
            tc_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

        done_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return chat_sse_generator()


# ---------------------------------------------------------------------------
# Pure Python stdlib Asyncio HTTP/1.1 + SSE Server (Zero external dependencies)
# ---------------------------------------------------------------------------
async def _read_http_request(
    reader: asyncio.StreamReader,
) -> Optional[tuple[str, str, dict[str, str], bytes]]:
    request_line = await reader.readline()
    if not request_line:
        return None
    parts = request_line.decode("latin-1").strip().split(" ")
    if len(parts) < 2:
        return None
    method, path = parts[0].upper(), parts[1].split("?")[0]

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("latin-1").strip()
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    body = b""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks: list[bytes] = []
        while True:
            size_line = await reader.readline()
            if not size_line:
                break
            chunk_len = int(size_line.strip().split(b";")[0], 16)
            if chunk_len < 0 or sum(len(chunk) for chunk in chunks) + chunk_len > MAX_REQUEST_BODY_BYTES:
                raise RequestTooLargeError(
                    f"Request body exceeds MAX_REQUEST_BODY_BYTES ({MAX_REQUEST_BODY_BYTES} bytes)"
                )
            if chunk_len == 0:
                await reader.readline()
                break
            chunks.append(await reader.readexactly(chunk_len))
            await reader.readline()
        body = b"".join(chunks)
    else:
        content_len = int(headers.get("content-length", "0"))
        if content_len > MAX_REQUEST_BODY_BYTES:
            raise RequestTooLargeError(
                f"Request body exceeds MAX_REQUEST_BODY_BYTES ({MAX_REQUEST_BODY_BYTES} bytes)"
            )
        if content_len > 0:
            body = await reader.readexactly(content_len)

    return method, path, headers, body


async def _send_json(
    writer: asyncio.StreamWriter, status: int, status_text: str, payload: dict[str, Any]
) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(data)}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(header + data)
    await writer.drain()


async def _send_sse_stream(
    writer: asyncio.StreamWriter, gen: AsyncIterator[str]
) -> None:
    header = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(header)
    await writer.drain()
    async for chunk in gen:
        writer.write(chunk.encode("utf-8"))
        await writer.drain()


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        req = await asyncio.wait_for(
            _read_http_request(reader), timeout=HTTP_READ_TIMEOUT_SECONDS
        )
        if req is None:
            return
        method, path, _headers, raw_body = req

        if method == "OPTIONS":
            writer.write(
                b"HTTP/1.1 204 No Content\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                b"Access-Control-Allow-Headers: *\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            return

        if method == "GET" and path in ("/health", "/v1/health"):
            await _send_json(writer, 200, "OK", get_health_payload())
            return

        if method == "GET" and path == "/v1/models":
            await _send_json(writer, 200, "OK", get_models_payload())
            return

        if method == "POST" and path in ("/v1/responses", "/responses"):
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception as e:
                await _send_json(writer, 400, "Bad Request", {"error": {"message": f"Invalid JSON: {e}"}})
                return

            try:
                result = await process_responses_api(body)
            except ValueError as e:
                await _send_json(
                    writer,
                    413,
                    "Payload Too Large",
                    {"error": {"message": str(e), "type": "context_length_exceeded"}},
                )
                return
            if isinstance(result, dict):
                await _send_json(writer, 200, "OK", result)
            else:
                await _send_sse_stream(writer, result)
            return

        if method == "POST" and path in ("/v1/chat/completions", "/chat/completions"):
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception as e:
                await _send_json(writer, 400, "Bad Request", {"error": {"message": f"Invalid JSON: {e}"}})
                return

            try:
                result = await process_chat_completions_api(body)
            except ValueError as e:
                await _send_json(
                    writer,
                    413,
                    "Payload Too Large",
                    {"error": {"message": str(e), "type": "context_length_exceeded"}},
                )
                return
            if isinstance(result, dict):
                await _send_json(writer, 200, "OK", result)
            else:
                await _send_sse_stream(writer, result)
            return

        await _send_json(writer, 404, "Not Found", {"error": {"message": f"Path not found: {path}"}})
    except RequestTooLargeError as e:
        try:
            await _send_json(
                writer,
                413,
                "Payload Too Large",
                {"error": {"message": str(e), "type": "request_too_large"}},
            )
        except Exception:
            pass
    except asyncio.TimeoutError:
        try:
            await _send_json(
                writer,
                408,
                "Request Timeout",
                {"error": {"message": "HTTP request read timed out", "type": "request_timeout"}},
            )
        except Exception:
            pass
    except Exception as e:
        try:
            await _send_json(
                writer,
                500,
                "Internal Server Error",
                {"error": {"message": str(e), "type": "server_error"}},
            )
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def run_server(host: str = HOST, port: int = PORT) -> None:
    server = await asyncio.start_server(handle_client, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"[AGY Wrapper Server] Listening on http://{addrs} (Responses API: /v1/responses)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n[AGY Wrapper Server] Shutting down.")
