import asyncio
import json
import urllib.error
import urllib.request

import server
from server import (
    RESPONSE_STATE_STORE,
    _merge_response_input,
    _store_response_state,
    handle_client,
)


def run_state_store_unit_tests() -> None:
    """AGY 호출 없이 히스토리 중복 제거와 LRU 상한을 검증한다."""
    old_limit = server.RESPONSE_STATE_MAX_ENTRIES
    try:
        prior = [
            {"type": "message", "role": "user", "content": "첫 질문"},
            {"type": "message", "role": "assistant", "content": "첫 답변"},
        ]
        resent_transcript = prior + [
            {"type": "message", "role": "user", "content": "두 번째 질문"},
        ]
        assert _merge_response_input(prior, resent_transcript) == resent_transcript

        tool_result = [
            {"type": "function_call_output", "call_id": "call_1", "output": "완료"}
        ]
        assert _merge_response_input(prior, tool_result) == prior + tool_result

        RESPONSE_STATE_STORE.clear()
        server.RESPONSE_STATE_MAX_ENTRIES = 2
        _store_response_state("resp_1", prior)
        _store_response_state("resp_2", prior)
        _store_response_state("resp_3", prior)
        assert list(RESPONSE_STATE_STORE) == ["resp_2", "resp_3"]
        print("State overlap and LRU tests passed")
    finally:
        RESPONSE_STATE_STORE.clear()
        server.RESPONSE_STATE_MAX_ENTRIES = old_limit


async def http_request(port: int, method: str, path: str, payload: dict | None = None) -> tuple[int, str]:
    def _sync_call() -> tuple[int, str]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    return await asyncio.to_thread(_sync_call)


async def run_verification() -> None:
    run_state_store_unit_tests()

    # Start the server on an ephemeral local port
    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    print(f"Started test server on http://127.0.0.1:{port}")

    async with server:
        print("\n=== 1. Health & Models Check ===")
        status, body = await http_request(port, "GET", "/health")
        assert status == 200, body
        print("Health:", body)

        status, body = await http_request(port, "GET", "/v1/models")
        assert status == 200, body
        models_data = json.loads(body)
        print("Models count:", len(models_data["data"]))

        print("\n=== 2. Responses API: Tool Calling Turn (Turn 1) ===")
        hermes_tools = [
            {
                "type": "function",
                "name": "terminal",
                "description": "Execute a shell command in the Hermes sandbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"}
                    },
                    "required": ["command"],
                },
            },
            {
                "type": "function",
                "name": "clarify",
                "description": "Ask the user a clarifying question before proceeding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to ask the user"}
                    },
                    "required": ["question"],
                },
            },
        ]

        turn1_payload = {
            "model": "agy-hermes-backend",
            "instructions": "You are Hermes. Use the `terminal` tool when asked to check system files.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Check the files in /tmp using the terminal tool (`ls -la /tmp`).",
                }
            ],
            "tools": hermes_tools,
            "tool_choice": "required",
            "stream": False,
        }

        status, body = await http_request(port, "POST", "/v1/responses", turn1_payload)
        assert status == 200, f"HTTP {status}: {body}"
        turn1_data = json.loads(body)
        print("Turn 1 Response:", json.dumps(turn1_data, indent=2, ensure_ascii=False))

        fc_items = [item for item in turn1_data["output"] if item["type"] == "function_call"]
        assert len(fc_items) >= 1, "Expected at least one function_call item!"
        fc = fc_items[0]
        assert fc["name"] == "terminal", f"Expected terminal tool, got {fc['name']}"
        parsed_args = json.loads(fc["arguments"])
        print(f"Verified function_call -> name={fc['name']}, call_id={fc['call_id']}, args={parsed_args}")

        print("\n=== 3. Responses API: Function Call Output -> Final Answer (Turn 2 via previous_response_id) ===")
        turn2_payload = {
            "model": "agy-hermes-backend",
            "previous_response_id": turn1_data["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": fc["call_id"],
                    "output": "total 0\ndrwxrwxrwt 2 root root 40 Nov 3 05:00 .\n-rw-r--r-- 1 ubuntu ubuntu 12 Nov 3 05:00 hello.txt",
                }
            ],
            "tools": hermes_tools,
            "stream": False,
        }

        status, body = await http_request(port, "POST", "/v1/responses", turn2_payload)
        assert status == 200, f"HTTP {status}: {body}"
        turn2_data = json.loads(body)
        print("Turn 2 Response:", json.dumps(turn2_data, indent=2, ensure_ascii=False))

        msg_items = [item for item in turn2_data["output"] if item["type"] == "message"]
        assert len(msg_items) >= 1, "Expected assistant message in Turn 2!"
        print("Final Assistant Text:", msg_items[0]["content"][0]["text"])

        print("\n=== 4. Responses API: Streaming SSE (`stream: true`) with Question Tool (`clarify`) ===")
        stream_payload = {
            "model": "agy-hermes-backend",
            "instructions": "When requirements are ambiguous, call the `clarify` tool to ask the user.",
            "input": "Deploy the project now.",
            "tools": hermes_tools,
            "tool_choice": {"type": "function", "name": "clarify"},
            "stream": True,
        }

        status, sse_body = await http_request(port, "POST", "/v1/responses", stream_payload)
        assert status == 200, f"HTTP {status}: {sse_body}"
        events = [
            line.replace("event: ", "").strip()
            for line in sse_body.splitlines()
            if line.startswith("event: ")
        ]
        print("Emitted SSE Event Sequence:", events)
        assert "response.created" in events
        assert "response.function_call_arguments.delta" in events
        assert "response.function_call_arguments.done" in events
        assert "response.completed" in events

        print("\nALL VERIFICATION TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_verification())
