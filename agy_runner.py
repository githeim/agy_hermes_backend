import asyncio
import json
from typing import Optional

from config import (
    AGY_BIN,
    AGY_RUNTIME_CWD,
    AGY_TIMEOUT_SECONDS,
    DEFAULT_AGY_EFFORT,
    DEFAULT_AGY_MODEL,
)
from prompt_builder import parse_brain_output
from schemas import BrainTurnResult


# Valid AGY model IDs discovered via `agy models`
VALID_AGY_MODELS = {
    "gemini-4-argon-high",
    "gemini-4-argon-medium",
    "gemini-4-argon-low",
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "gemini-3.7-flash-high",
    "gemini-3.7-flash-medium",
    "gemini-3.7-flash-low",
    "gemini-3.6-flash-high",
    "gemini-3.6-flash-medium",
    "gemini-3.6-flash-low",
    "gemini-3.1-pro-low",
    "gemini-3.1-pro-high",
}


def resolve_agy_model(requested_model: Optional[str]) -> Optional[str]:
    """Return a valid AGY model flag if requested_model matches one, else DEFAULT_AGY_MODEL."""
    if requested_model and requested_model in VALID_AGY_MODELS:
        return requested_model
    if DEFAULT_AGY_MODEL:
        return DEFAULT_AGY_MODEL
    return None


async def run_agy_brain_turn(
    prompt: str,
    requested_model: Optional[str] = None,
    effort: Optional[str] = None,
) -> BrainTurnResult:
    """
    Execute a single stateless reasoning turn using `agy` in brain-only mode
    via stdin (`--input-format stream-json`) so prompts of any size bypass OS argv limits.
    """
    cmd: list[str] = [
        AGY_BIN,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
        # The wrapper is a decision-only backend; Hermes executes tool calls.
        "--settings",
        '{"permissions":{"allow":[],"deny":["*"]}}',
    ]

    agy_model = resolve_agy_model(requested_model)
    # Versioned model aliases already encode low/medium/high; AGY rejects a
    # contradictory separate --effort for those aliases.
    if effort and not (agy_model and agy_model.endswith(("-low", "-medium", "-high"))):
        cmd.extend(["--effort", effort])
    elif not agy_model and DEFAULT_AGY_EFFORT:
        cmd.extend(["--effort", DEFAULT_AGY_EFFORT])

    if agy_model:
        cmd.extend(["--model", agy_model])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(AGY_RUNTIME_CWD),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdin_payload = (
        json.dumps(
            {"event": "user", "message": {"content": prompt}},
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin_payload),
            timeout=AGY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"AGY CLI timed out after {AGY_TIMEOUT_SECONDS}s")
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            try:
                await asyncio.shield(proc.communicate())
            except BaseException:
                pass
        raise

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    final_response_text = ""
    delta_chunks: list[str] = []
    usage_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    conversation_id: Optional[str] = None

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue

        event_type = evt.get("event")
        if event_type == "init":
            conversation_id = evt.get("conversation_id")
        elif event_type == "step_update":
            su = evt.get("step_update", {})
            if su.get("step_type") == "agent_response" and su.get("text_delta"):
                delta_chunks.append(su["text_delta"])
        elif event_type == "result":
            res = evt.get("result", {})
            conversation_id = res.get("conversation_id") or conversation_id
            if res.get("status") == "ERROR":
                err_msg = res.get("error") or stderr_text or "Unknown AGY error"
                raise RuntimeError(f"AGY execution error: {err_msg}")
            final_response_text = res.get("response") or ""
            raw_usage = res.get("usage") or {}
            usage_info = {
                "input_tokens": int(raw_usage.get("input_tokens", 0)),
                "output_tokens": int(raw_usage.get("output_tokens", 0)),
                "total_tokens": int(raw_usage.get("total_tokens", 0)),
            }

    if not final_response_text and delta_chunks:
        final_response_text = "".join(delta_chunks)

    if proc.returncode != 0 and not final_response_text:
        raise RuntimeError(
            f"AGY exited with code {proc.returncode}: {stderr_text.strip() or stdout_text.strip()}"
        )

    content, tool_calls = parse_brain_output(final_response_text)

    return BrainTurnResult(
        content=content,
        tool_calls=tool_calls,
        usage=usage_info,
        conversation_id=conversation_id,
    )
