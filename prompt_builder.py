import json
import re
from typing import Any, Optional
from schemas import NormalizedToolCall


def normalize_input_items(raw_input: Any) -> list[dict[str, Any]]:
    """Normalize Responses API `input` (str or list) into a list of item dicts."""
    if isinstance(raw_input, str):
        return [{"type": "message", "role": "user", "content": raw_input}]
    if isinstance(raw_input, list):
        return [item for item in raw_input if isinstance(item, dict)]
    return []


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from string or Responses/ChatCompletions content array."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and part["text"] is not None:
                    parts.append(str(part["text"]))
                elif "content" in part and part["content"] is not None:
                    parts.append(str(part["content"]))
        return "\n".join(parts)
    return str(content)


def normalize_tools(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Normalize both Responses API flat tools and ChatCompletions nested tools."""
    if not tools:
        return []
    normalized: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            normalized.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        else:
            normalized.append(
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                }
            )
    return [t for t in normalized if t.get("name")]


def build_agy_prompt(
    instructions: Optional[str],
    input_items: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Any = None,
) -> str:
    """Construct the single-turn reasoning prompt for AGY CLI."""
    sections: list[str] = []

    sections.append(
        "### SYSTEM ROLE: HERMES BRAIN-ONLY BACKEND\n"
        "You are the stateless reasoning brain behind the Hermes runtime.\n"
        "- NEVER call any of your own built-in Antigravity tools (`view_file`, `run_command`, `list_dir`, etc.).\n"
        "- All tool execution happens externally inside Hermes.\n"
        "- You MUST reply with ONLY a single raw JSON object (no markdown fences, no extra text) in this exact format:\n"
        '{"content": "<string or null>", "tool_calls": [{"name": "<tool_name>", "arguments": {<json_object>}}]}'
    )

    if instructions:
        sections.append(f"### SYSTEM INSTRUCTIONS\n{instructions}")

    clean_tools = normalize_tools(tools)
    if clean_tools and tool_choice != "none":
        tools_json = json.dumps(clean_tools, ensure_ascii=False, indent=2)
        tool_policy = (
            "If you need to invoke one or more tools from `AVAILABLE_HERMES_TOOLS`, "
            "list them in `tool_calls` with their exact `name` and `arguments` object. "
            "If no tool call is needed, set `tool_calls` to `[]` and provide your response in `content`."
        )
        if tool_choice == "required":
            tool_policy += "\nIMPORTANT: `tool_choice` is `required`. You MUST return at least one tool call in `tool_calls`."
        elif isinstance(tool_choice, dict):
            forced_name = tool_choice.get("name") or (
                tool_choice.get("function", {}) if isinstance(tool_choice.get("function"), dict) else {}
            ).get("name")
            if forced_name:
                tool_policy += f"\nIMPORTANT: You MUST call the tool named `{forced_name}` in `tool_calls`."

        sections.append(f"### AVAILABLE_HERMES_TOOLS\n{tool_policy}\n{tools_json}")
    else:
        sections.append(
            "### TOOL POLICY\nNo external tools are enabled for this turn. Set `\"tool_calls\": []` and put your reply in `\"content\"`."
        )

    transcript_lines: list[str] = []
    for item in input_items:
        item_type = item.get("type", "message")
        role = item.get("role")

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id", "call_unknown")
            name = item.get("name", "")
            args = item.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            transcript_lines.append(
                f"[ASSISTANT_TOOL_CALL call_id={call_id} name={name}]\n{args}"
            )
        elif item_type == "function_call_output" or role == "tool":
            call_id = item.get("call_id") or item.get("tool_call_id", "call_unknown")
            output = item.get("output") if "output" in item else item.get("content", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            transcript_lines.append(
                f"[TOOL_OUTPUT call_id={call_id}]\n{output}"
            )
        else:
            text = extract_text_from_content(item.get("content", ""))
            if role in ("system", "developer"):
                sections.append(f"### ADDITIONAL SYSTEM INSTRUCTION\n{text}")
            elif role == "assistant":
                tc_list = item.get("tool_calls") or []
                if text:
                    transcript_lines.append(f"[ASSISTANT]\n{text}")
                for tc in tc_list:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    c_id = tc.get("id", "call_unknown")
                    c_name = fn.get("name") or tc.get("name", "")
                    c_args = fn.get("arguments") or tc.get("arguments", "{}")
                    transcript_lines.append(
                        f"[ASSISTANT_TOOL_CALL call_id={c_id} name={c_name}]\n{c_args}"
                    )
            else:
                transcript_lines.append(f"[USER]\n{text}")

    if transcript_lines:
        sections.append("### CONVERSATION HISTORY\n" + "\n\n".join(transcript_lines))
    else:
        sections.append("### CONVERSATION HISTORY\n[USER]\nHello.")

    sections.append(
        "### OUTPUT REQUIREMENT\n"
        "Respond now with ONLY the raw JSON object (`{\"content\": ..., \"tool_calls\": [...]}`)."
    )

    return "\n\n".join(sections)


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract and parse a JSON dict from model text (handles raw JSON or fenced blocks)."""
    stripped = text.strip()

    # 1. Try direct parse
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. Try ```json ... ``` fenced block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3. Scan for outermost balanced { ... } containing "tool_calls" or "content"
    start_idx = stripped.find("{")
    end_idx = stripped.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = stripped[start_idx : end_idx + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def parse_brain_output(raw_text: str) -> tuple[Optional[str], list[NormalizedToolCall]]:
    """Parse AGY's output into `(content, tool_calls)`."""
    obj = _extract_first_json_object(raw_text)
    if obj is None or ("content" not in obj and "tool_calls" not in obj):
        # Fallback: treat entire output as assistant text content
        return raw_text.strip(), []

    raw_content = obj.get("content")
    content: Optional[str] = str(raw_content) if raw_content is not None else None

    raw_tool_calls = obj.get("tool_calls") or []
    normalized_calls: list[NormalizedToolCall] = []

    if isinstance(raw_tool_calls, list):
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            # Support either {"name": ..., "arguments": ...} or {"function": {"name": ..., "arguments": ...}}
            fn_block = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = fn_block.get("name")
            if not name:
                continue

            raw_args = (
                fn_block.get("arguments")
                if "arguments" in fn_block
                else fn_block.get("arguments_json", {})
            )
            if isinstance(raw_args, dict):
                args_json_str = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                    if isinstance(parsed_args, dict):
                        args_json_str = json.dumps(parsed_args, ensure_ascii=False)
                    else:
                        args_json_str = json.dumps({"value": parsed_args}, ensure_ascii=False)
                except Exception:
                    args_json_str = json.dumps({"raw": raw_args}, ensure_ascii=False)
            else:
                args_json_str = "{}"

            normalized_calls.append(
                NormalizedToolCall(name=str(name), arguments_json=args_json_str)
            )

    return content, normalized_calls
