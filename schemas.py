from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizedToolCall:
    name: str
    arguments_json: str = "{}"


@dataclass
class BrainTurnResult:
    content: Optional[str] = None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    conversation_id: Optional[str] = None
