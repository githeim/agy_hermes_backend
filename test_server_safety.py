"""AGY backend의 context와 request 안전 한도를 검증한다."""

import asyncio

import pytest

import server
from schemas import BrainTurnResult


async def _fake_runner(**_kwargs: object) -> BrainTurnResult:
    """외부 AGY 없이 API guard를 검증하기 위한 정상 응답을 반환한다."""
    return BrainTurnResult(content="ok")


def test_chat_completions_rejects_oversized_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat Completions도 Responses와 동일한 prompt 상한을 적용해야 한다."""
    monkeypatch.setattr(server, "run_agy_brain_turn", _fake_runner)
    monkeypatch.setattr(server, "AGY_MAX_PROMPT_CHARS", 1)

    with pytest.raises(ValueError, match="AGY_MAX_PROMPT_CHARS"):
        asyncio.run(
            server.process_chat_completions_api(
                {"messages": [{"role": "user", "content": "too large"}]}
            )
        )
