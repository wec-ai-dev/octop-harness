"""Summary-input cap and failure fallback for context compaction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.exceptions import ContextOverflowError

from octop_harness.middleware.summarization_limits import (
    SUMMARY_FALLBACK_TEXT,
    SUMMARY_INPUT_RETRY_TOKEN_LIMIT,
    SUMMARY_INPUT_TOKEN_LIMIT,
    harden_summarization_middleware,
)


def _middleware(
    *,
    trim: int | None,
    sync_error: Exception | None = None,
    errors: list[Exception] | None = None,
    seen: list[int | None] | None = None,
) -> SimpleNamespace:
    helper = SimpleNamespace(trim_tokens_to_summarize=trim)
    pending = list(errors or [])

    def _fail() -> None:
        if sync_error is not None:
            raise sync_error
        if pending:
            raise pending.pop(0)

    def _create(_messages: object) -> str:
        if seen is not None:
            seen.append(helper.trim_tokens_to_summarize)
        _fail()
        return "summary"

    async def _acreate(_messages: object) -> str:
        if seen is not None:
            seen.append(helper.trim_tokens_to_summarize)
        _fail()
        return "summary"

    return SimpleNamespace(_lc_helper=helper, _create_summary=_create, _acreate_summary=_acreate)


def test_caps_unset_summary_input() -> None:
    mw = _middleware(trim=None)
    harden_summarization_middleware(mw)
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT
    assert mw._create_summary([]) == "summary"


def test_keeps_explicit_summary_input_limit() -> None:
    mw = _middleware(trim=2000)
    harden_summarization_middleware(mw)
    assert mw._lc_helper.trim_tokens_to_summarize == 2000


def test_summary_failure_retries_then_uses_placeholder() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=None, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert mw._create_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


def test_summary_failure_skips_retry_when_window_already_small() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=2000, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert mw._create_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [2000]


@pytest.mark.asyncio
async def test_async_summary_failure_uses_placeholder() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=None, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert await mw._acreate_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]


def test_rate_limit_failure_preserves_the_thread() -> None:
    mw = _middleware(trim=None, sync_error=RuntimeError("HTTP 429"))
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        mw._create_summary(["history"])


def test_rate_limit_after_overflow_preserves_the_thread() -> None:
    seen: list[int | None] = []
    mw = _middleware(
        trim=None,
        errors=[ContextOverflowError("prompt is too long"), RuntimeError("HTTP 429")],
        seen=seen,
    )
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        mw._create_summary(["history"])
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


@pytest.mark.asyncio
async def test_async_rate_limit_after_overflow_preserves_the_thread() -> None:
    mw = _middleware(
        trim=None,
        errors=[ContextOverflowError("prompt is too long"), RuntimeError("HTTP 429")],
    )
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        await mw._acreate_summary(["history"])
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


# --- Keeping the turn's task message in the summary input -------------------


def _langchain_helper(limit: int | None) -> object:
    """A real deepagents summarization helper — the object the patch wraps."""
    from deepagents.middleware.summarization import (
        SummarizationMiddleware as DeepAgentsSummarizationMiddleware,
    )
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    model = GenericFakeChatModel(messages=iter([]))
    return DeepAgentsSummarizationMiddleware(
        model=model, backend=None, trim_tokens_to_summarize=limit
    )._lc_helper


def _single_turn_history(rounds: int = 25) -> list[object]:
    """One user message, then that many shell calls with ~2.4k-character results."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages: list[object] = [
        HumanMessage(
            content=(
                "Reconcile the invoice file at /workspace/invoices.xlsx against the ledger. "
                "Write the result to /workspace/recon.xlsx and do not modify the input file."
            )
        )
    ]
    for index in range(rounds):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": f"python step{index}.py"},
                        "id": f"c{index}",
                    }
                ],
            )
        )
        messages.append(
            ToolMessage(content="row,amount,diff\n" + "x,1,2\n" * 400, tool_call_id=f"c{index}")
        )
    return messages


def test_summary_input_keeps_the_task_message() -> None:
    helper = _langchain_helper(8000)
    mw = SimpleNamespace(_lc_helper=helper)
    harden_summarization_middleware(mw)

    trimmed = helper._trim_messages_for_summary(_single_turn_history())

    # The goal, the constraint and the output path must survive compaction.
    assert trimmed
    assert any(
        "Reconcile the invoice file" in str(getattr(message, "content", "")) for message in trimmed
    )
    assert "do not modify the input file" in str(trimmed[0].content)


def test_summary_input_trims_an_oversized_task_message() -> None:
    from langchain_core.messages import HumanMessage

    helper = _langchain_helper(200)
    mw = SimpleNamespace(_lc_helper=helper)
    harden_summarization_middleware(mw)

    messages = [HumanMessage(content="task " * 2000), *_single_turn_history(3)[1:]]
    trimmed = helper._trim_messages_for_summary(messages)

    assert len(trimmed) == 1
    assert helper.token_counter(trimmed) <= 200
    assert str(trimmed[0].content).startswith("task")


def test_summary_input_without_a_user_message_keeps_the_tail() -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    helper = _langchain_helper(8000)
    mw = SimpleNamespace(_lc_helper=helper)
    harden_summarization_middleware(mw)

    messages = [
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {}, "id": "c1"}]),
        ToolMessage(content="output", tool_call_id="c1"),
    ]
    trimmed = helper._trim_messages_for_summary(messages)

    assert [type(message).__name__ for message in trimmed] == ["AIMessage", "ToolMessage"]
