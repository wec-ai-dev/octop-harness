"""Truncated replies are logged and tell the model to split the work."""

from __future__ import annotations

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage

from octop_harness.middleware.truncation_notice import (
    MARKER_KEY,
    NOTE,
    OutputTruncationMiddleware,
    is_truncated,
)


def _response(message: AIMessage) -> ModelResponse:
    return ModelResponse(result=[message])


def test_openai_length_stop_is_recognised() -> None:
    assert is_truncated(AIMessage(content="", response_metadata={"finish_reason": "length"}))


def test_anthropic_max_tokens_stop_is_recognised() -> None:
    assert is_truncated(AIMessage(content="", response_metadata={"stop_reason": "max_tokens"}))


def test_a_normal_stop_is_not_truncation() -> None:
    assert not is_truncated(AIMessage(content="", response_metadata={"finish_reason": "stop"}))
    assert not is_truncated(AIMessage(content=""))


def test_a_truncated_tool_call_gets_the_split_instruction() -> None:
    message = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "x"}, "id": "c1"}],
        response_metadata={"finish_reason": "length"},
    )
    middleware = OutputTruncationMiddleware()

    result = middleware.wrap_model_call(None, lambda _request: _response(message))

    content = result.result[0].content
    assert isinstance(content, str)
    assert content.endswith(NOTE)
    assert result.result[0].additional_kwargs[MARKER_KEY] is True
    # The call itself is untouched: the host decides what to do with it.
    assert result.result[0].tool_calls


def test_an_untruncated_reply_is_left_alone() -> None:
    message = AIMessage(content="done", response_metadata={"finish_reason": "stop"})
    middleware = OutputTruncationMiddleware()

    result = middleware.wrap_model_call(None, lambda _request: _response(message))

    assert result.result[0] is message
    assert MARKER_KEY not in result.result[0].additional_kwargs


async def test_async_path_annotates_the_same_way() -> None:
    message = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    middleware = OutputTruncationMiddleware()

    async def handler(_request: object) -> ModelResponse:
        return _response(message)

    result = await middleware.awrap_model_call(None, handler)

    assert str(result.result[0].content).endswith(NOTE)
