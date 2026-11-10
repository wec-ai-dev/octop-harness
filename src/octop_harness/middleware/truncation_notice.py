"""Tell the model when its reply hit the output limit.

A reply cut at the token limit is the most damaging failure a tool-using turn
can have: a ``write_file`` or ``edit_file`` call arrives as incomplete JSON, its
arguments cannot be parsed, and the model is left guessing why. The provider
reports the cut in the response metadata — ``finish_reason: "length"`` for the
OpenAI protocol, ``stop_reason: "max_tokens"`` for Anthropic — and nothing in
the graph reads it.

This middleware turns that marker into two things: a warning in the log, and a
note appended to the reply so the next model call knows the content was cut and
can split the work instead of retrying the same oversized call.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Response-metadata keys that mean "the provider stopped for the output limit".
_TRUNCATED_VALUES: dict[str, frozenset[str]] = {
    "finish_reason": frozenset({"length"}),
    "stop_reason": frozenset({"max_tokens"}),
}
MARKER_KEY = "octop_output_truncated"

NOTE = (
    "\n\n[Output truncated] The reply above was cut off at the model's output-token limit, "
    "so it is incomplete. It was not executed. Split the content into several smaller tool "
    "calls (write the file in parts, then append) instead of repeating the same call."
)


def is_truncated(message: Any) -> bool:
    """Whether the provider stopped *message* for the output limit."""
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return False
    return any(
        str(metadata.get(key) or "").lower() in values for key, values in _TRUNCATED_VALUES.items()
    )


def _with_notice(message: AIMessage) -> AIMessage:
    content = message.content
    if isinstance(content, list):
        updated: Any = [*content, {"type": "text", "text": NOTE}]
    else:
        updated = f"{content}{NOTE}"
    extra = dict(message.additional_kwargs or {})
    extra[MARKER_KEY] = True
    return message.model_copy(update={"content": updated, "additional_kwargs": extra})


class OutputTruncationMiddleware(AgentMiddleware[Any, Any]):
    """Log a truncated reply and tell the model to split its next attempt."""

    @staticmethod
    def _annotate(response: ModelResponse[Any]) -> ModelResponse[Any]:
        truncated = [
            index
            for index, message in enumerate(response.result)
            if isinstance(message, AIMessage) and is_truncated(message)
        ]
        if not truncated:
            return response
        for index in truncated:
            message = response.result[index]
            has_tool_calls = bool(getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None))
            logger.warning(
                "model reply truncated at the output-token limit (tool_calls=%s)",
                has_tool_calls,
            )
            response.result[index] = _with_notice(message)  # type: ignore[arg-type]
        return response

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return self._annotate(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return self._annotate(await handler(request))


__all__ = [
    "MARKER_KEY",
    "NOTE",
    "OutputTruncationMiddleware",
    "is_truncated",
]
