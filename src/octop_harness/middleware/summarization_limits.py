"""Keep context compaction from resending an oversized history to the model.

deepagents' ``create_summarization_middleware`` leaves
``trim_tokens_to_summarize`` unset, so the summary call includes every evicted
message. Cap the summary input, and only degrade to a placeholder after a
provider explicitly reports that the reduced summary still exceeds its context
window. Other failures (including ordinary rate limits) leave the thread intact.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.exceptions import ContextOverflowError

logger = logging.getLogger(__name__)

SUMMARY_INPUT_TOKEN_LIMIT = 8000
SUMMARY_INPUT_RETRY_TOKEN_LIMIT = 4000
SUMMARY_FALLBACK_TEXT = (
    "The earlier conversation was too large to summarize. "
    "Older messages were saved to the history file. "
    "Continue from the recent messages and that file; do not repeat finished tool calls."
)


def harden_summarization_middleware(mw: Any) -> Any:
    """Cap summary input, keep the task goal, and degrade only on overflow."""
    helper = getattr(mw, "_lc_helper", None)
    if helper is not None and getattr(helper, "trim_tokens_to_summarize", None) is None:
        helper.trim_tokens_to_summarize = SUMMARY_INPUT_TOKEN_LIMIT
    if helper is not None:
        _patch_summary_trim(helper)
    if helper is None or getattr(mw, "_octop_summary_fallback", False):
        return mw
    if not callable(getattr(mw, "_create_summary", None)) or not callable(getattr(mw, "_acreate_summary", None)):
        return mw

    mw._octop_summary_fallback = True
    orig_sync = mw._create_summary
    orig_async = mw._acreate_summary
    # One graph uses either the sync or the async summary path. Each lock keeps
    # the temporary smaller window from leaking into another call on that path.
    sync_lock = threading.Lock()
    async_lock = asyncio.Lock()

    def _create_summary(messages: Any) -> str:
        with sync_lock:
            return _summary_or_fallback(helper, orig_sync, messages)

    async def _acreate_summary(messages: Any) -> str:
        async with async_lock:
            return await _asummary_or_fallback(helper, orig_async, messages)

    mw._create_summary = _create_summary
    mw._acreate_summary = _acreate_summary
    return mw


def _patch_summary_trim(helper: Any) -> None:
    """Keep the turn's task message in the summary input.

    LangChain trims the summary input to the most recent messages and makes the
    window start on a user message. In a single-turn task the only user message
    is the first one, so once the evicted part is over budget the trim comes out
    empty and the summary degrades to a placeholder — the goal, the constraints
    and the output location are lost. Put that message in first, then spend what
    is left of the budget on the most recent messages.
    """
    if getattr(helper, "_octop_goal_trim", False):
        return
    original = getattr(helper, "_trim_messages_for_summary", None)
    if not callable(original):
        return
    helper._octop_goal_trim = True

    def _trim(messages: Any) -> Any:
        limit = getattr(helper, "trim_tokens_to_summarize", None)
        if not isinstance(limit, int) or limit <= 0:
            return original(messages)
        return _goal_first_trim(
            messages,
            limit=limit,
            token_counter=getattr(helper, "token_counter", None),
        )

    helper._trim_messages_for_summary = _trim


def _goal_first_trim(messages: Any, *, limit: int, token_counter: Any) -> Any:
    """The turn's first user message, then the newest messages that still fit."""
    from langchain_core.messages import HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately, trim_messages

    if token_counter is None:
        token_counter = count_tokens_approximately
    items = list(messages)
    goal_at = next((index for index, msg in enumerate(items) if isinstance(msg, HumanMessage)), None)
    if goal_at is None:
        # Nothing identifies the task; keep the newest messages as before.
        return trim_messages(
            items,
            max_tokens=limit,
            token_counter=token_counter,
            strategy="last",
            allow_partial=True,
            include_system=True,
        )

    goal = items[goal_at]
    goal_tokens = token_counter([goal])
    if goal_tokens >= limit:
        # An oversized task message: keep its beginning, which holds the ask.
        head = trim_messages(
            [goal],
            max_tokens=limit,
            token_counter=token_counter,
            strategy="first",
            allow_partial=True,
        )
        return head or _clip_message(goal, limit=limit, token_counter=token_counter)
    rest = items[:goal_at] + items[goal_at + 1 :]
    tail = trim_messages(
        rest,
        max_tokens=limit - goal_tokens,
        token_counter=token_counter,
        strategy="last",
        allow_partial=True,
        include_system=True,
    )
    # The goal comes first: it is the instruction the summary must preserve, and
    # the tail never starts on a user message, so the upstream ``start_on``
    # requirement is dropped.
    return [goal, *tail]


def _clip_message(message: Any, *, limit: int, token_counter: Any) -> list[Any]:
    """Keep the head of one oversized message that the splitter cannot cut.

    ``trim_messages`` splits message text on newlines. A task written as one
    long paragraph has no split point, so it is dropped whole; cut it by
    characters instead and keep the opening, where the ask is stated.
    """
    text = message.content if isinstance(message.content, str) else str(message.content)
    low, high = 0, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = message.model_copy(update={"content": text[:middle]})
        if token_counter([candidate]) <= limit:
            best = text[:middle]
            low = middle + 1
        else:
            high = middle - 1
    return [message.model_copy(update={"content": best})] if best else []


def _summary_or_fallback(helper: Any, call: Callable[[Any], str], messages: Any) -> str:
    try:
        return call(messages)
    except Exception as exc:
        if not _is_context_overflow(exc):
            raise
        logger.warning("conversation summary failed (%s); retrying with a smaller window", type(exc).__name__)
        if not _use_smaller_window(helper):
            return SUMMARY_FALLBACK_TEXT
        previous = helper.trim_tokens_to_summarize
        helper.trim_tokens_to_summarize = SUMMARY_INPUT_RETRY_TOKEN_LIMIT
        try:
            return call(messages)
        except Exception as retry_exc:
            return _placeholder_after_overflow(retry_exc)
        finally:
            helper.trim_tokens_to_summarize = previous


async def _asummary_or_fallback(
    helper: Any,
    call: Callable[[Any], Awaitable[str]],
    messages: Any,
) -> str:
    try:
        return await call(messages)
    except Exception as exc:
        if not _is_context_overflow(exc):
            raise
        logger.warning("conversation summary failed (%s); retrying with a smaller window", type(exc).__name__)
        if not _use_smaller_window(helper):
            return SUMMARY_FALLBACK_TEXT
        previous = helper.trim_tokens_to_summarize
        helper.trim_tokens_to_summarize = SUMMARY_INPUT_RETRY_TOKEN_LIMIT
        try:
            return await call(messages)
        except Exception as retry_exc:
            return _placeholder_after_overflow(retry_exc)
        finally:
            helper.trim_tokens_to_summarize = previous


def _placeholder_after_overflow(exc: Exception) -> str:
    """Use the placeholder only when the smaller window still overflows."""
    if not _is_context_overflow(exc):
        raise exc
    logger.warning(
        "conversation summary failed after a smaller window (%s); using a placeholder",
        type(exc).__name__,
    )
    return SUMMARY_FALLBACK_TEXT


def _use_smaller_window(helper: Any) -> bool:
    current = getattr(helper, "trim_tokens_to_summarize", None)
    return not isinstance(current, int) or current > SUMMARY_INPUT_RETRY_TOKEN_LIMIT


def _is_context_overflow(exc: Exception) -> bool:
    """Match provider context-limit errors without treating ordinary 429s as data loss."""
    if isinstance(exc, ContextOverflowError):
        return True
    status = getattr(exc, "status_code", None)
    if status not in {400, 413, 422}:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "exceeds the context window",
            "exceeds the available context size",
            "context window exceeded",
            "context limit exceeded",
            "input tokens exceed",
            "prompt is too long",
        )
    )
