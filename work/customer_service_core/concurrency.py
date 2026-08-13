from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar


LOGGER = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


def _consume_background_result(
    task: asyncio.Task[Any],
    *,
    log_failure: bool,
) -> None:
    try:
        exception = task.exception()
    except asyncio.CancelledError:
        return
    if exception is not None and log_failure:
        LOGGER.error(
            "blocking request task failed after its client response ended",
            exc_info=(type(exception), exception, exception.__traceback__),
        )


def release_slot_when_task_finishes(
    semaphore: asyncio.Semaphore,
    task: asyncio.Task[Any],
) -> None:
    """Release a slot only after its non-cancellable worker thread is finished."""

    def release(completed: asyncio.Task[Any]) -> None:
        _consume_background_result(completed, log_failure=True)
        semaphore.release()

    if task.done():
        _consume_background_result(task, log_failure=False)
        semaphore.release()
    else:
        task.add_done_callback(release)


async def acquire_slot_before(
    semaphore: asyncio.Semaphore,
    deadline: float,
) -> None:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    await asyncio.wait_for(semaphore.acquire(), timeout=remaining)


async def run_blocking_with_timeout(
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
    function: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    """Run blocking work without releasing capacity while its thread is alive."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_seconds)
    await acquire_slot_before(semaphore, deadline)
    remaining = deadline - loop.time()
    if remaining <= 0:
        semaphore.release()
        raise asyncio.TimeoutError
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    finally:
        release_slot_when_task_finishes(semaphore, task)
