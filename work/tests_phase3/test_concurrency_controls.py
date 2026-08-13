from __future__ import annotations

import asyncio
import threading
import time

import pytest

from customer_service_core.concurrency import run_blocking_with_timeout


class BlockingProbe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.calls = 0
        self.max_active = 0

    def run(self, delay: float, value: str) -> str:
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(delay)
            return value
        finally:
            with self.lock:
                self.active -= 1


def test_timeout_retains_slot_until_worker_thread_finishes() -> None:
    async def scenario() -> None:
        semaphore = asyncio.Semaphore(1)
        probe = BlockingProbe()

        with pytest.raises(asyncio.TimeoutError):
            await run_blocking_with_timeout(
                semaphore,
                0.05,
                probe.run,
                0.25,
                "first",
            )
        with pytest.raises(asyncio.TimeoutError):
            await run_blocking_with_timeout(
                semaphore,
                0.05,
                probe.run,
                0.0,
                "must-not-start",
            )

        assert probe.calls == 1
        assert probe.max_active == 1
        await asyncio.sleep(0.25)
        assert await run_blocking_with_timeout(
            semaphore,
            0.2,
            probe.run,
            0.0,
            "recovered",
        ) == "recovered"
        assert probe.max_active == 1

    asyncio.run(scenario())


def test_cancellation_retains_slot_until_worker_thread_finishes() -> None:
    async def scenario() -> None:
        semaphore = asyncio.Semaphore(1)
        probe = BlockingProbe()
        request = asyncio.create_task(
            run_blocking_with_timeout(
                semaphore,
                1.0,
                probe.run,
                0.25,
                "cancelled-client",
            )
        )
        await asyncio.sleep(0.02)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(asyncio.TimeoutError):
            await run_blocking_with_timeout(
                semaphore,
                0.05,
                probe.run,
                0.0,
                "must-not-start",
            )

        assert probe.calls == 1
        await asyncio.sleep(0.25)
        assert await run_blocking_with_timeout(
            semaphore,
            0.2,
            probe.run,
            0.0,
            "recovered",
        ) == "recovered"

    asyncio.run(scenario())
