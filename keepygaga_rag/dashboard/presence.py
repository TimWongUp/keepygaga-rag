from __future__ import annotations

import asyncio
from collections.abc import Callable


class DashboardPresence:
    def __init__(
        self,
        *,
        shutdown: Callable[[], None],
        grace_seconds: float = 10.0,
    ):
        self.shutdown = shutdown
        self.grace_seconds = grace_seconds
        self.clients: set[int] = set()
        self.armed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def connected(self, client_id: int) -> None:
        self.clients.add(client_id)
        self.armed = True
        self._cancel_shutdown()

    def disconnected(self, client_id: int) -> None:
        self.clients.discard(client_id)
        if self.armed and not self.clients:
            self._cancel_shutdown()
            self._shutdown_task = asyncio.create_task(self._shutdown_after_grace())

    async def close(self) -> None:
        self._cancel_shutdown()
        await asyncio.sleep(0)

    def _cancel_shutdown(self) -> None:
        if self._shutdown_task is not None:
            self._shutdown_task.cancel()
            self._shutdown_task = None

    async def _shutdown_after_grace(self) -> None:
        task = asyncio.current_task()
        try:
            await asyncio.sleep(self.grace_seconds)
            if self.armed and not self.clients:
                self.shutdown()
        except asyncio.CancelledError:
            pass
        finally:
            if self._shutdown_task is task:
                self._shutdown_task = None
