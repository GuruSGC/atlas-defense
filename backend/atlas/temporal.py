"""
Insertion history for the temporal trust signal.

The tracker remembers when recent candidate memories were submitted so that a burst of
insertions (a common way to flood memory with poisoned entries) lowers trust, while a
normal conversational pace does not. The clock can be injected, which keeps evaluation
runs deterministic and lets tests simulate bursts without sleeping.
"""

import time
from collections import deque
from typing import Deque, Optional

DEFAULT_WINDOW_SECONDS = 60.0


class TemporalTracker:
    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self._events: Deque[float] = deque()

    def record(self, timestamp: Optional[float] = None) -> None:
        self._events.append(time.time() if timestamp is None else timestamp)

    def recent_count(self, now: Optional[float] = None) -> int:
        """Number of recorded insertions inside the window ending at `now`."""
        now = time.time() if now is None else now
        cutoff = now - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        return sum(1 for t in self._events if t <= now)

    def reset(self) -> None:
        self._events.clear()
