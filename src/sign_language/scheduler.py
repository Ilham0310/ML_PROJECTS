"""Operational time-window enforcement for sign language detection."""

from __future__ import annotations

from datetime import datetime, time
from typing import Callable, Optional


class Scheduler:
    """Checks whether the system is inside the configured operating window.

    The default window is 18:00 inclusive through 22:00 exclusive local time.
    A ``now_provider`` can be injected in tests or for deterministic scheduling.
    """

    def __init__(
        self,
        start_hour: int = 18,
        end_hour: int = 22,
        now_provider: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not 0 <= start_hour <= 23:
            raise ValueError("start_hour must be in [0, 23]")
        if not 0 <= end_hour <= 23:
            raise ValueError("end_hour must be in [0, 23]")
        if start_hour == end_hour:
            raise ValueError("start_hour and end_hour must define a non-empty window")

        self.start_hour = start_hour
        self.end_hour = end_hour
        self._now_provider = now_provider or datetime.now

    def is_operational(self, at: Optional[datetime | time] = None) -> bool:
        """Return True iff ``at`` is inside the configured local-time window."""

        current = at if at is not None else self._now_provider()
        current_time = current.time() if isinstance(current, datetime) else current
        start = time(self.start_hour, 0)
        end = time(self.end_hour, 0)

        if self.start_hour < self.end_hour:
            return start <= current_time < end
        return current_time >= start or current_time < end

    @property
    def unavailable_message(self) -> str:
        """User-facing message shown when the system is outside operating hours."""

        return "System is not available. Operational hours are 6 PM to 10 PM."
