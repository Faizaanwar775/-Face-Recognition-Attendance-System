"""
Attendance event logging.

Given a confirmed (matched + live) employee, this module decides whether
the event is a clock-in or a clock-out (by toggling off the employee's
last recorded event) and persists it, with a cooldown to prevent the same
person triggering duplicate events while lingering in frame.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

from src.config import settings
from src.logging_events.logger import get_logger
from src.storage.database import AttendanceDatabase
from src.storage.models import AttendanceLogEntry, EventType

logger = get_logger(__name__)


class AttendanceLogger:
    """Decides event type and writes attendance events, with a cooldown."""

    def __init__(self, database: AttendanceDatabase) -> None:
        self._db = database
        self._lock = threading.Lock()
        # In-memory cache of last-logged time per employee to enforce the
        # cooldown without hitting the DB on every single frame.
        self._last_logged_at: dict[str, datetime] = {}

    def _next_event_type(self, employee_id: str) -> EventType:
        last_event = self._db.get_last_event(employee_id)
        if last_event is None or last_event.event_type == EventType.CLOCK_OUT.value:
            return EventType.CLOCK_IN
        return EventType.CLOCK_OUT

    def try_log_attendance(
        self, employee_id: str, confidence: float
    ) -> Optional[AttendanceLogEntry]:
        """Attempt to log an attendance event, respecting the cooldown window.

        Returns the created entry, or None if the event was suppressed
        because it fell within the cooldown period for that employee.
        """
        now = datetime.now()
        with self._lock:
            last_logged = self._last_logged_at.get(employee_id)
            if (
                last_logged is not None
                and (now - last_logged).total_seconds() < settings.log_cooldown_seconds
            ):
                return None

            event_type = self._next_event_type(employee_id)
            entry = AttendanceLogEntry(
                employee_id=employee_id,
                event_type=event_type.value,
                timestamp=now,
                confidence=confidence,
            )
            try:
                self._db.insert_log(entry)
            except Exception:  # noqa: BLE001 - a logging failure must not crash the app
                logger.exception("Failed to persist attendance log for '%s'", employee_id)
                return None

            self._last_logged_at[employee_id] = now
            return entry
