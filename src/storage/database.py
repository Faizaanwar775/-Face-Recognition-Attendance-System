"""
SQLite persistence layer.

Only two tables exist:
  * employees        -> employee_id, full_name, embedding (JSON list[float]), enrolled_at
  * attendance_logs  -> id, employee_id, event_type, timestamp, confidence

No table, column, or code path in this module ever stores raw image
bytes. Embeddings are stored as a JSON-encoded list of floats, which is
the only serialisation of a face that ever touches disk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from src.config import settings
from src.logging_events.logger import get_logger
from src.storage.models import AttendanceLogEntry, EmployeeRecord

logger = get_logger(__name__)


class DatabaseError(RuntimeError):
    """Raised when a storage operation fails."""


class AttendanceDatabase:
    """Thread-safe SQLite wrapper.

    SQLite connections are not safe to share across threads by default,
    so each thread gets its own connection via `threading.local`, all
    pointed at the same file, guarded by a re-entrant lock for writes.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or settings.db_path
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._initialize_schema()

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    @property
    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = self._connection
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("Database operation failed: %s", exc)
            raise DatabaseError(str(exc)) from exc
        finally:
            cur.close()

    def _initialize_schema(self) -> None:
        with self._write_lock, self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id  TEXT PRIMARY KEY,
                    full_name    TEXT NOT NULL,
                    embedding    TEXT NOT NULL,   -- JSON list[float], never an image
                    enrolled_at  TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id   TEXT NOT NULL,
                    event_type    TEXT NOT NULL CHECK (event_type IN ('clock_in', 'clock_out')),
                    timestamp     TEXT NOT NULL,
                    confidence    REAL NOT NULL,
                    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_employee_ts "
                "ON attendance_logs (employee_id, timestamp);"
            )
        logger.info("Database schema ready at %s", self._db_path)

    # ------------------------------------------------------------------ #
    # Employees
    # ------------------------------------------------------------------ #
    def upsert_employee(self, record: EmployeeRecord) -> None:
        """Insert or update an employee's stored embedding."""
        with self._write_lock, self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO employees (employee_id, full_name, embedding, enrolled_at)
                VALUES (:employee_id, :full_name, :embedding, :enrolled_at)
                ON CONFLICT(employee_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    embedding = excluded.embedding,
                    enrolled_at = excluded.enrolled_at;
                """,
                {
                    "employee_id": record.employee_id,
                    "full_name": record.full_name,
                    "embedding": json.dumps(record.embedding),
                    "enrolled_at": record.enrolled_at.isoformat(),
                },
            )
        logger.info("Enrolled/updated employee '%s' (%s)", record.employee_id, record.full_name)

    def get_all_employees(self) -> list[EmployeeRecord]:
        """Load every enrolled employee's embedding for matching."""
        with self._cursor() as cur:
            cur.execute("SELECT employee_id, full_name, embedding, enrolled_at FROM employees;")
            rows = cur.fetchall()

        records: list[EmployeeRecord] = []
        for row in rows:
            try:
                records.append(
                    EmployeeRecord(
                        employee_id=row["employee_id"],
                        full_name=row["full_name"],
                        embedding=json.loads(row["embedding"]),
                        enrolled_at=datetime.fromisoformat(row["enrolled_at"]),
                    )
                )
            except (ValueError, json.JSONDecodeError) as exc:
                # A corrupted row should not crash the whole matcher --
                # skip it and keep going, but make noise in the logs.
                logger.error("Skipping corrupted employee row '%s': %s", row["employee_id"], exc)
        return records

    def employee_exists(self, employee_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM employees WHERE employee_id = ?;", (employee_id,))
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    # Attendance logs
    # ------------------------------------------------------------------ #
    def insert_log(self, entry: AttendanceLogEntry) -> None:
        with self._write_lock, self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_logs (employee_id, event_type, timestamp, confidence)
                VALUES (:employee_id, :event_type, :timestamp, :confidence);
                """,
                {
                    "employee_id": entry.employee_id,
                    "event_type": entry.event_type,
                    "timestamp": entry.timestamp.isoformat(),
                    "confidence": entry.confidence,
                },
            )
        logger.info(
            "Logged %s for employee '%s' (confidence=%.3f)",
            entry.event_type,
            entry.employee_id,
            entry.confidence,
        )

    def get_last_event(self, employee_id: str) -> Optional[AttendanceLogEntry]:
        """Return the most recent attendance event for an employee, if any."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT employee_id, event_type, timestamp, confidence
                FROM attendance_logs
                WHERE employee_id = ?
                ORDER BY timestamp DESC
                LIMIT 1;
                """,
                (employee_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return AttendanceLogEntry(
            employee_id=row["employee_id"],
            event_type=row["event_type"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            confidence=row["confidence"],
        )

    def get_logs_for_employee(self, employee_id: str, limit: int = 50) -> list[AttendanceLogEntry]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT employee_id, event_type, timestamp, confidence
                FROM attendance_logs WHERE employee_id = ?
                ORDER BY timestamp DESC LIMIT ?;
                """,
                (employee_id, limit),
            )
            rows = cur.fetchall()
        return [
            AttendanceLogEntry(
                employee_id=row["employee_id"],
                event_type=row["event_type"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                confidence=row["confidence"],
            )
            for row in rows
        ]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
