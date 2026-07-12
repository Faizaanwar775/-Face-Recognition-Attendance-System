#!/usr/bin/env python3
"""
Khizex Face Recognition Attendance System - CLI entry point.

Usage:
    python main.py enroll --id E001 --name "Jane Doe"
    python main.py run
    python main.py list-employees
    python main.py logs --id E001
"""

from __future__ import annotations

import argparse
import sys

from src.core.enrollment import EnrollmentError, EnrollmentService
from src.core.pipeline import AttendancePipeline
from src.logging_events.logger import get_logger
from src.storage.database import AttendanceDatabase
from src.storage.models import EnrollmentRequest

logger = get_logger("main")


def cmd_enroll(args: argparse.Namespace) -> int:
    db = AttendanceDatabase()
    try:
        request = EnrollmentRequest(employee_id=args.id, full_name=args.name)
    except Exception as exc:  # noqa: BLE001 - surface validation errors cleanly
        print(f"Invalid enrollment request: {exc}", file=sys.stderr)
        return 1

    service = EnrollmentService(db)
    try:
        record = service.enroll_interactive(request, show_preview=not args.no_preview)
    except EnrollmentError as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(f"Enrolled '{record.employee_id}' ({record.full_name}) successfully.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pipeline = AttendancePipeline()
    try:
        pipeline.run_display_loop()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_list_employees(args: argparse.Namespace) -> int:
    db = AttendanceDatabase()
    try:
        employees = db.get_all_employees()
    finally:
        db.close()

    if not employees:
        print("No employees enrolled yet.")
        return 0

    print(f"{'Employee ID':<15}{'Full Name':<30}{'Enrolled At'}")
    print("-" * 70)
    for emp in employees:
        print(f"{emp.employee_id:<15}{emp.full_name:<30}{emp.enrolled_at.isoformat()}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    db = AttendanceDatabase()
    try:
        logs = db.get_logs_for_employee(args.id, limit=args.limit)
    finally:
        db.close()

    if not logs:
        print(f"No attendance logs found for '{args.id}'.")
        return 0

    print(f"{'Timestamp':<26}{'Event':<12}{'Confidence'}")
    print("-" * 55)
    for entry in logs:
        print(f"{entry.timestamp.isoformat():<26}{entry.event_type:<12}{entry.confidence:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="khizex-attendance",
        description="Face Recognition Attendance System with liveness detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new employee via webcam.")
    enroll_parser.add_argument("--id", required=True, help="Unique employee ID.")
    enroll_parser.add_argument("--name", required=True, help="Employee full name.")
    enroll_parser.add_argument(
        "--no-preview", action="store_true", help="Run enrollment without an OpenCV preview window."
    )
    enroll_parser.set_defaults(func=cmd_enroll)

    run_parser = subparsers.add_parser("run", help="Run the live attendance recognition loop.")
    run_parser.set_defaults(func=cmd_run)

    list_parser = subparsers.add_parser("list-employees", help="List all enrolled employees.")
    list_parser.set_defaults(func=cmd_list_employees)

    logs_parser = subparsers.add_parser("logs", help="Show attendance logs for an employee.")
    logs_parser.add_argument("--id", required=True, help="Employee ID.")
    logs_parser.add_argument("--limit", type=int, default=50, help="Max number of log rows to show.")
    logs_parser.set_defaults(func=cmd_logs)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
