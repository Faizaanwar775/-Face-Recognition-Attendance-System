# enroll_employee.py
"""
enroll_employee.py

CLI entrypoint: run this to enroll a new employee.

Usage:
    python enroll_employee.py --id E001 --name "Ali Raza"
"""

from __future__ import annotations

import argparse
import logging
import sys

from enrollment.enroll import EnrollmentError, enroll_employee

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a new employee for face recognition attendance.")
    parser.add_argument("--id", required=True, dest="employee_id", help="Unique employee ID, e.g. E001")
    parser.add_argument("--name", required=True, dest="full_name", help="Employee full name")
    parser.add_argument("--frames", type=int, default=5, help="Number of reference frames to capture (default: 5)")
    args = parser.parse_args()

    try:
        enroll_employee(
            employee_id=args.employee_id,
            full_name=args.full_name,
            num_reference_frames=args.frames,
        )
    except EnrollmentError as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # camera/model init failures, etc.
        print(f"Unexpected error during enrollment: {exc}", file=sys.stderr)
        return 1

    print(f"Employee {args.employee_id} ({args.full_name}) enrolled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())