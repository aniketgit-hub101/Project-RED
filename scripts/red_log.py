"""
scripts.red_log
CLI logger utility appending 7-field records to evidence/daily-reports.md per AR-00.2.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["append_daily_report", "main"]


def append_daily_report(
    member: str,
    phase: str,
    completed: str,
    committed: str,
    blocked: str,
    plan: str,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    log_path = repo_root / "evidence" / "daily-reports.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"{today} | {member} | {phase} | {completed} | {committed} | {blocked} | {plan}\n"

    if not log_path.exists():
        header = (
            "# PROJECT RED — DAILY EXECUTION LOGS\n\n"
            "DATE | MEMBER | PHASE | COMPLETED TODAY | COMMITTED TODAY | BLOCKED ON | PLAN FOR TOMORROW\n"
        )
        log_path.write_text(header + entry, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)


def main() -> int:
    if len(sys.argv) < 7:
        print("Usage: python scripts/red_log.py <MEMBER> <PHASE> <COMPLETED> <COMMITTED> <BLOCKED> <PLAN>")
        return 1
    append_daily_report(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    print("[SUCCESS] Daily report logged to evidence/daily-reports.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
