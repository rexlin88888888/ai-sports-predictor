from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.automation_status import update_automation_status


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def run_step(label: str, args: list[str]) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "app.py"), *args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        update_automation_status(automation_status="failed", failed_step=label, failed_returncode=completed.returncode)
        raise SystemExit(completed.returncode)


def main() -> None:
    update_automation_status(automation_status="running", failed_step="")
    run_step("daily_predictions", ["--daily"])
    run_step("update_results", ["--update-results"])
    update_automation_status(automation_status="success", failed_step="")
    print("Daily automation complete")


if __name__ == "__main__":
    main()
