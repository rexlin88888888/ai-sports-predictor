from __future__ import annotations

import datetime as dt
import json
from typing import Any

try:
    from ..config import AUTOMATION_STATUS_JSON, ensure_project_dirs
except ImportError:
    from config import AUTOMATION_STATUS_JSON, ensure_project_dirs


def read_automation_status() -> dict[str, Any]:
    if not AUTOMATION_STATUS_JSON.exists():
        return {}
    try:
        return json.loads(AUTOMATION_STATUS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_automation_status(**updates: Any) -> dict[str, Any]:
    ensure_project_dirs()
    payload = read_automation_status()
    payload.update(updates)
    payload["last_status_write"] = dt.datetime.now().isoformat(timespec="seconds")
    AUTOMATION_STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload
