from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from ..config import APP_LOG, ensure_project_dirs
except ImportError:
    from config import APP_LOG, ensure_project_dirs


LOGGER = logging.getLogger("sports_predictor")


def configure_logging(verbose: bool = False) -> None:
    ensure_project_dirs()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(APP_LOG, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_target_date(value: str | None) -> dt.date:
    today = dt.date.today()
    if not value or value.lower() == "tomorrow":
        return today + dt.timedelta(days=1)
    if value.lower() == "today":
        return today
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be today, tomorrow, or YYYY-MM-DD") from exc


def season_from_date(target_date: dt.date) -> str:
    start_year = target_date.year if target_date.month >= 10 else target_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    existing_fields: list[str] = []
    if path.exists():
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)
        except Exception:
            existing_fields = []
            existing_rows = []
    merged_fields = list(existing_fields)
    for field in fieldnames:
        if field not in merged_fields:
            merged_fields.append(field)
    for field in row:
        if field not in merged_fields:
            merged_fields.append(field)
    if existing_rows and merged_fields != existing_fields:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fields)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({key: existing.get(key, "") for key in merged_fields})
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields or fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in (merged_fields or fieldnames)})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace(".", "").split())


def names_match(left: str, right: str) -> bool:
    try:
        from core.team_names import team_names_match
    except ImportError:  # pragma: no cover
        from .team_names import team_names_match

    if team_names_match(left, right):
        return True
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    return left_norm == right_norm


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"
