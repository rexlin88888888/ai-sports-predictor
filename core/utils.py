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
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


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
