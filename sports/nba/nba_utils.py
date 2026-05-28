from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("ai_sports_predictor")
PROJECT_DIR = Path(__file__).resolve().parent
try:
    from config import NBA_DATA_DIR

    DATA_DIR = NBA_DATA_DIR
except Exception:
    DATA_DIR = PROJECT_DIR / "data"
NBA_GAMES_CSV = DATA_DIR / "nba_games.csv"
NBA_TEAM_STATS_CSV = DATA_DIR / "nba_team_stats.csv"
NBA_PREDICTIONS_CSV = DATA_DIR / "nba_predictions.csv"
NBA_BACKTEST_RESULTS_CSV = DATA_DIR / "nba_backtest_results.csv"


def configure_logging(verbose: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_target_date(value: str | None, today: dt.date | None = None) -> dt.date:
    today = today or dt.date.today()
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
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    ensure_data_dir()
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
