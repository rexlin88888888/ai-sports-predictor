from __future__ import annotations

import datetime as dt
import re
from typing import Any

from .common import LOGGER, configure_pipeline_logging, request_json
from .db import upsert_match
try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name


OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"


def fetch_openfootball_schedule() -> list[dict[str, Any]]:
    configure_pipeline_logging()
    payload = request_json(OPENFOOTBALL_URL)
    matches = [normalize_openfootball_match(item) for item in payload.get("matches", [])]
    matches = [item for item in matches if item]
    for match in matches:
        upsert_match(match)
    LOGGER.info("openfootball schedule update complete count=%s", len(matches))
    return matches


def normalize_openfootball_match(item: dict[str, Any]) -> dict[str, Any] | None:
    home = normalize_team_name(item.get("team1") or item.get("home_team") or "")
    away = normalize_team_name(item.get("team2") or item.get("away_team") or "")
    date_text = str(item.get("date") or "").strip()
    if not home or not away or not date_text:
        return None
    match_time = parse_openfootball_time(date_text, str(item.get("time") or "00:00 UTC"))
    match_id = stable_match_id(home, away, match_time.date())
    return {
        "match_id": str(match_id),
        "home_team": home,
        "away_team": away,
        "match_time_utc": match_time.isoformat(),
        "status": "scheduled",
        "home_score": None,
        "away_score": None,
        "venue": str(item.get("ground") or ""),
        "stage": str(item.get("round") or ""),
        "group_name": str(item.get("group") or ""),
        "data_source": "openfootball",
        "data_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "elo_diff_home_advantage": None,
    }


def parse_openfootball_time(date_text: str, time_text: str) -> dt.datetime:
    base_date = dt.date.fromisoformat(date_text[:10])
    match = re.match(r"(\d{1,2}):(\d{2})\s*UTC([+-]\d+)?", time_text.strip())
    if not match:
        return dt.datetime.combine(base_date, dt.time(0, 0))
    hour = int(match.group(1))
    minute = int(match.group(2))
    offset = int(match.group(3) or "0")
    local_time = dt.datetime.combine(base_date, dt.time(hour, minute))
    return local_time - dt.timedelta(hours=offset)


def stable_match_id(home: str, away: str, match_date: dt.date) -> str:
    home = normalize_team_name(home)
    away = normalize_team_name(away)
    safe_home = re.sub(r"[^A-Za-z0-9]+", "_", home).strip("_")
    safe_away = re.sub(r"[^A-Za-z0-9]+", "_", away).strip("_")
    return f"wc_{match_date.isoformat()}_{safe_home}_vs_{safe_away}"
