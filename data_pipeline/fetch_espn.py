from __future__ import annotations

import datetime as dt
import os
from typing import Any

try:
    from config import ESPN_BASE_URL
except ImportError:  # pragma: no cover
    from ..config import ESPN_BASE_URL

from .common import LOGGER, configure_pipeline_logging, date_range, request_json, yyyymmdd
from .db import upsert_match, upsert_team_stat
from .fetch_schedule import stable_match_id
try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name


STATUS_MAP = {
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_PRE_GAME": "scheduled",
    "STATUS_FIRST_HALF": "live",
    "STATUS_SECOND_HALF": "live",
    "STATUS_HALFTIME": "live",
    "STATUS_FULL_TIME": "finished",
    "STATUS_FINAL": "finished",
    "STATUS_FINAL_PEN": "finished",
    "STATUS_POSTPONED": "cancelled",
    "STATUS_CANCELED": "cancelled",
}


def fetch_espn_live_matches(date_str: str) -> list[dict[str, Any]]:
    """Fetch normalized FIFA World Cup matches from ESPN public scoreboard."""

    configure_pipeline_logging()
    base_url = os.getenv("ESPN_BASE_URL") or ESPN_BASE_URL
    payload = request_json(base_url, params={"dates": date_str})
    events = payload.get("events") or []
    matches: list[dict[str, Any]] = []
    for event in events:
        normalized = normalize_espn_event(event)
        if normalized:
            matches.append(normalized)
    LOGGER.info("ESPN fetch date=%s matches=%s", date_str, len(matches))
    return matches


def fetch_live_matches(days_back: int = 1, days_forward: int = 7) -> list[dict[str, Any]]:
    configure_pipeline_logging()
    all_matches: list[dict[str, Any]] = []
    for day in date_range(days_back, days_forward):
        try:
            matches = fetch_espn_live_matches(yyyymmdd(day))
        except Exception as exc:
            LOGGER.warning("ESPN live fetch failed date=%s error=%s", day, exc)
            continue
        for match in matches:
            persist_espn_match(match)
        all_matches.extend(matches)
    LOGGER.info("ESPN live update complete count=%s", len(all_matches))
    return all_matches


def normalize_espn_event(event: dict[str, Any]) -> dict[str, Any] | None:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    competition = competitions[0]
    competitors = competition.get("competitors") or []
    home = next((item for item in competitors if item.get("homeAway") == "home"), None)
    away = next((item for item in competitors if item.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    status_payload = competition.get("status") if isinstance(competition.get("status"), dict) else event.get("status")
    status_payload = status_payload if isinstance(status_payload, dict) else {}
    status_type = status_payload.get("type") if isinstance(status_payload.get("type"), dict) else {}
    status = STATUS_MAP.get(str(status_type.get("name") or ""), "live" if status_type.get("state") == "in" else "scheduled")
    home_team = team_display_name(home)
    away_team = team_display_name(away)
    match_time = str(event.get("date") or competition.get("date") or "")
    season = event.get("season") if isinstance(event.get("season"), dict) else {}
    season_type = season.get("type") if isinstance(season.get("type"), dict) else {}
    match_date = str(event.get("date") or competition.get("date") or "")[:10]
    try:
        stable_id = stable_match_id(home_team, away_team, dt.date.fromisoformat(match_date))
    except ValueError:
        stable_id = str(event.get("id") or f"espn_{home_team}_{away_team}_{match_time[:10]}")
    return {
        "match_id": stable_id,
        "home_team": home_team,
        "away_team": away_team,
        "match_time_utc": match_time,
        "status": status,
        "home_score": parse_score(home.get("score")),
        "away_score": parse_score(away.get("score")),
        "venue": ((competition.get("venue") or {}).get("fullName") or (competition.get("venue") or {}).get("displayName") or ""),
        "stage": str(season_type.get("name") or season.get("displayName") or ""),
        "group_name": "",
        "data_source": "ESPN",
        "data_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "elo_diff_home_advantage": None,
    }


def team_display_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") or {}
    return normalize_team_name(team.get("displayName") or team.get("name") or competitor.get("displayName") or "")


def parse_score(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def persist_espn_match(match: dict[str, Any]) -> None:
    upsert_match(match)
    if match.get("status") != "finished" or match.get("home_score") is None or match.get("away_score") is None:
        return
    match_date = dt.date.fromisoformat(str(match["match_time_utc"])[:10])
    home_score = int(match["home_score"])
    away_score = int(match["away_score"])
    home_result = "W" if home_score > away_score else "D" if home_score == away_score else "L"
    away_result = "W" if away_score > home_score else "D" if home_score == away_score else "L"
    upsert_team_stat(match["home_team"], match_date, match["away_team"], home_score, away_score, home_result, "ESPN")
    upsert_team_stat(match["away_team"], match_date, match["home_team"], away_score, home_score, away_result, "ESPN")
