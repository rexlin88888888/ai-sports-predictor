from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from .team_names import normalize_team_name


LOGGER = logging.getLogger("sports_predictor")
TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class LiveFixture:
    date: dt.date
    home_team: str
    away_team: str
    sport: str
    mode: str = ""
    game_id: str = ""
    time_text: str = "TBD"
    data_source: str = "live_api"


def fetch_nba_schedule_from_espn(target_date: dt.date) -> list[LiveFixture]:
    """Fetch NBA games for one date from ESPN's public scoreboard endpoint."""

    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {"dates": target_date.strftime("%Y%m%d")}
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("NBA ESPN schedule API failed for %s: %s", target_date, exc)
        return []
    fixtures: list[LiveFixture] = []
    for event in payload.get("events", []) or []:
        fixture = _fixture_from_espn_event(event, target_date, "nba", "")
        if fixture:
            fixtures.append(fixture)
    if fixtures:
        LOGGER.info("Using ESPN NBA schedule API for %s: %s game(s).", target_date, len(fixtures))
    return fixtures


def fetch_football_schedule(target_date: dt.date, mode: str = "WORLD_CUP") -> list[LiveFixture]:
    """Fetch football fixtures from live APIs; returns [] when no provider has data."""

    fixtures = fetch_world_cup_schedule_from_espn(target_date, mode)
    if fixtures:
        return fixtures
    fixtures = fetch_football_schedule_from_football_data(target_date, mode)
    if fixtures:
        return fixtures
    fixtures = fetch_football_schedule_from_espn(target_date, mode)
    if fixtures:
        return fixtures
    LOGGER.warning("No live football fixtures found for %s.", target_date)
    return []


def fetch_world_cup_schedule_from_espn(target_date: dt.date, mode: str = "WORLD_CUP") -> list[LiveFixture]:
    """Fetch FIFA World Cup fixtures for one date from ESPN's public web endpoint."""

    url = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    params = {"dates": target_date.strftime("%Y%m%d")}
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("ESPN World Cup schedule API failed for %s: %s", target_date, exc)
        return []
    fixtures: list[LiveFixture] = []
    for event in payload.get("events", []) or []:
        fixture = _fixture_from_espn_event(event, target_date, "football", mode)
        if fixture:
            fixtures.append(fixture)
    if fixtures:
        LOGGER.info("Using ESPN World Cup schedule API for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def fetch_football_schedule_from_football_data(target_date: dt.date, mode: str) -> list[LiveFixture]:
    api_key = os.getenv("FOOTBALL_DATA_KEY")
    if not api_key:
        LOGGER.warning("FOOTBALL_DATA_KEY is not set; skipping Football-Data live schedule.")
        return []
    url = "https://api.football-data.org/v4/matches"
    params = {"dateFrom": target_date.isoformat(), "dateTo": target_date.isoformat()}
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Football-Data live schedule API failed for %s: %s", target_date, exc)
        return []
    fixtures: list[LiveFixture] = []
    for match in payload.get("matches", []) or []:
        home = normalize_team_name((match.get("homeTeam") or {}).get("name") or "")
        away = normalize_team_name((match.get("awayTeam") or {}).get("name") or "")
        if not home or not away:
            continue
        competition = str((match.get("competition") or {}).get("name") or "")
        status = str(match.get("status") or "SCHEDULED")
        fixtures.append(
            LiveFixture(
                date=target_date,
                home_team=home,
                away_team=away,
                sport="football",
                mode=competition or mode,
                game_id=str(match.get("id") or ""),
                time_text=status,
                data_source="live_api",
            )
        )
    if fixtures:
        LOGGER.info("Using Football-Data live schedule for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def fetch_football_schedule_from_espn(target_date: dt.date, mode: str) -> list[LiveFixture]:
    leagues = ["fifa.world", "fifa.friendly"]
    fixtures: list[LiveFixture] = []
    for league in leagues:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
        params = {"dates": target_date.strftime("%Y%m%d")}
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("ESPN football schedule API failed for %s %s: %s", league, target_date, exc)
            continue
        for event in payload.get("events", []) or []:
            fixture = _fixture_from_espn_event(event, target_date, "football", mode)
            if fixture:
                fixtures.append(fixture)
    if fixtures:
        LOGGER.info("Using ESPN football schedule API for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def _fixture_from_espn_event(event: dict[str, Any], target_date: dt.date, sport: str, mode: str) -> LiveFixture | None:
    competitors: list[dict[str, Any]] = []
    competitions = event.get("competitions") or []
    if competitions:
        competitors = competitions[0].get("competitors") or []
    home = away = ""
    for item in competitors:
        team = item.get("team") or {}
        name = str(team.get("displayName") or team.get("name") or "").strip()
        if item.get("homeAway") == "home":
            home = name
        elif item.get("homeAway") == "away":
            away = name
    home = normalize_team_name(home)
    away = normalize_team_name(away)
    if not home or not away:
        return None
    status = (event.get("status") or {}).get("type") or {}
    event_time = str(event.get("date") or "").strip()
    status_text = str(status.get("shortDetail") or status.get("detail") or "TBD")
    return LiveFixture(
        date=target_date,
        home_team=home,
        away_team=away,
        sport=sport,
        mode=mode,
        game_id=str(event.get("id") or ""),
        time_text=event_time or status_text,
        data_source="live_api",
    )
