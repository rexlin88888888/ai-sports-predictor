from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from .team_names import normalize_team_name


LOGGER = logging.getLogger("sports_predictor")
TIMEOUT_SECONDS = 15
FIFA_OFFICIAL_SCHEDULE_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures"
OPENFOOTBALL_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"


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
    competition_name: str = ""
    stage: str = ""
    venue: str = ""


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
        fixture = _fixture_from_espn_event(event, target_date, "nba", "", "NBA")
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
        fixture = _fixture_from_espn_event(event, target_date, "football", mode, "FIFA World Cup")
        if fixture:
            fixtures.append(fixture)
    if fixtures:
        LOGGER.info("Using ESPN World Cup schedule API for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def fetch_fifa_official_schedule(target_date: dt.date) -> list[LiveFixture]:
    """Try to read FIFA's official public schedule page.

    FIFA currently renders fixtures through client-side assets. This function
    only returns matches when the official page exposes parsable structured
    match data; otherwise it returns [] so callers can safely fall back to
    OpenFootball and ESPN without inventing fixtures.
    """

    try:
        response = requests.get(
            FIFA_OFFICIAL_SCHEDULE_URL,
            headers={"User-Agent": "Mozilla/5.0 AI Sports Predictor schedule check"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        html = response.text
    except requests.RequestException as exc:
        LOGGER.warning("FIFA official schedule fetch failed for %s: %s", target_date, exc)
        return []
    return parse_fifa_official_schedule_html(html, target_date)


def parse_fifa_official_schedule_html(html: str, target_date: dt.date) -> list[LiveFixture]:
    fixtures: list[LiveFixture] = []
    # The official page may include JSON-LD Event objects in some deployments.
    # Keep parsing intentionally strict so non-fixture page text is ignored.
    for block in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I | re.S):
        try:
            import json

            payload = json.loads(block)
        except Exception:
            continue
        nodes = payload if isinstance(payload, list) else [payload]
        for node in nodes:
            if not isinstance(node, dict) or node.get("@type") not in {"SportsEvent", "Event"}:
                continue
            start = str(node.get("startDate") or "")
            if not start.startswith(target_date.isoformat()):
                continue
            name = str(node.get("name") or "")
            home, away = split_match_name(name)
            if not home or not away:
                continue
            venue = ""
            location = node.get("location")
            if isinstance(location, dict):
                venue = str(location.get("name") or "")
            fixtures.append(
                LiveFixture(
                    date=target_date,
                    home_team=normalize_team_name(home),
                    away_team=normalize_team_name(away),
                    sport="football",
                    mode="WORLD_CUP",
                    game_id=f"fifa_{target_date.isoformat()}_{normalize_team_name(home)}_{normalize_team_name(away)}",
                    time_text=start,
                    data_source="FIFA",
                    competition_name="FIFA World Cup",
                    stage=str(node.get("eventStatus") or ""),
                    venue=venue,
                )
            )
    if fixtures:
        LOGGER.info("Using FIFA official schedule for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def split_match_name(name: str) -> tuple[str, str]:
    for separator in (" vs ", " v ", " - ", " at "):
        if separator in name:
            left, right = name.split(separator, 1)
            if separator == " at ":
                return right.strip(), left.strip()
            return left.strip(), right.strip()
    return "", ""


def fetch_openfootball_world_cup_schedule(target_date: dt.date) -> list[LiveFixture]:
    """Fetch 2026 World Cup fixtures from openfootball without writing storage."""

    try:
        response = requests.get(OPENFOOTBALL_2026_URL, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("OpenFootball schedule fetch failed for %s: %s", target_date, exc)
        return []
    fixtures: list[LiveFixture] = []
    for item in payload.get("matches", []) or []:
        try:
            match_date = dt.date.fromisoformat(str(item.get("date") or "")[:10])
        except ValueError:
            continue
        if match_date != target_date:
            continue
        home = normalize_team_name(item.get("team1") or item.get("home_team") or "")
        away = normalize_team_name(item.get("team2") or item.get("away_team") or "")
        if not home or not away:
            continue
        time_text = str(item.get("time") or "")
        fixtures.append(
            LiveFixture(
                date=target_date,
                home_team=home,
                away_team=away,
                sport="football",
                mode=str(item.get("round") or "WORLD_CUP"),
                game_id=f"openfootball_{target_date.isoformat()}_{home}_{away}",
                time_text=f"{target_date.isoformat()} {time_text}".strip(),
                data_source="OpenFootball",
                competition_name="FIFA World Cup",
                stage=str(item.get("round") or item.get("group") or ""),
                venue=str(item.get("ground") or ""),
            )
        )
    if fixtures:
        LOGGER.info("Using OpenFootball schedule for %s: %s fixture(s).", target_date, len(fixtures))
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
                competition_name=competition or mode,
            )
        )
    if fixtures:
        LOGGER.info("Using Football-Data live schedule for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def fetch_football_schedule_from_espn(target_date: dt.date, mode: str) -> list[LiveFixture]:
    leagues = [
        ("fifa.world", "FIFA World Cup"),
        ("fifa.friendly", "International Friendly"),
    ]
    fixtures: list[LiveFixture] = []
    for league, competition_name in leagues:
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
            fixture = _fixture_from_espn_event(event, target_date, "football", mode, competition_name)
            if fixture:
                fixtures.append(fixture)
    if fixtures:
        LOGGER.info("Using ESPN football schedule API for %s: %s fixture(s).", target_date, len(fixtures))
    return fixtures


def fetch_international_schedule_from_espn(target_date: dt.date, competition_filter: str = "FIFA World Cup") -> list[LiveFixture]:
    """Fetch international fixtures by competition category from ESPN public endpoints."""

    endpoints = {
        "FIFA World Cup": [("fifa.world", "FIFA World Cup", "web")],
        "World Cup Qualifiers": [
            ("fifa.worldq.uefa", "World Cup Qualifiers", "site"),
            ("fifa.worldq.conmebol", "World Cup Qualifiers", "site"),
            ("fifa.worldq.concacaf", "World Cup Qualifiers", "site"),
            ("fifa.worldq.afc", "World Cup Qualifiers", "site"),
            ("fifa.worldq.caf", "World Cup Qualifiers", "site"),
        ],
        "All International Matches": [
            ("fifa.world", "FIFA World Cup", "web"),
            ("fifa.worldq.uefa", "World Cup Qualifiers", "site"),
            ("fifa.worldq.conmebol", "World Cup Qualifiers", "site"),
            ("fifa.worldq.concacaf", "World Cup Qualifiers", "site"),
            ("fifa.worldq.afc", "World Cup Qualifiers", "site"),
            ("fifa.worldq.caf", "World Cup Qualifiers", "site"),
        ],
    }
    selected = endpoints.get(competition_filter, endpoints["FIFA World Cup"])
    fixtures: list[LiveFixture] = []
    for league, competition_name, host_type in selected:
        host = "site.web.api.espn.com" if host_type == "web" else "site.api.espn.com"
        url = f"https://{host}/apis/site/v2/sports/soccer/{league}/scoreboard"
        params = {"dates": target_date.strftime("%Y%m%d")}
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("ESPN international schedule API failed for %s %s: %s", league, target_date, exc)
            continue
        for event in payload.get("events", []) or []:
            fixture = _fixture_from_espn_event(event, target_date, "football", competition_name, competition_name)
            if fixture:
                fixtures.append(fixture)
    return fixtures


def _fixture_from_espn_event(event: dict[str, Any], target_date: dt.date, sport: str, mode: str, competition_name: str = "") -> LiveFixture | None:
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
    competition = (competitions[0] if competitions else {}) or {}
    venue_payload = competition.get("venue") or {}
    stage = ""
    season = event.get("season") or {}
    if isinstance(season, dict):
        stage = str(season.get("slug") or "")
    return LiveFixture(
        date=target_date,
        home_team=home,
        away_team=away,
        sport=sport,
        mode=mode,
        game_id=str(event.get("id") or ""),
        time_text=event_time or status_text,
        data_source="live_api",
        competition_name=competition_name or mode,
        stage=stage,
        venue=str(venue_payload.get("fullName") or venue_payload.get("displayName") or ""),
    )
