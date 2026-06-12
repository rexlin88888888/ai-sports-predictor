from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

try:
    from ...config import FOOTBALL_DATA_DIR, WORKSPACE_ROOT
    from ...core.data_loader import read_csv_checked
    from ...core.live_schedule import fetch_football_schedule
    from ...core.team_names import normalize_team_name, normalized_team_key
    from ...core.utils import names_match, safe_int
    from ...data_pipeline.db import fetch_all
except ImportError:
    from config import FOOTBALL_DATA_DIR, WORKSPACE_ROOT
    from core.data_loader import read_csv_checked
    from core.live_schedule import fetch_football_schedule
    from core.team_names import normalize_team_name, normalized_team_key
    from core.utils import names_match, safe_int
    from data_pipeline.db import fetch_all


LOGGER = logging.getLogger("sports_predictor")
INTERNATIONAL_CSV = FOOTBALL_DATA_DIR / "international_football.csv"
FOOTBALL_LIVE_CACHE_CSV = FOOTBALL_DATA_DIR / "football_live_cache.csv"
ROOT_INTERNATIONAL_CSV = WORKSPACE_ROOT / "international_football.csv"
ALLOWED_TOURNAMENTS = {"Friendly", "FIFA World Cup"}
MAX_REASONABLE_GOALS = 7


@dataclass(frozen=True)
class FootballMatch:
    date: dt.date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    tournament: str

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


@dataclass(frozen=True)
class FootballFixture:
    date: dt.date
    home_team: str
    away_team: str
    mode: str = "WORLD_CUP"
    data_source: str = "unknown"
    match_id: str = ""
    time_text: str = ""
    competition_name: str = ""
    stage: str = ""
    venue: str = ""


def load_matches() -> list[FootballMatch]:
    path = INTERNATIONAL_CSV if INTERNATIONAL_CSV.exists() else ROOT_INTERNATIONAL_CSV
    frame = read_csv_checked(path, {"date", "home_team", "away_team", "home_score", "away_score"})
    if frame is None:
        return []
    matches: list[FootballMatch] = []
    skipped = 0
    for _, row in frame.iterrows():
        tournament = str(row.get("tournament") or row.get("stage") or "")
        if tournament not in ALLOWED_TOURNAMENTS:
            skipped += 1
            continue
        home_goals = safe_int(row.get("home_score"))
        away_goals = safe_int(row.get("away_score"))
        if home_goals is None or away_goals is None:
            skipped += 1
            continue
        if home_goals > MAX_REASONABLE_GOALS or away_goals > MAX_REASONABLE_GOALS:
            skipped += 1
            continue
        try:
            match_date = dt.date.fromisoformat(str(row["date"])[:10])
        except ValueError:
            skipped += 1
            continue
        matches.append(
            FootballMatch(
                date=match_date,
                home_team=normalize_team_name(row["home_team"]),
                away_team=normalize_team_name(row["away_team"]),
                home_goals=home_goals,
                away_goals=away_goals,
                tournament=tournament,
            )
        )
    matches.sort(key=lambda item: item.date)
    LOGGER.info("Loaded football matches: kept=%s skipped=%s source=%s", len(matches), skipped, path)
    return matches


def team_matches(matches: list[FootballMatch], team: str, before_date: dt.date | None = None) -> list[FootballMatch]:
    selected = [
        match for match in matches
        if (before_date is None or match.date < before_date)
        and (names_match(match.home_team, team) or names_match(match.away_team, team))
    ]
    selected.sort(key=lambda item: item.date, reverse=True)
    return selected


def load_live_fixtures(target_date: dt.date) -> list[FootballFixture]:
    """Load football fixtures from live APIs, then local cache."""

    database_fixtures = load_database_fixtures(target_date)
    if database_fixtures:
        return database_fixtures
    live = fetch_football_schedule(target_date, "WORLD_CUP")
    if live:
        return [
            FootballFixture(item.date, normalize_team_name(item.home_team), normalize_team_name(item.away_team), item.mode or "WORLD_CUP", item.data_source)
            for item in live
        ]
    fixtures = load_cached_live_fixtures(target_date)
    if fixtures:
        return fixtures
    LOGGER.warning("No football fixtures available from live APIs or fallback cache for %s.", target_date)
    return []


def load_database_fixtures(target_date: dt.date) -> list[FootballFixture]:
    try:
        rows = fetch_all(
            "SELECT match_id, match_time_utc, home_team, away_team, stage, group_name, venue, data_source FROM matches WHERE substr(match_time_utc, 1, 10)=? AND status <> 'cancelled' ORDER BY match_time_utc",
            (target_date.isoformat(),),
        )
    except Exception:
        return []
    keyed: dict[tuple[str, str, str], FootballFixture] = {}
    for row in rows:
        try:
            fixture_date = dt.date.fromisoformat(str(row["match_time_utc"])[:10])
        except ValueError:
            fixture_date = target_date
        key = (fixture_date.isoformat(), normalized_team_key(row["home_team"]), normalized_team_key(row["away_team"]))
        fixture = FootballFixture(
            fixture_date,
            normalize_team_name(row["home_team"]),
            normalize_team_name(row["away_team"]),
            str(row.get("stage") or "WORLD_CUP"),
            "database:" + str(row.get("data_source") or "matches"),
            str(row.get("match_id") or ""),
            str(row.get("match_time_utc") or ""),
            "FIFA World Cup",
            str(row.get("stage") or row.get("group_name") or ""),
            str(row.get("venue") or ""),
        )
        existing = keyed.get(key)
        if existing is None or fixture_source_priority(fixture.data_source) > fixture_source_priority(existing.data_source):
            keyed[key] = fixture
    return list(keyed.values())


def fixture_source_priority(source: object) -> int:
    lowered = str(source or "").lower()
    if "fifa" in lowered and "world cup" not in lowered:
        return 40
    if "openfootball" in lowered:
        return 30
    if "espn" in lowered:
        return 20
    return 10 if lowered else 0


def load_cached_live_fixtures(target_date: dt.date) -> list[FootballFixture]:
    if not FOOTBALL_LIVE_CACHE_CSV.exists():
        return []
    frame = read_csv_checked(FOOTBALL_LIVE_CACHE_CSV, {"date", "home_team", "away_team"})
    if frame is None:
        return []
    fixtures: list[FootballFixture] = []
    for _, row in frame.iterrows():
        try:
            fixture_date = dt.date.fromisoformat(str(row["date"])[:10])
        except ValueError:
            continue
        if fixture_date != target_date:
            continue
        fixtures.append(
            FootballFixture(
                fixture_date,
                normalize_team_name(row["home_team"]),
                normalize_team_name(row["away_team"]),
                str(row.get("mode") or "WORLD_CUP"),
                "fallback_cache",
            )
        )
    return fixtures
