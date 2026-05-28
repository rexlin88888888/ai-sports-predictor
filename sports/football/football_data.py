from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

try:
    from ...config import FOOTBALL_DATA_DIR, WORKSPACE_ROOT
    from ...core.data_loader import read_csv_checked
    from ...core.utils import names_match, safe_int
except ImportError:
    from config import FOOTBALL_DATA_DIR, WORKSPACE_ROOT
    from core.data_loader import read_csv_checked
    from core.utils import names_match, safe_int


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
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
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
    """Load today's football fixtures from cache, falling back to deployment demo fixtures."""

    fixtures = load_cached_live_fixtures(target_date)
    if fixtures:
        return fixtures
    LOGGER.warning("Football live fixture cache unavailable; using public demo fixtures.")
    return [
        FootballFixture(target_date, "Mexico", "South Africa", "WORLD_CUP"),
        FootballFixture(target_date, "United States", "Belgium", "WORLD_CUP"),
    ]


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
                str(row["home_team"]),
                str(row["away_team"]),
                str(row.get("mode") or "WORLD_CUP"),
            )
        )
    return fixtures
