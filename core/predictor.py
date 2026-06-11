from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

try:
    from data_pipeline.db import fetch_all
except ImportError:  # pragma: no cover
    from ..data_pipeline.db import fetch_all

try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from .team_names import normalize_team_name


@dataclass(frozen=True)
class TeamRealData:
    team: str
    elo_rating: float | None
    elo_source: str
    goals_for: float | None
    goals_against: float | None
    recent_form_weighted: float | None
    match_count: int
    recent_source: str
    estimated: bool


def load_team_data(team: str, before_date: dt.date | None = None) -> TeamRealData:
    """Load team data from DB, falling back gracefully when tables are empty."""

    team = normalize_team_name(team)
    elo_rows = safe_fetch_all("SELECT team, elo_rating, last_updated, source FROM team_elo WHERE lower(team)=lower(?)", (team,))
    elo_rating = float(elo_rows[0]["elo_rating"]) if elo_rows else None
    elo_source = f"{elo_rows[0].get('source') or 'eloratings.net'}, {elo_rows[0].get('last_updated')}" if elo_rows else "local estimate"
    stats_rows = load_recent_stats(team, before_date)
    if not stats_rows:
        return TeamRealData(team, elo_rating, elo_source, None, None, None, 0, "local estimate", True)
    weighted_for = weighted_average([float(row["goals_scored"]) for row in stats_rows])
    weighted_against = weighted_average([float(row["goals_conceded"]) for row in stats_rows])
    form_points = [3.0 if row["result"] == "W" else 1.0 if row["result"] == "D" else 0.0 for row in stats_rows]
    recent_source = f"{stats_rows[0].get('source') or 'database'}, {len(stats_rows)} actual matches"
    return TeamRealData(
        team=team,
        elo_rating=elo_rating,
        elo_source=elo_source,
        goals_for=weighted_for,
        goals_against=weighted_against,
        recent_form_weighted=weighted_average(form_points),
        match_count=len(stats_rows),
        recent_source=recent_source,
        estimated=len(stats_rows) < 5,
    )


def load_match_data(home_team: str, away_team: str, before_date: dt.date | None = None) -> dict[str, TeamRealData]:
    return {"home": load_team_data(home_team, before_date), "away": load_team_data(away_team, before_date)}


def load_recent_stats(team: str, before_date: dt.date | None = None) -> list[dict[str, Any]]:
    team = normalize_team_name(team)
    if before_date:
        rows = safe_fetch_all(
            "SELECT * FROM team_stats WHERE lower(team)=lower(?) AND match_date < ? ORDER BY match_date DESC LIMIT 5",
            (team, before_date.isoformat()),
        )
    else:
        rows = safe_fetch_all("SELECT * FROM team_stats WHERE lower(team)=lower(?) ORDER BY match_date DESC LIMIT 5", (team,))
    return rows


def weighted_average(values: list[float], half_life: float = 4.0) -> float:
    total = 0.0
    total_weight = 0.0
    for index, value in enumerate(values):
        weight = 0.5 ** (index / half_life)
        total += value * weight
        total_weight += weight
    return total / max(1e-9, total_weight)


def safe_fetch_all(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    try:
        return fetch_all(sql, params)
    except Exception:
        return []
