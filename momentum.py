from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .core.utils import clamp, mean, names_match
except ImportError:
    from core.utils import clamp, mean, names_match


@dataclass(frozen=True)
class MomentumProfile:
    """Recent form features shared by NBA and football predictors."""

    team: str
    recent_form: str
    streak: int
    offense_state: float
    defense_state: float
    momentum_score: float


def nba_momentum(team: str, games: list[Any]) -> MomentumProfile:
    recent = games[:5]
    if not recent:
        return MomentumProfile(team, "-----", 0, 0.0, 0.0, 0.0)
    form = "".join("W" if game.win else "L" for game in recent)
    streak = streak_value(form)
    offense_state = mean([game.team_score for game in recent], 114.0) - mean([game.team_score for game in games[:10]], 114.0)
    defense_state = mean([game.opponent_score for game in games[:10]], 114.0) - mean([game.opponent_score for game in recent], 114.0)
    momentum_score = clamp(streak * 1.4 + offense_state * 0.10 + defense_state * 0.10, -8.0, 8.0)
    return MomentumProfile(team, form, streak, offense_state, defense_state, momentum_score)


def football_momentum(team: str, matches: list[Any]) -> MomentumProfile:
    recent = matches[:5]
    if not recent:
        return MomentumProfile(team, "-----", 0, 0.0, 0.0, 0.0)
    results: list[str] = []
    goals_for: list[int] = []
    goals_against: list[int] = []
    baseline_for: list[int] = []
    baseline_against: list[int] = []
    for match in matches[:10]:
        gf, ga = football_scores_for_team(team, match)
        baseline_for.append(gf)
        baseline_against.append(ga)
    for match in recent:
        gf, ga = football_scores_for_team(team, match)
        goals_for.append(gf)
        goals_against.append(ga)
        if gf > ga:
            results.append("W")
        elif gf < ga:
            results.append("L")
        else:
            results.append("D")
    form = "".join(results)
    streak = streak_value(form)
    offense_state = mean(goals_for, 1.2) - mean(baseline_for, 1.2)
    defense_state = mean(baseline_against, 1.2) - mean(goals_against, 1.2)
    momentum_score = clamp(streak * 0.7 + offense_state * 1.4 + defense_state * 1.1, -5.0, 5.0)
    return MomentumProfile(team, form, streak, offense_state, defense_state, momentum_score)


def football_scores_for_team(team: str, match: Any) -> tuple[int, int]:
    if names_match(match.home_team, team):
        return int(match.home_goals), int(match.away_goals)
    return int(match.away_goals), int(match.home_goals)


def streak_value(form: str) -> int:
    if not form or form[0] == "-":
        return 0
    first = form[0]
    count = 0
    for char in form:
        if char != first:
            break
        count += 1
    if first == "W":
        return count
    if first == "L":
        return -count
    return 0

