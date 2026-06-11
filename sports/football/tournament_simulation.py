from __future__ import annotations

import datetime as dt
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ...config import OUTPUTS_DIR
    from ...core.team_names import normalize_team_name, normalized_team_key
    from ...data_pipeline.db import fetch_all, initialize_database
    from .football_data import FootballMatch, load_matches, team_matches
except ImportError:  # pragma: no cover
    from config import OUTPUTS_DIR
    from core.team_names import normalize_team_name, normalized_team_key
    from data_pipeline.db import fetch_all, initialize_database
    from sports.football.football_data import FootballMatch, load_matches, team_matches


GROUP_ADVANCE_LABEL = "group_advance_probability"
ROUND_OF_32_LABEL = "round_of_32_probability"
ROUND_OF_16_LABEL = "round_of_16_probability"
QUARTERFINAL_LABEL = "quarterfinal_probability"
SEMIFINAL_LABEL = "semifinal_probability"
FINAL_LABEL = "final_probability"
CHAMPION_LABEL = "champion_probability"

KNOCKOUT_START_NUMBERS = {
    "Round of 32": 73,
    "Round of 16": 89,
    "Quarter-final": 97,
    "Semi-final": 101,
    "Final": 103,
}

KNOCKOUT_STAGE_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]


@dataclass(frozen=True)
class TournamentFixture:
    home_team: str
    away_team: str
    match_time_utc: str
    stage: str
    group_name: str
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None


@dataclass(frozen=True)
class TeamProfile:
    team: str
    group: str
    elo: float
    goals_for: float
    goals_against: float
    data_source: str


def simulate_world_cup(iterations: int = 1000, seed: int = 20260612) -> pd.DataFrame:
    """Run a deterministic Monte Carlo simulation for the 2026 World Cup field."""

    inputs = load_tournament_inputs()
    if not inputs["group_fixtures"]:
        return empty_simulation_frame()
    results = simulate_tournament(
        group_fixtures=inputs["group_fixtures"],
        knockout_fixtures=inputs["knockout_fixtures"],
        team_profiles=inputs["team_profiles"],
        iterations=iterations,
        seed=seed,
    )
    output_path = OUTPUTS_DIR / "world_cup_winner_prediction.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8")
    return results


def load_tournament_inputs() -> dict[str, Any]:
    initialize_database()
    rows = fetch_all(
        """
        SELECT home_team, away_team, match_time_utc, status, home_score, away_score, stage, group_name, data_source
        FROM matches
        WHERE home_team IS NOT NULL AND away_team IS NOT NULL
        ORDER BY match_time_utc
        """
    )
    fixtures = [fixture_from_row(row) for row in rows]
    group_map = infer_group_map(fixtures)
    group_fixtures = group_stage_fixtures(fixtures, group_map)
    knockout_fixtures = [fixture for fixture in fixtures if fixture.stage in KNOCKOUT_STAGE_ORDER]
    team_profiles = build_team_profiles(group_map)
    return {
        "group_fixtures": group_fixtures,
        "knockout_fixtures": knockout_fixtures,
        "team_profiles": team_profiles,
    }


def fixture_from_row(row: dict[str, Any]) -> TournamentFixture:
    return TournamentFixture(
        home_team=normalize_team_name(row.get("home_team") or ""),
        away_team=normalize_team_name(row.get("away_team") or ""),
        match_time_utc=str(row.get("match_time_utc") or ""),
        stage=str(row.get("stage") or ""),
        group_name=str(row.get("group_name") or ""),
        status=str(row.get("status") or "scheduled"),
        home_score=score_or_none(row.get("home_score")),
        away_score=score_or_none(row.get("away_score")),
    )


def score_or_none(value: object) -> int | None:
    if value is None or str(value) == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_group_map(fixtures: list[TournamentFixture]) -> dict[str, str]:
    group_map: dict[str, str] = {}
    for fixture in fixtures:
        code = group_code(fixture.group_name)
        if not code:
            continue
        for team in (fixture.home_team, fixture.away_team):
            if not is_placeholder(team):
                group_map[team] = code
    return group_map


def group_stage_fixtures(fixtures: list[TournamentFixture], group_map: dict[str, str]) -> list[TournamentFixture]:
    selected: dict[tuple[str, str, str], TournamentFixture] = {}
    for fixture in fixtures:
        if is_placeholder(fixture.home_team) or is_placeholder(fixture.away_team):
            continue
        home_group = group_map.get(fixture.home_team)
        away_group = group_map.get(fixture.away_team)
        explicit_group = bool(group_code(fixture.group_name))
        inferred_group = home_group and home_group == away_group
        if not explicit_group and not inferred_group:
            continue
        group_name = fixture.group_name or f"Group {home_group}"
        normalized = TournamentFixture(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            match_time_utc=fixture.match_time_utc,
            stage=fixture.stage or "Group Stage",
            group_name=group_name,
            status=fixture.status,
            home_score=fixture.home_score,
            away_score=fixture.away_score,
        )
        key = (normalized.match_time_utc[:10], normalized.home_team, normalized.away_team)
        selected[key] = normalized
    return sorted(selected.values(), key=lambda item: item.match_time_utc)


def build_team_profiles(group_map: dict[str, str]) -> dict[str, TeamProfile]:
    elo_rows = fetch_all("SELECT team, elo_rating, source FROM team_elo")
    elo_by_team = {
        normalize_team_name(row.get("team") or ""): float(row.get("elo_rating") or 1500.0)
        for row in elo_rows
    }
    elo_source = {
        normalize_team_name(row.get("team") or ""): str(row.get("source") or "Estimated")
        for row in elo_rows
    }
    historical = load_matches()
    profiles: dict[str, TeamProfile] = {}
    for team, group in sorted(group_map.items(), key=lambda item: (item[1], item[0])):
        history = team_matches(historical, team, dt.date(2026, 6, 11))
        goals_for, goals_against, recent_source = recent_goal_profile(history, team)
        source = elo_source.get(team, "Estimated")
        if recent_source != "historical":
            source = f"{source}+Estimated"
        profiles[team] = TeamProfile(
            team=team,
            group=group,
            elo=elo_by_team.get(team, 1500.0),
            goals_for=goals_for,
            goals_against=goals_against,
            data_source=source,
        )
    return profiles


def recent_goal_profile(history: list[FootballMatch], team: str) -> tuple[float, float, str]:
    if not history:
        return 1.22, 1.18, "estimated"
    recent = history[:8]
    goals_for: list[int] = []
    goals_against: list[int] = []
    for match in recent:
        if normalized_team_key(match.home_team) == normalized_team_key(team):
            goals_for.append(match.home_goals)
            goals_against.append(match.away_goals)
        else:
            goals_for.append(match.away_goals)
            goals_against.append(match.home_goals)
    return sum(goals_for) / len(goals_for), sum(goals_against) / len(goals_against), "historical"


def simulate_tournament(
    group_fixtures: list[TournamentFixture],
    knockout_fixtures: list[TournamentFixture],
    team_profiles: dict[str, TeamProfile],
    iterations: int = 1000,
    seed: int = 20260612,
) -> pd.DataFrame:
    rng = random.Random(seed)
    teams = sorted(team_profiles)
    counts = {
        team: {
            GROUP_ADVANCE_LABEL: 0,
            ROUND_OF_32_LABEL: 0,
            ROUND_OF_16_LABEL: 0,
            QUARTERFINAL_LABEL: 0,
            SEMIFINAL_LABEL: 0,
            FINAL_LABEL: 0,
            CHAMPION_LABEL: 0,
        }
        for team in teams
    }
    knockout_by_stage = {
        stage: [fixture for fixture in knockout_fixtures if fixture.stage == stage]
        for stage in KNOCKOUT_STAGE_ORDER
    }
    for _ in range(max(1, iterations)):
        group_rankings, qualified_thirds = simulate_group_stage(group_fixtures, team_profiles, rng)
        qualified = set()
        for ranked in group_rankings.values():
            qualified.update(ranked[:2])
        qualified.update(team for _, _, _, _, team in qualified_thirds[:8])
        for team in qualified:
            if team in counts:
                counts[team][GROUP_ADVANCE_LABEL] += 1
                counts[team][ROUND_OF_32_LABEL] += 1
        winners: dict[str, str] = {}
        losers: dict[str, str] = {}
        used_third_groups: set[str] = set()
        for stage in KNOCKOUT_STAGE_ORDER:
            stage_fixtures = knockout_by_stage.get(stage, [])
            start_number = KNOCKOUT_START_NUMBERS[stage]
            for index, fixture in enumerate(stage_fixtures):
                match_number = start_number + index
                home = resolve_slot(fixture.home_team, group_rankings, qualified_thirds, used_third_groups, winners, losers)
                away = resolve_slot(fixture.away_team, group_rankings, qualified_thirds, used_third_groups, winners, losers)
                if not home or not away or home == away:
                    continue
                winner, loser = simulate_knockout_match(home, away, team_profiles, rng)
                winners[f"W{match_number}"] = winner
                losers[f"L{match_number}"] = loser
                if stage == "Round of 32":
                    counts[winner][ROUND_OF_16_LABEL] += 1
                elif stage == "Round of 16":
                    counts[winner][QUARTERFINAL_LABEL] += 1
                elif stage == "Quarter-final":
                    counts[winner][SEMIFINAL_LABEL] += 1
                elif stage == "Semi-final":
                    counts[winner][FINAL_LABEL] += 1
                elif stage == "Final":
                    counts[winner][CHAMPION_LABEL] += 1
    rows: list[dict[str, Any]] = []
    denominator = max(1, iterations)
    for team in teams:
        profile = team_profiles[team]
        rows.append(
            {
                "team": team,
                "group": profile.group,
                "elo": round(profile.elo, 1),
                "data_source": profile.data_source,
                GROUP_ADVANCE_LABEL: counts[team][GROUP_ADVANCE_LABEL] / denominator,
                ROUND_OF_32_LABEL: counts[team][ROUND_OF_32_LABEL] / denominator,
                ROUND_OF_16_LABEL: counts[team][ROUND_OF_16_LABEL] / denominator,
                QUARTERFINAL_LABEL: counts[team][QUARTERFINAL_LABEL] / denominator,
                SEMIFINAL_LABEL: counts[team][SEMIFINAL_LABEL] / denominator,
                FINAL_LABEL: counts[team][FINAL_LABEL] / denominator,
                CHAMPION_LABEL: counts[team][CHAMPION_LABEL] / denominator,
            }
        )
    return pd.DataFrame(rows).sort_values(CHAMPION_LABEL, ascending=False).reset_index(drop=True)


def simulate_group_stage(
    fixtures: list[TournamentFixture],
    profiles: dict[str, TeamProfile],
    rng: random.Random,
) -> tuple[dict[str, list[str]], list[tuple[int, int, int, str, str]]]:
    standings = {
        group: {
            team: {"points": 0, "gd": 0, "gf": 0}
            for team, profile in profiles.items()
            if profile.group == group
        }
        for group in sorted({profile.group for profile in profiles.values()})
    }
    for fixture in fixtures:
        group = group_code(fixture.group_name) or profiles.get(fixture.home_team, TeamProfile("", "", 1500, 1.2, 1.2, "")).group
        if group not in standings or fixture.home_team not in standings[group] or fixture.away_team not in standings[group]:
            continue
        home_goals, away_goals = match_score(fixture, profiles, rng)
        apply_group_result(standings[group], fixture.home_team, fixture.away_team, home_goals, away_goals)
    rankings: dict[str, list[str]] = {}
    third_candidates: list[tuple[int, int, int, str, str]] = []
    for group, table in standings.items():
        ranked = sorted(
            table,
            key=lambda team: (
                table[team]["points"],
                table[team]["gd"],
                table[team]["gf"],
                profiles.get(team, TeamProfile(team, group, 1500, 1.2, 1.2, "")).elo,
            ),
            reverse=True,
        )
        rankings[group] = ranked
        if len(ranked) >= 3:
            team = ranked[2]
            third_candidates.append((table[team]["points"], table[team]["gd"], table[team]["gf"], group, team))
    third_candidates.sort(reverse=True)
    return rankings, third_candidates


def apply_group_result(table: dict[str, dict[str, int]], home: str, away: str, home_goals: int, away_goals: int) -> None:
    table[home]["gf"] += home_goals
    table[home]["gd"] += home_goals - away_goals
    table[away]["gf"] += away_goals
    table[away]["gd"] += away_goals - home_goals
    if home_goals > away_goals:
        table[home]["points"] += 3
    elif away_goals > home_goals:
        table[away]["points"] += 3
    else:
        table[home]["points"] += 1
        table[away]["points"] += 1


def match_score(fixture: TournamentFixture, profiles: dict[str, TeamProfile], rng: random.Random) -> tuple[int, int]:
    if fixture.status == "finished" and fixture.home_score is not None and fixture.away_score is not None:
        return fixture.home_score, fixture.away_score
    home_xg, away_xg = expected_goals(fixture.home_team, fixture.away_team, profiles, neutral=False)
    return poisson_sample(rng, home_xg), poisson_sample(rng, away_xg)


def simulate_knockout_match(
    home: str,
    away: str,
    profiles: dict[str, TeamProfile],
    rng: random.Random,
) -> tuple[str, str]:
    home_xg, away_xg = expected_goals(home, away, profiles, neutral=True)
    home_goals = poisson_sample(rng, home_xg)
    away_goals = poisson_sample(rng, away_xg)
    if home_goals > away_goals:
        return home, away
    if away_goals > home_goals:
        return away, home
    home_profile = profiles.get(home, fallback_profile(home))
    away_profile = profiles.get(away, fallback_profile(away))
    home_probability = elo_win_probability(home_profile.elo - away_profile.elo)
    return (home, away) if rng.random() < home_probability else (away, home)


def expected_goals(
    home: str,
    away: str,
    profiles: dict[str, TeamProfile],
    neutral: bool = False,
) -> tuple[float, float]:
    home_profile = profiles.get(home, fallback_profile(home))
    away_profile = profiles.get(away, fallback_profile(away))
    home_advantage = 22.0 if not neutral else 0.0
    elo_diff = home_profile.elo - away_profile.elo + home_advantage
    home_xg = 0.58 * home_profile.goals_for + 0.42 * away_profile.goals_against + 0.0012 * elo_diff + (0.06 if not neutral else 0.0)
    away_xg = 0.58 * away_profile.goals_for + 0.42 * home_profile.goals_against - 0.0010 * elo_diff
    return clamp(home_xg, 0.20, 3.60), clamp(away_xg, 0.20, 3.60)


def fallback_profile(team: str) -> TeamProfile:
    return TeamProfile(team=team, group="", elo=1500.0, goals_for=1.22, goals_against=1.18, data_source="Estimated")


def resolve_slot(
    slot: str,
    group_rankings: dict[str, list[str]],
    qualified_thirds: list[tuple[int, int, int, str, str]],
    used_third_groups: set[str],
    winners: dict[str, str],
    losers: dict[str, str],
) -> str | None:
    slot = slot.strip()
    if slot in winners:
        return winners[slot]
    if slot in losers:
        return losers[slot]
    seed_match = re.fullmatch(r"([123])([A-L])", slot)
    if seed_match:
        rank = int(seed_match.group(1))
        group = seed_match.group(2)
        ranked = group_rankings.get(group, [])
        if len(ranked) >= rank:
            return ranked[rank - 1]
        return None
    third_match = re.fullmatch(r"3([A-L](?:/[A-L])*)", slot)
    if third_match:
        allowed = set(third_match.group(1).split("/"))
        for _, _, _, group, team in qualified_thirds[:8]:
            if group in allowed and group not in used_third_groups:
                used_third_groups.add(group)
                return team
        for _, _, _, group, team in qualified_thirds[:8]:
            if group not in used_third_groups:
                used_third_groups.add(group)
                return team
    return None


def group_code(group_name: str) -> str:
    match = re.search(r"Group\s+([A-L])", str(group_name or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def is_placeholder(team: str) -> bool:
    value = str(team or "").strip()
    return bool(re.search(r"^(?:[123][A-L]|[WL]\d+|3[A-L]/)", value) or "/" in value)


def elo_win_probability(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def poisson_sample(rng: random.Random, lam: float) -> int:
    lam = clamp(lam, 0.05, 4.0)
    limit = math.exp(-lam)
    product = 1.0
    value = 0
    while product > limit:
        value += 1
        product *= rng.random()
    return max(0, min(7, value - 1))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def empty_simulation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team",
            "group",
            "elo",
            "data_source",
            GROUP_ADVANCE_LABEL,
            ROUND_OF_32_LABEL,
            ROUND_OF_16_LABEL,
            QUARTERFINAL_LABEL,
            SEMIFINAL_LABEL,
            FINAL_LABEL,
            CHAMPION_LABEL,
        ]
    )


def simulation_output_path() -> Path:
    return OUTPUTS_DIR / "world_cup_winner_prediction.csv"
