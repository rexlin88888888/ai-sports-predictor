from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from core.team_names import normalize_team_name, normalized_team_key
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name, normalized_team_key

from .db import duplicate_match_groups, fetch_all


def build_team_name_report(before_duplicate_groups: int | None = None, after_duplicate_groups: int | None = None) -> dict[str, Any]:
    match_rows = fetch_all("SELECT home_team, away_team, data_source FROM matches")
    elo_rows = fetch_all("SELECT team FROM team_elo")
    match_teams = sorted({normalize_team_name(row["home_team"]) for row in match_rows} | {normalize_team_name(row["away_team"]) for row in match_rows})
    elo_teams = sorted({normalize_team_name(row["team"]) for row in elo_rows})
    elo_keys = {normalized_team_key(team) for team in elo_teams}

    matched = [team for team in match_teams if normalized_team_key(team) in elo_keys]
    unrecognized = [team for team in match_teams if normalized_team_key(team) not in elo_keys]

    suggestions = suggest_aliases(unrecognized, elo_teams)
    duplicates = duplicate_match_groups()
    return {
        "duplicate_match_groups_before": before_duplicate_groups if before_duplicate_groups is not None else len(duplicates),
        "duplicate_match_groups_after": after_duplicate_groups if after_duplicate_groups is not None else len(duplicates),
        "matched_team_count": len(matched),
        "unrecognized_team_count": len(unrecognized),
        "unrecognized_teams": unrecognized,
        "alias_suggestions": suggestions,
    }


def suggest_aliases(unrecognized: list[str], canonical_teams: list[str]) -> dict[str, str]:
    suggestions: dict[str, str] = {}
    by_simplified = defaultdict(list)
    for team in canonical_teams:
        by_simplified[simplify_name(team)].append(team)
    for team in unrecognized:
        simplified = simplify_name(team)
        if simplified in by_simplified:
            suggestions[team] = by_simplified[simplified][0]
            continue
        best_team = ""
        best_score = 0.0
        for candidate in canonical_teams:
            score = SequenceMatcher(None, normalized_team_key(team), normalized_team_key(candidate)).ratio()
            if score > best_score:
                best_score = score
                best_team = candidate
        if best_score >= 0.82 and best_team:
            suggestions[team] = best_team
    return suggestions


def simplify_name(name: str) -> str:
    value = normalize_team_name(name)
    value = value.replace("&", "and")
    value = re.sub(r"\b(the|republic|of|and)\b", "", value, flags=re.I)
    value = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return value


def write_team_name_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
