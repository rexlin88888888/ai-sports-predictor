from __future__ import annotations

import json
import re
from typing import Any

try:
    from config import OUTPUTS_DIR
except ImportError:  # pragma: no cover
    from ..config import OUTPUTS_DIR

from .db import fetch_all, initialize_database


def build_elo_source_report() -> dict[str, Any]:
    initialize_database()
    elo_rows = fetch_all("SELECT team, source FROM team_elo")
    match_rows = fetch_all("SELECT home_team, away_team FROM matches")
    source_counts: dict[str, int] = {}
    for row in elo_rows:
        source = canonical_source(row.get("source"))
        source_counts[source] = source_counts.get(source, 0) + 1

    elo_by_team = {str(row.get("team") or ""): canonical_source(row.get("source")) for row in elo_rows}
    match_teams = sorted({str(row.get("home_team") or "") for row in match_rows} | {str(row.get("away_team") or "") for row in match_rows})
    placeholder_teams = [team for team in match_teams if is_placeholder_team(team)]
    estimated_teams = [
        team
        for team in match_teams
        if team and not is_placeholder_team(team) and elo_by_team.get(team) != "ELO"
    ]
    return {
        "successfully_read_real_elo_teams": source_counts.get("ELO", 0),
        "estimated_match_teams": len(estimated_teams),
        "placeholder_match_teams": len(placeholder_teams),
        "source_counts": source_counts,
        "estimated_team_list": estimated_teams,
        "placeholder_team_list": placeholder_teams,
    }


def write_elo_source_report() -> dict[str, Any]:
    report = build_elo_source_report()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "elo_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def canonical_source(source: object) -> str:
    value = str(source or "").strip().upper()
    if value == "ELO":
        return "ELO"
    if value == "ESPN":
        return "ESPN"
    return "Estimated"


def is_placeholder_team(team: str) -> bool:
    value = str(team or "").strip()
    if not value:
        return True
    return bool(
        re.fullmatch(r"[12][A-L]", value)
        or re.fullmatch(r"3[A-L](?:/[A-L])+", value)
        or re.fullmatch(r"[WL]\d+", value)
        or value.upper() in {"TBD", "TBA"}
    )


if __name__ == "__main__":
    print(json.dumps(write_elo_source_report(), ensure_ascii=False, indent=2))
