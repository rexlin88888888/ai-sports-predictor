from __future__ import annotations

import csv
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from ...config import NBA_DATA_DIR
    from ...core.utils import names_match, normalize_name, safe_float
except ImportError:
    from config import NBA_DATA_DIR
    from core.utils import names_match, normalize_name, safe_float


LOGGER = logging.getLogger("sports_predictor")
INJURIES_CSV = NBA_DATA_DIR / "injuries.csv"
INJURY_CACHE_CSV = NBA_DATA_DIR / "injury_cache.csv"


SUPERSTAR_IMPACT = {
    "Nikola Jokic": 6.5,
    "Luka Doncic": 6.5,
    "Giannis Antetokounmpo": 6.3,
    "Shai Gilgeous-Alexander": 6.2,
    "Joel Embiid": 6.0,
    "Jayson Tatum": 5.7,
    "Stephen Curry": 5.7,
    "LeBron James": 5.5,
    "Anthony Davis": 5.4,
    "Kevin Durant": 5.2,
    "Jalen Brunson": 5.0,
    "Anthony Edwards": 5.0,
    "Jimmy Butler": 4.8,
    "Jimmy Butler III": 4.8,
}

STARTER_IMPACT = {
    "Aaron Gordon": 3.0,
    "Jalen Williams": 4.0,
    "Kyrie Irving": 4.8,
    "Jaylen Brown": 4.5,
    "Kristaps Porzingis": 3.8,
    "Tyrese Haliburton": 5.2,
    "Donovan Mitchell": 5.0,
    "Ja Morant": 5.0,
    "Zach LaVine": 3.8,
}

STATUS_MULTIPLIER = {
    "out": 1.0,
    "doubtful": 0.8,
    "questionable": 0.45,
    "probable": 0.15,
}

TEAM_NAME_ALIASES = {
    "los angeles clippers": ["la clippers"],
    "la clippers": ["los angeles clippers"],
}


@dataclass(frozen=True)
class InjuryRecord:
    team: str
    player: str
    status: str
    injury: str
    comment: str
    source: str
    updated_at: str
    player_impact_score: float
    is_projected_starter: bool

    @property
    def weighted_impact(self) -> float:
        multiplier = status_multiplier(self.status)
        return self.player_impact_score * multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "player": self.player,
            "status": self.status,
            "injury": self.injury,
            "comment": self.comment,
            "source": self.source,
            "updated_at": self.updated_at,
            "player_impact_score": round(self.player_impact_score, 3),
            "weighted_impact": round(self.weighted_impact, 3),
            "is_projected_starter": self.is_projected_starter,
        }


@dataclass(frozen=True)
class TeamInjuryImpact:
    team: str
    injuries: list[InjuryRecord]
    team_injury_penalty: float
    missing_starters: int
    source: str

    @property
    def uncertainty_count(self) -> int:
        return sum(1 for item in self.injuries if item.status.lower() in {"questionable", "probable", "doubtful"})

    def as_model_injuries(self) -> list[dict[str, Any]]:
        return [
            {
                "player": item.player,
                "status": item.status,
                "injury": item.injury,
                "impact_score": item.player_impact_score,
                "weighted_impact": item.weighted_impact,
                "is_projected_starter": item.is_projected_starter,
            }
            for item in self.injuries
        ]


class InjuryDataClient:
    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout
        self._team_lookup: dict[str, dict[str, str]] | None = None
        self._impact_cache: dict[str, TeamInjuryImpact] = {}

    def get_team_impact(self, team_name: str) -> TeamInjuryImpact:
        if team_name in self._impact_cache:
            return self._impact_cache[team_name]
        cached = self.load_cached_team_injuries(team_name)
        if cached:
            impact = self.build_impact(team_name, cached, "cache")
            self._impact_cache[team_name] = impact
            return impact
        injuries = self.fetch_team_injuries(team_name)
        if not injuries:
            LOGGER.warning("WARNING: missing injury data for %s", team_name)
            impact = TeamInjuryImpact(team_name, [], 0.0, 0, "missing")
            self._impact_cache[team_name] = impact
            return impact
        self.save_injuries(injuries)
        impact = self.build_impact(team_name, injuries, "ESPN")
        self._impact_cache[team_name] = impact
        return impact

    def build_impact(self, team_name: str, injuries: list[InjuryRecord], source: str) -> TeamInjuryImpact:
        penalty = -min(12.0, sum(item.weighted_impact for item in injuries))
        missing_starters = sum(1 for item in injuries if item.is_projected_starter and item.status.lower() in {"out", "doubtful"})
        return TeamInjuryImpact(team_name, injuries, penalty, missing_starters, source)

    def fetch_team_injuries(self, team_name: str) -> list[InjuryRecord]:
        team_ref = self.resolve_espn_team_ref(team_name)
        if not team_ref:
            LOGGER.warning("WARNING: could not resolve ESPN injury team id for %s", team_name)
            return []
        url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_ref}/injuries"
        params = {"lang": "en", "region": "us"}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            LOGGER.warning("WARNING: ESPN injury API failed for %s: %s", team_name, exc)
            return []
        records: list[InjuryRecord] = []
        for item in (data.get("items") or [])[:12]:
            ref = str(item.get("$ref") or "").replace("http://", "https://")
            if not ref:
                continue
            record = self.fetch_injury_ref(ref, team_name)
            if record:
                records.append(record)
        return records

    def fetch_injury_ref(self, ref: str, fallback_team: str) -> InjuryRecord | None:
        try:
            injury = requests.get(ref, timeout=self.timeout).json()
            athlete_ref = str((injury.get("athlete") or {}).get("$ref") or "").replace("http://", "https://")
            athlete = requests.get(athlete_ref, timeout=self.timeout).json() if athlete_ref else {}
            team_ref = str((injury.get("team") or {}).get("$ref") or "").replace("http://", "https://")
            injury_team = requests.get(team_ref, timeout=self.timeout).json() if team_ref else {}
        except Exception as exc:
            LOGGER.warning("WARNING: ESPN injury detail fetch failed: %s", exc)
            return None
        injury_team_name = str(injury_team.get("displayName") or injury_team.get("name") or fallback_team)
        if not team_names_equivalent(injury_team_name, fallback_team):
            LOGGER.warning(
                "WARNING: skipping mismatched injury record for %s assigned to %s.",
                injury_team_name,
                fallback_team,
            )
            return None
        player = str(athlete.get("displayName") or athlete.get("fullName") or "Unknown Player")
        status = normalize_status(str(injury.get("status") or (injury.get("type") or {}).get("description") or "Questionable"))
        details = injury.get("details") or {}
        injury_text = " ".join(str(details.get(key) or "") for key in ("type", "location", "detail")).strip()
        impact = player_impact_score(player)
        return InjuryRecord(
            team=fallback_team,
            player=player,
            status=status,
            injury=injury_text or "Undisclosed",
            comment=str(injury.get("shortComment") or injury.get("longComment") or ""),
            source="ESPN",
            updated_at=str(injury.get("date") or dt.datetime.now().isoformat(timespec="seconds")),
            player_impact_score=impact,
            is_projected_starter=impact >= 2.8,
        )

    def resolve_espn_team_ref(self, team_name: str) -> str | None:
        lookup = self.team_lookup()
        requested = normalize_name(team_name)
        candidate_names = [team_name, *TEAM_NAME_ALIASES.get(requested, [])]
        for _, row in lookup.items():
            abbreviation = normalize_name(row["abbreviation"])
            if any(names_match(row["display_name"], candidate) for candidate in candidate_names) or (
                abbreviation and abbreviation == requested
            ):
                return row["id"] or row["abbreviation"].lower()
        return None

    def team_lookup(self) -> dict[str, dict[str, str]]:
        if self._team_lookup is not None:
            return self._team_lookup
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
        try:
            data = requests.get(url, timeout=self.timeout).json()
        except Exception as exc:
            LOGGER.warning("WARNING: ESPN team lookup failed: %s", exc)
            self._team_lookup = {}
            return self._team_lookup
        lookup: dict[str, dict[str, str]] = {}
        for item in (((data.get("sports") or [{}])[0].get("leagues") or [{}])[0].get("teams") or []):
            team = item.get("team") or {}
            abbreviation = str(team.get("abbreviation") or "")
            if not abbreviation:
                continue
            lookup[abbreviation.lower()] = {
                "abbreviation": abbreviation,
                "display_name": str(team.get("displayName") or ""),
                "id": str(team.get("id") or ""),
                "slug": str(team.get("slug") or ""),
            }
        self._team_lookup = lookup
        return lookup

    def load_cached_team_injuries(self, team_name: str) -> list[InjuryRecord]:
        path = INJURY_CACHE_CSV if INJURY_CACHE_CSV.exists() else INJURIES_CSV
        if not path.exists():
            return []
        target_team = normalize_name(team_name)
        if not target_team:
            return []
        records: list[InjuryRecord] = []
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    row = clean_csv_row(row)
                    cached_team = normalize_name(str(row.get("team") or ""))
                    if not cached_team or not team_names_equivalent(cached_team, target_team):
                        continue
                    records.append(
                        InjuryRecord(
                            team=team_name,
                            player=str(row.get("player") or "Unknown Player"),
                            status=normalize_status(str(row.get("status") or "Questionable")),
                            injury=str(row.get("injury") or "Undisclosed"),
                            comment=str(row.get("comment") or ""),
                            source=str(row.get("source") or "cache"),
                            updated_at=str(row.get("updated_at") or ""),
                            player_impact_score=max(
                                safe_float(row.get("player_impact_score"), 0.7),
                                player_impact_score(str(row.get("player") or "")),
                            ),
                            is_projected_starter=(
                                str(row.get("is_projected_starter") or "").lower() in {"true", "1", "yes"}
                                or player_impact_score(str(row.get("player") or "")) >= 2.8
                            ),
                        )
                    )
        except Exception as exc:
            LOGGER.warning("WARNING: injury cache read failed for %s: %s", team_name, exc)
        return records

    def cached_team_names(self) -> list[str]:
        path = INJURY_CACHE_CSV if INJURY_CACHE_CSV.exists() else INJURIES_CSV
        if not path.exists():
            return []
        teams: set[str] = set()
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    row = clean_csv_row(row)
                    team = str(row.get("team") or "").strip()
                    if team:
                        teams.add(team)
        except Exception as exc:
            LOGGER.warning("WARNING: injury cache team scan failed: %s", exc)
        return sorted(teams)

    def save_injuries(self, records: list[InjuryRecord]) -> None:
        if not records:
            return
        NBA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing: dict[tuple[str, str], dict[str, Any]] = {}
        for path in (INJURIES_CSV,):
            if path.exists():
                try:
                    with path.open("r", newline="", encoding="utf-8-sig") as handle:
                        for row in csv.DictReader(handle):
                            row = clean_csv_row(row)
                            existing[(str(row.get("team")), str(row.get("player")))] = row
                except Exception:
                    pass
        for record in records:
            existing[(record.team, record.player)] = record.to_dict()
        fieldnames = [
            "team",
            "player",
            "status",
            "injury",
            "comment",
            "source",
            "updated_at",
            "player_impact_score",
            "weighted_impact",
            "is_projected_starter",
        ]
        for path in (INJURIES_CSV, INJURY_CACHE_CSV):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    {key: row.get(key, "") for key in fieldnames}
                    for row in existing.values()
                )


def player_impact_score(player: str) -> float:
    for name, score in SUPERSTAR_IMPACT.items():
        if names_match(name, player):
            return score
    for name, score in STARTER_IMPACT.items():
        if names_match(name, player):
            return score
    return 0.7


def normalize_status(value: str) -> str:
    lowered = value.strip().lower()
    if "out" in lowered:
        return "Out"
    if "doubt" in lowered:
        return "Doubtful"
    if "prob" in lowered:
        return "Probable"
    if "question" in lowered:
        return "Questionable"
    return value.strip().title() or "Questionable"


def status_multiplier(status: str) -> float:
    return STATUS_MULTIPLIER.get(status.lower(), 0.45)


def clean_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key or "").lstrip("\ufeff").strip('"'): value for key, value in row.items()}


def team_names_equivalent(left: str, right: str) -> bool:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return right_norm in TEAM_NAME_ALIASES.get(left_norm, []) or left_norm in TEAM_NAME_ALIASES.get(right_norm, [])
