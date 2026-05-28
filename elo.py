from __future__ import annotations

import csv
import datetime as dt
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .config import ELO_RATINGS_CSV, ensure_project_dirs
    from .core.utils import normalize_name, safe_float
except ImportError:
    from config import ELO_RATINGS_CSV, ensure_project_dirs
    from core.utils import normalize_name, safe_float


LOGGER = logging.getLogger("sports_predictor")


@dataclass(frozen=True)
class EloMatch:
    """Finished match input used by the generic Elo engine."""

    date: dt.date
    sport: str
    home_team: str
    away_team: str
    home_score: float
    away_score: float


@dataclass(frozen=True)
class EloSnapshot:
    """Elo features consumed by sport-specific predictors."""

    home_elo: float
    away_elo: float
    elo_diff: float
    elo_win_probability: float


class EloRatingSystem:
    """Shared Elo system for NBA and football with home-field correction."""

    def __init__(
        self,
        sport: str,
        ratings_path: Path = ELO_RATINGS_CSV,
        initial_elo: float = 1500.0,
        k_factor: float = 24.0,
        home_advantage: float = 55.0,
    ) -> None:
        self.sport = sport
        self.ratings_path = ratings_path
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.display_names: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not self.ratings_path.exists():
            return
        try:
            with self.ratings_path.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("sport") or "").lower() != self.sport.lower():
                        continue
                    team = str(row.get("team") or "")
                    key = normalize_name(team)
                    if not key:
                        continue
                    self.ratings[key] = safe_float(row.get("elo"), self.initial_elo)
                    self.display_names[key] = team
        except Exception as exc:
            LOGGER.warning("WARNING: could not read Elo ratings from %s: %s", self.ratings_path, exc)

    def save(self) -> None:
        ensure_project_dirs()
        rows: list[dict[str, str]] = []
        if self.ratings_path.exists():
            try:
                with self.ratings_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    rows = [row for row in csv.DictReader(handle) if str(row.get("sport") or "").lower() != self.sport.lower()]
            except Exception as exc:
                LOGGER.warning("WARNING: could not preserve existing Elo rows: %s", exc)
                rows = []
        now = dt.datetime.now().isoformat(timespec="seconds")
        for key, elo in sorted(self.ratings.items()):
            rows.append(
                {
                    "sport": self.sport,
                    "team": self.display_names.get(key, key.title()),
                    "elo": f"{elo:.2f}",
                    "updated_at": now,
                }
            )
        with self.ratings_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sport", "team", "elo", "updated_at"])
            writer.writeheader()
            writer.writerows(rows)

    def get(self, team: str) -> float:
        key = normalize_name(team)
        self.display_names.setdefault(key, team)
        return self.ratings.get(key, self.initial_elo)

    def set(self, team: str, elo: float) -> None:
        key = normalize_name(team)
        if key:
            self.ratings[key] = elo
            self.display_names[key] = team

    def expected_home_probability(self, home_team: str, away_team: str, home_advantage: float | None = None) -> float:
        home_elo = self.get(home_team)
        away_elo = self.get(away_team)
        advantage = self.home_advantage if home_advantage is None else home_advantage
        return 1.0 / (1.0 + math.pow(10.0, ((away_elo - (home_elo + advantage)) / 400.0)))

    def snapshot(self, home_team: str, away_team: str, home_advantage: float | None = None) -> EloSnapshot:
        home_elo = self.get(home_team)
        away_elo = self.get(away_team)
        probability = self.expected_home_probability(home_team, away_team, home_advantage)
        return EloSnapshot(
            home_elo=home_elo,
            away_elo=away_elo,
            elo_diff=home_elo - away_elo,
            elo_win_probability=probability,
        )

    def update_match(self, match: EloMatch) -> None:
        home_elo = self.get(match.home_team)
        away_elo = self.get(match.away_team)
        expected_home = self.expected_home_probability(match.home_team, match.away_team)
        if match.home_score > match.away_score:
            actual_home = 1.0
        elif match.home_score < match.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        margin = abs(match.home_score - match.away_score)
        multiplier = margin_multiplier(margin, home_elo - away_elo)
        change = self.k_factor * multiplier * (actual_home - expected_home)
        self.set(match.home_team, home_elo + change)
        self.set(match.away_team, away_elo - change)

    def rebuild(self, matches: Iterable[EloMatch], save: bool = True) -> None:
        self.ratings = {}
        self.display_names = {}
        for match in sorted(matches, key=lambda item: item.date):
            self.update_match(match)
        if save:
            self.save()


def margin_multiplier(margin: float, elo_diff: float) -> float:
    """Increase updates for meaningful upsets without letting blowouts dominate."""

    if margin <= 0:
        return 1.0
    return max(0.75, min(2.25, math.log(margin + 1.0) * (2.2 / ((abs(elo_diff) * 0.001) + 2.2))))

