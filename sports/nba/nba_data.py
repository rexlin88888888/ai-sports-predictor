from __future__ import annotations

import csv
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from core.live_schedule import fetch_nba_schedule_from_espn
except ImportError:
    from ...core.live_schedule import fetch_nba_schedule_from_espn

try:
    from .nba_utils import (
        NBA_GAMES_CSV,
        NBA_PREDICTIONS_CSV,
        NBA_TEAM_STATS_CSV,
        append_csv_row,
        ensure_data_dir,
        names_match,
        normalize_name,
        safe_int,
        season_from_date,
    )
except ImportError:
    from .nba_utils import (
        NBA_GAMES_CSV,
        NBA_PREDICTIONS_CSV,
        NBA_TEAM_STATS_CSV,
        append_csv_row,
        ensure_data_dir,
        names_match,
        normalize_name,
        safe_int,
        season_from_date,
    )

try:
    from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
    from nba_api.stats.static import teams as nba_static_teams
except ImportError:
    leaguegamefinder = None
    scoreboardv2 = None
    nba_static_teams = None


LOGGER = logging.getLogger("ai_sports_predictor")


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    date: dt.date
    time_text: str
    home_team: str
    away_team: str
    data_source: str = "unknown"


@dataclass(frozen=True)
class TeamGame:
    game_id: str
    date: dt.date
    team: str
    opponent: str
    is_home: bool
    team_score: int
    opponent_score: int

    @property
    def win(self) -> bool:
        return self.team_score > self.opponent_score

    @property
    def margin(self) -> int:
        return self.team_score - self.opponent_score

    @property
    def total_points(self) -> int:
        return self.team_score + self.opponent_score


@dataclass(frozen=True)
class CompletedGame:
    game_id: str
    date: dt.date
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    @property
    def winner(self) -> str:
        return self.home_team if self.home_score > self.away_score else self.away_team


class NBADataClient:
    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._team_cache: list[dict[str, Any]] | None = None

    def get_tomorrow_schedule(self, target_date: dt.date) -> list[ScheduledGame]:
        api_games = self._get_schedule_from_api(target_date)
        if api_games:
            LOGGER.info("Using NBA Stats API schedule for %s: %s game(s).", target_date, len(api_games))
            return api_games
        espn_games = self._get_schedule_from_espn(target_date)
        if espn_games:
            LOGGER.info("Using ESPN NBA schedule for %s: %s game(s).", target_date, len(espn_games))
            return espn_games
        csv_games = self._get_schedule_from_csv(target_date, NBA_GAMES_CSV)
        if csv_games:
            LOGGER.warning("NBA schedule API unavailable/empty. Using local CSV schedule: %s.", NBA_GAMES_CSV)
            return csv_games
        return []

    def _get_schedule_from_api(self, target_date: dt.date) -> list[ScheduledGame]:
        if scoreboardv2 is None or nba_static_teams is None:
            LOGGER.warning("nba_api is not installed. Run: pip install nba_api")
            return []
        try:
            board = scoreboardv2.ScoreboardV2(game_date=target_date.strftime("%m/%d/%Y"), timeout=self.timeout)
            frames = board.get_data_frames()
            if not frames:
                return []
            frame = frames[0]
            if frame.empty:
                return []
            id_to_name = self.team_id_to_name()
            games: list[ScheduledGame] = []
            for _, row in frame.iterrows():
                home_id = safe_int(row.get("HOME_TEAM_ID"))
                away_id = safe_int(row.get("VISITOR_TEAM_ID"))
                if home_id is None or away_id is None:
                    continue
                games.append(
                    ScheduledGame(
                        game_id=str(row.get("GAME_ID") or ""),
                        date=target_date,
                        time_text=str(row.get("GAME_STATUS_TEXT") or row.get("GAME_SEQUENCE") or "TBD"),
                        home_team=id_to_name.get(home_id, str(row.get("HOME_TEAM_ABBREVIATION") or home_id)),
                        away_team=id_to_name.get(away_id, str(row.get("VISITOR_TEAM_ABBREVIATION") or away_id)),
                        data_source="live_api",
                    )
                )
            return games
        except Exception as exc:
            LOGGER.warning("Could not fetch NBA schedule from NBA Stats API: %s", exc)
            return []

    def _get_schedule_from_espn(self, target_date: dt.date) -> list[ScheduledGame]:
        fixtures = fetch_nba_schedule_from_espn(target_date)
        return [
            ScheduledGame(
                game_id=fixture.game_id,
                date=fixture.date,
                time_text=fixture.time_text,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                data_source=fixture.data_source,
            )
            for fixture in fixtures
        ]

    def _get_schedule_from_csv(self, target_date: dt.date, path: Path) -> list[ScheduledGame]:
        if not path.exists():
            LOGGER.warning("Local NBA schedule/history CSV not found: %s", path)
            return []
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Could not read local NBA CSV %s: %s", path, exc)
            return []
        required = {"date", "home_team", "away_team"}
        if missing := required - set(frame.columns):
            LOGGER.warning("Local NBA CSV missing schedule columns: %s", ", ".join(sorted(missing)))
            return []
        games: list[ScheduledGame] = []
        for _, row in frame.iterrows():
            try:
                game_date = dt.date.fromisoformat(str(row["date"])[:10])
            except ValueError:
                continue
            if game_date != target_date:
                continue
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                continue
            games.append(
                ScheduledGame(
                    game_id=str(row.get("game_id") or f"csv-{game_date}-{row['home_team']}-{row['away_team']}"),
                    date=game_date,
                    time_text=str(row.get("time") or "TBD"),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    data_source="fallback_cache",
                )
            )
        return games

    def get_team_recent_games(self, team_name: str, target_date: dt.date, limit: int = 40) -> list[TeamGame]:
        season = season_from_date(target_date)
        games = self._get_team_games_from_csv(team_name, NBA_GAMES_CSV)
        if len([game for game in games if game.date < target_date]) < 10:
            games = self._get_team_games_from_api(team_name, season, limit=limit * 2)
        filtered = [game for game in games if game.date < target_date]
        filtered.sort(key=lambda game: game.date, reverse=True)
        if len(filtered) < 10:
            LOGGER.warning("%s has only %s usable historical game(s) before %s.", team_name, len(filtered), target_date)
        return filtered[:limit]

    def refresh_completed_games_cache(self, season: str) -> int:
        if leaguegamefinder is None:
            LOGGER.warning("nba_api is not installed; cannot refresh NBA history cache.")
            return 0
        frames: list[pd.DataFrame] = []
        for season_type in ("Regular Season", "Playoffs"):
            try:
                finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=season_type,
                    timeout=self.timeout,
                )
                frame = finder.get_data_frames()[0]
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                LOGGER.warning("Could not refresh NBA %s %s history cache: %s", season, season_type, exc)
        if not frames:
            LOGGER.warning("NBA history cache refresh returned no games for season %s.", season)
            return 0
        merged = pd.concat(frames, ignore_index=True)
        rows: list[dict[str, Any]] = []
        for game_id, group in merged.groupby("GAME_ID"):
            if len(group) < 2:
                continue
            home_rows = group[group["MATCHUP"].astype(str).str.contains(" vs. ", regex=False)]
            away_rows = group[group["MATCHUP"].astype(str).str.contains(" @ ", regex=False)]
            if home_rows.empty or away_rows.empty:
                continue
            home = home_rows.iloc[0]
            away = away_rows.iloc[0]
            home_score = safe_int(home.get("PTS"))
            away_score = safe_int(away.get("PTS"))
            if home_score is None or away_score is None:
                continue
            rows.append(
                {
                    "game_id": str(game_id),
                    "date": str(home.get("GAME_DATE"))[:10],
                    "home_team": str(home.get("TEAM_NAME")),
                    "away_team": str(away.get("TEAM_NAME")),
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
        if not rows:
            return 0
        ensure_data_dir()
        with NBA_GAMES_CSV.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["game_id", "date", "home_team", "away_team", "home_score", "away_score"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Refreshed NBA history cache: %s completed game(s) saved to %s.", len(rows), NBA_GAMES_CSV)
        return len(rows)

    def _get_team_games_from_api(self, team_name: str, season: str, limit: int) -> list[TeamGame]:
        if leaguegamefinder is None or nba_static_teams is None:
            return []
        team_id = self.resolve_team_id(team_name)
        if team_id is None:
            LOGGER.warning("Could not resolve NBA team name: %s", team_name)
            return []
        rows: list[pd.DataFrame] = []
        for season_type in ("Regular Season", "Playoffs"):
            try:
                finder = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=team_id,
                    season_nullable=season,
                    season_type_nullable=season_type,
                    timeout=self.timeout,
                )
                frame = finder.get_data_frames()[0]
                if not frame.empty:
                    rows.append(frame)
            except Exception as exc:
                LOGGER.warning("NBA gamefinder failed for %s %s %s: %s", team_name, season, season_type, exc)
        if not rows:
            return []
        frame = pd.concat(rows, ignore_index=True).head(limit)
        abbr = self.team_abbreviation_to_name()
        games: list[TeamGame] = []
        for _, row in frame.iterrows():
            team_score = safe_int(row.get("PTS"))
            plus_minus = safe_int(row.get("PLUS_MINUS"))
            if team_score is None or plus_minus is None:
                continue
            matchup = str(row.get("MATCHUP") or "")
            is_home = " vs. " in matchup
            opponent_abbr = matchup.split()[-1] if matchup else ""
            opponent = abbr.get(opponent_abbr, opponent_abbr or "Unknown")
            try:
                game_date = dt.date.fromisoformat(str(row.get("GAME_DATE"))[:10])
            except ValueError:
                continue
            games.append(
                TeamGame(
                    game_id=str(row.get("GAME_ID") or ""),
                    date=game_date,
                    team=team_name,
                    opponent=opponent,
                    is_home=is_home,
                    team_score=int(team_score),
                    opponent_score=int(team_score - plus_minus),
                )
            )
        return games

    def _get_team_games_from_csv(self, team_name: str, path: Path) -> list[TeamGame]:
        if not path.exists():
            return []
        games: list[TeamGame] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    home_score = safe_int(row.get("home_score"))
                    away_score = safe_int(row.get("away_score"))
                    if home_score is None or away_score is None:
                        continue
                    try:
                        game_date = dt.date.fromisoformat(str(row.get("date", ""))[:10])
                    except ValueError:
                        continue
                    home = str(row.get("home_team") or "")
                    away = str(row.get("away_team") or "")
                    if names_match(home, team_name):
                        games.append(TeamGame(str(row.get("game_id") or ""), game_date, home, away, True, home_score, away_score))
                    elif names_match(away, team_name):
                        games.append(TeamGame(str(row.get("game_id") or ""), game_date, away, home, False, away_score, home_score))
        except Exception as exc:
            LOGGER.warning("Could not read local team history from %s: %s", path, exc)
        return games

    def fetch_injuries(self, team_name: str) -> list[dict[str, str]]:
        LOGGER.warning("Injury API is not configured. Leaving injury list empty for %s.", team_name)
        return []

    def save_prediction(self, row: dict[str, Any]) -> None:
        append_csv_row(
            NBA_PREDICTIONS_CSV,
            row,
            [
                "run_date",
                "game_date",
                "game_id",
                "home_team",
                "away_team",
                "predicted_winner",
                "home_win_probability",
                "away_win_probability",
                "predicted_home_score",
                "predicted_away_score",
                "predicted_total",
                "actual_home_score",
                "actual_away_score",
                "actual_winner",
                "prediction_correct",
            ],
        )

    def save_team_stats(self, row: dict[str, Any]) -> None:
        append_csv_row(
            NBA_TEAM_STATS_CSV,
            row,
            [
                "run_date",
                "game_date",
                "team",
                "opponent",
                "is_home",
                "games_used",
                "recent5_win_pct",
                "recent10_win_pct",
                "season_win_pct",
                "avg_points_for",
                "avg_points_against",
                "recent_off_eff",
                "recent_def_eff",
                "home_win_pct",
                "away_win_pct",
                "rest_days",
                "back_to_back",
                "injury_penalty",
            ],
        )

    def team_id_to_name(self) -> dict[int, str]:
        return {int(team["id"]): str(team["full_name"]) for team in self._teams()}

    def team_abbreviation_to_name(self) -> dict[str, str]:
        return {str(team["abbreviation"]): str(team["full_name"]) for team in self._teams()}

    def resolve_team_id(self, team_name: str) -> int | None:
        query = normalize_name(team_name)
        for team in self._teams():
            names = [team.get("full_name"), team.get("nickname"), team.get("city"), team.get("abbreviation")]
            if any(query == normalize_name(str(name or "")) for name in names):
                return int(team["id"])
        for team in self._teams():
            if query in normalize_name(str(team.get("full_name") or "")):
                return int(team["id"])
        return None

    def canonical_team_name(self, team_name: str) -> str:
        query = normalize_name(team_name)
        for team in self._teams():
            names = [team.get("full_name"), team.get("nickname"), team.get("city"), team.get("abbreviation")]
            if any(query == normalize_name(str(name or "")) for name in names):
                return str(team.get("full_name") or team_name)
        for team in self._teams():
            full_name = str(team.get("full_name") or "")
            if query in normalize_name(full_name):
                return full_name
        return team_name

    def _teams(self) -> list[dict[str, Any]]:
        if self._team_cache is None:
            self._team_cache = nba_static_teams.get_teams() if nba_static_teams is not None else []
        return self._team_cache
