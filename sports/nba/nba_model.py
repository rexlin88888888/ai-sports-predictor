from __future__ import annotations

import datetime as dt
import csv
import logging
from argparse import Namespace

try:
    from ...config import NBA_PREDICTIONS_CSV
    from ...core.base_model import SportPredictor
    from ...core.prediction_result import PredictionResult
    from ...core.utils import append_csv_row, clamp, parse_target_date, safe_int, season_from_date
    from ...elo import EloMatch, EloRatingSystem
    from ...fatigue import calculate_nba_fatigue
    from ...momentum import nba_momentum
    from ...predictor import ai_explain, apply_common_nba_adjustments
except ImportError:
    from config import NBA_PREDICTIONS_CSV
    from core.base_model import SportPredictor
    from core.prediction_result import PredictionResult
    from core.utils import append_csv_row, clamp, parse_target_date, safe_int, season_from_date
    from elo import EloMatch, EloRatingSystem
    from fatigue import calculate_nba_fatigue
    from momentum import nba_momentum
    from predictor import ai_explain, apply_common_nba_adjustments

from .nba_data import NBADataClient, ScheduledGame  # noqa: E402
from .nba_scoring_model import build_team_metrics, predict_game  # noqa: E402
from .nba_utils import NBA_GAMES_CSV  # noqa: E402

from .injury_data import InjuryDataClient, TeamInjuryImpact  # noqa: E402


LOGGER = logging.getLogger("sports_predictor")


class NBAPredictor(SportPredictor):
    def __init__(self) -> None:
        self.client = NBADataClient()
        self.injury_client = InjuryDataClient()
        self.elo_system = EloRatingSystem("nba", k_factor=26.0, home_advantage=60.0)

    def predict(self, args: Namespace) -> list[PredictionResult]:
        target_date = parse_target_date(args.date)
        games = self.client.get_tomorrow_schedule(target_date)
        if not games:
            print("No NBA games tomorrow" if target_date == dt.date.today() + dt.timedelta(days=1) else f"No NBA games on {target_date}")
            if getattr(args, "injuries", False):
                self.print_injury_report([])
            return []
        self.client.refresh_completed_games_cache(season_from_date(target_date))
        self.elo_system = build_nba_elo_system(target_date)
        results = [self._predict_game(game) for game in games]
        if getattr(args, "injuries", False):
            self.print_injury_report(games)
        for result in results:
            self._save_nba_prediction(result)
        return results

    def _predict_game(self, game: ScheduledGame) -> PredictionResult:
        home_games = self.client.get_team_recent_games(game.home_team, game.date, limit=40)
        away_games = self.client.get_team_recent_games(game.away_team, game.date, limit=40)
        if len(home_games) < 10:
            LOGGER.warning("WARNING: missing data for %s recent NBA history", game.home_team)
        if len(away_games) < 10:
            LOGGER.warning("WARNING: missing data for %s recent NBA history", game.away_team)
        home_injury_impact = self.injury_client.get_team_impact(game.home_team)
        away_injury_impact = self.injury_client.get_team_impact(game.away_team)
        elo_snapshot = self.elo_system.snapshot(game.home_team, game.away_team)
        home_momentum = nba_momentum(game.home_team, home_games)
        away_momentum = nba_momentum(game.away_team, away_games)
        home_fatigue = calculate_nba_fatigue(game.home_team, home_games, game.date, game.home_team, True)
        away_fatigue = calculate_nba_fatigue(game.away_team, away_games, game.date, game.home_team, False)
        home_metrics = build_team_metrics(game.home_team, home_games, game.date, home_injury_impact.as_model_injuries())
        away_metrics = build_team_metrics(game.away_team, away_games, game.date, away_injury_impact.as_model_injuries())
        prediction = predict_game(game, home_metrics, away_metrics)
        adjusted_prediction = apply_lineup_impact(prediction, home_injury_impact, away_injury_impact)
        common_adjustment = apply_common_nba_adjustments(
            adjusted_prediction["home_probability"],
            adjusted_prediction["confidence"],
            game.home_team,
            game.away_team,
            elo_snapshot,
            home_momentum,
            away_momentum,
            home_fatigue,
            away_fatigue,
            home_injury_impact.team_injury_penalty,
            away_injury_impact.team_injury_penalty,
        )
        score_edge = (
            0.38 * (home_momentum.momentum_score - away_momentum.momentum_score)
            + 0.30 * (away_fatigue.fatigue_score - home_fatigue.fatigue_score)
            + 0.012 * elo_snapshot.elo_diff
        )
        home_score = int(round(clamp(float(adjusted_prediction["home_score"]) + score_edge, 82.0, 145.0)))
        away_score = int(round(clamp(float(adjusted_prediction["away_score"]) - score_edge, 82.0, 145.0)))
        predicted_winner = game.home_team if common_adjustment.home_probability >= common_adjustment.away_probability else game.away_team
        key_factors = list(adjusted_prediction["key_factors"])
        key_factors.extend(common_adjustment.key_factors)
        key_factors.extend(ai_explain(predicted_winner, common_adjustment.key_factors, common_adjustment.risk_factors))
        risk_factors = list(adjusted_prediction["risk_factors"])
        risk_factors.extend(common_adjustment.risk_factors)
        return PredictionResult(
            sport="nba",
            match=f"{game.away_team} at {game.home_team}",
            prediction_date=game.date,
            home_team=game.home_team,
            away_team=game.away_team,
            predicted_winner=predicted_winner,
            win_probability_home=common_adjustment.home_probability,
            win_probability_away=common_adjustment.away_probability,
            draw_probability=None,
            predicted_score=f"{game.home_team} {home_score} - {away_score} {game.away_team}",
            confidence=common_adjustment.confidence,
            key_factors=key_factors,
            risk_factors=risk_factors,
        )

    def _save_nba_prediction(self, result: PredictionResult) -> None:
        row = result.to_row()
        append_csv_row(NBA_PREDICTIONS_CSV, row, list(row.keys()))

    def print_injury_report(self, games: list[ScheduledGame]) -> None:
        if games:
            teams = sorted({team for game in games for team in (game.home_team, game.away_team)})
        else:
            teams = self.injury_client.cached_team_names()
        if not teams:
            print("\nNBA injury impact")
            print("=" * 88)
            print("WARNING: no injury cache available and no scheduled NBA teams to query.")
            print("=" * 88)
            return
        impacts = [self.injury_client.get_team_impact(team) for team in teams]
        impacts.sort(key=lambda item: abs(item.team_injury_penalty), reverse=True)
        print("\nNBA injury impact")
        print("=" * 88)
        for impact in impacts[:20]:
            print(
                f"{impact.team}: team_injury_penalty={impact.team_injury_penalty:.1f}, "
                f"missing_starters={impact.missing_starters}, source={impact.source}"
            )
            for item in sorted(impact.injuries, key=lambda injury: injury.weighted_impact, reverse=True)[:5]:
                starter = "starter" if item.is_projected_starter else "rotation"
                print(
                    f"  - {item.player}: {item.status}, impact={item.weighted_impact:.1f}, "
                    f"{starter}, injury={item.injury}"
                )
            if not impact.injuries:
                print("  - WARNING: no confirmed injury records available")
        print("=" * 88)

    def backtest(self, args: Namespace) -> dict[str, object]:
        from .nba_backtest import run_nba_backtest

        return run_nba_backtest(args)


def apply_lineup_impact(prediction, home_injuries: TeamInjuryImpact, away_injuries: TeamInjuryImpact) -> dict[str, object]:
    home_penalty = abs(home_injuries.team_injury_penalty)
    away_penalty = abs(away_injuries.team_injury_penalty)
    penalty_edge = away_penalty - home_penalty
    probability_shift = max(-0.12, min(0.12, penalty_edge * 0.012))
    home_probability = max(0.05, min(0.95, prediction.home_win_probability + probability_shift))
    away_probability = 1.0 - home_probability
    home_score = int(round(max(82, prediction.predicted_home_score - home_penalty * 0.7)))
    away_score = int(round(max(82, prediction.predicted_away_score - away_penalty * 0.7)))
    predicted_winner = prediction.game.home_team if home_probability >= away_probability else prediction.game.away_team
    confidence = prediction.confidence
    if home_injuries.uncertainty_count or away_injuries.uncertainty_count:
        confidence = lower_confidence(confidence)
    if home_penalty >= 6 or away_penalty >= 6:
        confidence = lower_confidence(confidence)
    key_factors = list(prediction.reasons)
    key_factors.extend(injury_key_factors(home_injuries, away_injuries))
    risk_factors = list(prediction.risks)
    risk_factors.extend(injury_risk_factors(home_injuries, away_injuries))
    return {
        "home_probability": home_probability,
        "away_probability": away_probability,
        "home_score": home_score,
        "away_score": away_score,
        "predicted_winner": predicted_winner,
        "confidence": confidence,
        "key_factors": key_factors,
        "risk_factors": risk_factors,
    }


def lower_confidence(value: str) -> str:
    if value == "High":
        return "Medium"
    if value == "Medium":
        return "Low"
    return value


def injury_key_factors(home: TeamInjuryImpact, away: TeamInjuryImpact) -> list[str]:
    factors: list[str] = []
    for impact in (home, away):
        if not impact.injuries:
            factors.append(f"{impact.team} no confirmed injury records available; penalty 0.0")
            continue
        top = sorted(impact.injuries, key=lambda item: item.weighted_impact, reverse=True)[:3]
        names = ", ".join(f"{item.player} {item.status} ({item.weighted_impact:.1f})" for item in top)
        factors.append(
            f"{impact.team} injury impact: team_injury_penalty={impact.team_injury_penalty:.1f}, "
            f"missing_starters={impact.missing_starters}, top={names}"
        )
    return factors


def injury_risk_factors(home: TeamInjuryImpact, away: TeamInjuryImpact) -> list[str]:
    risks: list[str] = []
    for impact in (home, away):
        if impact.source == "missing":
            risks.append(f"WARNING: missing injury data for {impact.team}")
        elif impact.source == "cache":
            risks.append(f"{impact.team} injury API failed; using injury_cache.csv fallback")
        if impact.uncertainty_count:
            risks.append(f"{impact.team} has {impact.uncertainty_count} questionable/probable/doubtful player(s)")
        if impact.missing_starters:
            risks.append(f"{impact.team} missing_starters={impact.missing_starters}")
    return risks


def build_nba_elo_system(target_date: dt.date) -> EloRatingSystem:
    elo = EloRatingSystem("nba", k_factor=26.0, home_advantage=60.0)
    matches: list[EloMatch] = []
    if not NBA_GAMES_CSV.exists():
        LOGGER.warning("WARNING: missing NBA Elo source file %s", NBA_GAMES_CSV)
        return elo
    try:
        with NBA_GAMES_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                home_score = safe_int(row.get("home_score"))
                away_score = safe_int(row.get("away_score"))
                if home_score is None or away_score is None:
                    continue
                try:
                    game_date = dt.date.fromisoformat(str(row.get("date") or "")[:10])
                except ValueError:
                    continue
                if game_date >= target_date:
                    continue
                matches.append(
                    EloMatch(
                        date=game_date,
                        sport="nba",
                        home_team=str(row.get("home_team") or ""),
                        away_team=str(row.get("away_team") or ""),
                        home_score=float(home_score),
                        away_score=float(away_score),
                    )
                )
    except Exception as exc:
        LOGGER.warning("WARNING: failed building NBA Elo ratings: %s", exc)
        return elo
    if not matches:
        LOGGER.warning("WARNING: no completed NBA matches available for Elo before %s", target_date)
        return elo
    elo.rebuild(matches, save=True)
    return elo
