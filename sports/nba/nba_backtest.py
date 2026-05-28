from __future__ import annotations

import csv
import logging
from argparse import Namespace

try:
    from ...config import NBA_DATA_DIR
    from ...core.utils import write_csv
except ImportError:
    from config import NBA_DATA_DIR
    from core.utils import write_csv

from .nba_history import (  # noqa: E402
    load_completed_games_for_backtest,
    team_history_from_completed,
)
from .nba_data import NBADataClient, ScheduledGame  # noqa: E402
from .nba_scoring_model import build_team_metrics, predict_game  # noqa: E402
from .nba_utils import mean  # noqa: E402


LOGGER = logging.getLogger("sports_predictor")


def run_nba_backtest(args: Namespace) -> dict[str, object]:
    season = getattr(args, "season", None) or "2025-26"
    limit = int(getattr(args, "limit", 100) or 100)
    client = NBADataClient()
    completed = load_completed_games_for_backtest(client, season)
    if not completed:
        LOGGER.warning("WARNING: missing data for NBA backtest")
        return {"sport": "nba", "games": 0, "accuracy": 0.0}
    rows: list[dict[str, object]] = []
    correct = 0
    score_errors: list[float] = []
    total_errors: list[float] = []
    for game in completed[-limit:]:
        home_history = team_history_from_completed(completed, game.home_team, game.date, 40)
        away_history = team_history_from_completed(completed, game.away_team, game.date, 40)
        home_metrics = build_team_metrics(game.home_team, home_history, game.date, [])
        away_metrics = build_team_metrics(game.away_team, away_history, game.date, [])
        scheduled = ScheduledGame(game.game_id, game.date, "Final", game.home_team, game.away_team)
        prediction = predict_game(scheduled, home_metrics, away_metrics)
        is_correct = prediction.predicted_winner == game.winner
        favorite = game.home_team if prediction.home_win_probability >= prediction.away_win_probability else game.away_team
        favorite_won = favorite == game.winner
        predicted_is_favorite = prediction.predicted_winner == favorite
        confidence_value = max(prediction.home_win_probability, prediction.away_win_probability)
        correct += int(is_correct)
        score_error = (
            abs(prediction.predicted_home_score - game.home_score)
            + abs(prediction.predicted_away_score - game.away_score)
        ) / 2.0
        total_error = abs(prediction.predicted_total - (game.home_score + game.away_score))
        score_errors.append(score_error)
        total_errors.append(total_error)
        rows.append(
            {
                "date": game.date.isoformat(),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "predicted_winner": prediction.predicted_winner,
                "actual_winner": game.winner,
                "correct": is_correct,
                "predicted_probability": round(confidence_value, 4),
                "actual_result": int(is_correct),
                "confidence_value": round(confidence_value, 4),
                "favorite": favorite,
                "favorite_won": favorite_won,
                "predicted_is_favorite": predicted_is_favorite,
                "predicted_is_underdog": not predicted_is_favorite,
                "home_win_probability": round(prediction.home_win_probability, 4),
                "away_win_probability": round(prediction.away_win_probability, 4),
                "confidence": prediction.confidence,
                "predicted_home_score": prediction.predicted_home_score,
                "predicted_away_score": prediction.predicted_away_score,
                "actual_home_score": game.home_score,
                "actual_away_score": game.away_score,
                "predicted_total": prediction.predicted_total,
                "actual_total": game.home_score + game.away_score,
                "score_error": round(score_error, 2),
                "total_points_error": round(total_error, 2),
                "recent_form_edge": round(prediction.home_breakdown.recent_form_score - prediction.away_breakdown.recent_form_score, 3),
                "home_advantage_edge": round(prediction.home_breakdown.home_advantage_score - prediction.away_breakdown.home_advantage_score, 3),
                "offense_edge": round(prediction.home_breakdown.offense_score - prediction.away_breakdown.offense_score, 3),
                "defense_edge": round(prediction.home_breakdown.defense_score - prediction.away_breakdown.defense_score, 3),
                "fatigue_edge": round(prediction.home_breakdown.rest_advantage_score - prediction.away_breakdown.rest_advantage_score, 3),
                "injury_edge": round(prediction.home_breakdown.injury_penalty - prediction.away_breakdown.injury_penalty, 3),
                "elo_difference": round(prediction.home_breakdown.team_strength_score - prediction.away_breakdown.team_strength_score, 3),
            }
        )
    output = NBA_DATA_DIR / "nba_backtest_results.csv"
    write_csv(output, rows)
    games = len(rows)
    accuracy = correct / games if games else 0.0
    summary = {
        "sport": "nba",
        "games": games,
        "accuracy": accuracy,
        "average_score_error": mean(score_errors),
        "total_points_error": mean(total_errors),
        "output": str(output),
    }
    return summary
