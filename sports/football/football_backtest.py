from __future__ import annotations

import logging
from argparse import Namespace

try:
    from ...config import FOOTBALL_DATA_DIR
    from ...core.utils import write_csv
except ImportError:
    from config import FOOTBALL_DATA_DIR
    from core.utils import write_csv

from .football_data import load_matches
from .football_model import predict_football_match


LOGGER = logging.getLogger("sports_predictor")


def run_football_backtest(args: Namespace) -> dict[str, object]:
    matches = load_matches()
    if len(matches) < 120:
        LOGGER.warning("WARNING: missing data for football backtest")
        return {"sport": "football", "games": 0, "accuracy": 0.0}
    limit = int(getattr(args, "limit", 100) or 100)
    test_matches = matches[-limit:]
    correct = 0
    top1_score_hits = 0
    top3_score_hits = 0
    rows: list[dict[str, object]] = []
    for match in test_matches:
        history = [item for item in matches if item.date < match.date]
        if len(history) < 100:
            continue
        prediction = predict_football_match(history, match.home_team, match.away_team, "BACKTEST", match.date)
        actual = match.home_team if match.home_goals > match.away_goals else match.away_team if match.away_goals > match.home_goals else "Draw"
        is_correct = prediction.predicted_winner == actual
        correct += int(is_correct)
        actual_label = "DRAW" if actual == "Draw" else "HOME_WIN" if actual == match.home_team else "AWAY_WIN"
        predicted_total = predicted_total_from_score(prediction.predicted_score)
        top_scores = extract_top_scores(prediction.key_factors)
        actual_score_pair = (match.home_goals, match.away_goals)
        top1_score_hit = bool(top_scores and top_scores[0] == actual_score_pair)
        top3_score_hit = actual_score_pair in top_scores[:3]
        top1_score_hits += int(top1_score_hit)
        top3_score_hits += int(top3_score_hit)
        actual_total = match.home_goals + match.away_goals
        actual_over = actual_total > 2.5
        predicted_over = predicted_total is not None and predicted_total > 2.5
        features = extract_feature_edges(prediction.key_factors)
        rows.append(
            {
                "date": match.date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "predicted_winner": prediction.predicted_winner,
                "actual_winner": actual,
                "correct": is_correct,
                "predicted_probability": round(
                    max(
                        prediction.win_probability_home or 0.0,
                        prediction.win_probability_away or 0.0,
                        prediction.draw_probability or 0.0,
                    ),
                    4,
                ),
                "actual_result": int(is_correct),
                "actual_label": actual_label,
                "is_draw": actual_label == "DRAW",
                "predicted_draw": prediction.predicted_winner == "Draw",
                "home_win_probability": round(prediction.win_probability_home or 0.0, 4),
                "away_win_probability": round(prediction.win_probability_away or 0.0, 4),
                "draw_probability": round(prediction.draw_probability or 0.0, 4),
                "confidence_value": round(
                    max(
                        prediction.win_probability_home or 0.0,
                        prediction.win_probability_away or 0.0,
                        prediction.draw_probability or 0.0,
                    ),
                    4,
                ),
                "confidence": prediction.confidence,
                "predicted_score": prediction.predicted_score,
                "top_score_1": format_score_pair(top_scores, 0),
                "top_score_2": format_score_pair(top_scores, 1),
                "top_score_3": format_score_pair(top_scores, 2),
                "top1_score_hit": top1_score_hit,
                "top3_score_hit": top3_score_hit,
                "actual_score": f"{match.home_goals}-{match.away_goals}",
                "actual_home_goals": match.home_goals,
                "actual_away_goals": match.away_goals,
                "predicted_total_goals": "" if predicted_total is None else predicted_total,
                "actual_total_goals": actual_total,
                "predicted_over_2_5": predicted_over,
                "actual_over_2_5": actual_over,
                "over_under_correct": predicted_over == actual_over if predicted_total is not None else False,
                "elo_difference": features["elo_difference"],
                "recent_form_edge": features["recent_form_edge"],
                "home_advantage_edge": features["home_advantage_edge"],
            }
        )
    output = FOOTBALL_DATA_DIR / "football_backtest_results.csv"
    write_csv(output, rows)
    games = len(rows)
    return {
        "sport": "football",
        "games": games,
        "accuracy": correct / games if games else 0.0,
        "top1_score_hit_rate": top1_score_hits / games if games else 0.0,
        "top3_score_hit_rate": top3_score_hits / games if games else 0.0,
        "output": str(output),
    }


def predicted_total_from_score(value: str) -> int | None:
    import re

    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    return numbers[-2] + numbers[-1]


def extract_top_scores(factors: list[str]) -> list[tuple[int, int]]:
    import re

    joined = " | ".join(factors)
    match = re.search(r"most_likely_scores=([^|]+)", joined)
    if not match:
        return []
    scores: list[tuple[int, int]] = []
    for item in match.group(1).split(","):
        parts = item.strip().split(":")
        if len(parts) < 2:
            continue
        try:
            scores.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return scores[:3]


def format_score_pair(scores: list[tuple[int, int]], index: int) -> str:
    if index >= len(scores):
        return ""
    return f"{scores[index][0]}:{scores[index][1]}"


def extract_feature_edges(factors: list[str]) -> dict[str, float]:
    import re

    joined = " | ".join(factors)
    elo_match = re.search(r"elo_diff=([+-]?\d+(?:\.\d+)?)", joined)
    momentum_match = re.search(r"momentum_score_edge=([+-]?\d+(?:\.\d+)?)", joined)
    home_adv_match = re.search(r"home_advantage_score=([+-]?\d+(?:\.\d+)?)", joined)
    return {
        "elo_difference": float(elo_match.group(1)) if elo_match else 0.0,
        "recent_form_edge": float(momentum_match.group(1)) if momentum_match else 0.0,
        "home_advantage_edge": float(home_adv_match.group(1)) if home_adv_match else 0.0,
    }
