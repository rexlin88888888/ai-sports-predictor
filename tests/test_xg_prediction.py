from __future__ import annotations

import datetime as dt
import unittest

from sports.football.football_data import FootballMatch
from sports.football.football_model import estimate_xg, predict_football_match, top_score_probabilities


class XGPredictionTests(unittest.TestCase):
    def test_xg_prediction_uses_weighted_elo_attack_and_defence(self) -> None:
        home_stats = {"goals_for": 1.8, "goals_against": 0.8}
        away_stats = {"goals_for": 1.0, "goals_against": 1.5}

        xg_home, xg_away = estimate_xg(
            home_stats=home_stats,
            away_stats=away_stats,
            elo_diff=80,
            weighted_elo_home=1.2,
            weighted_elo_away=-0.4,
            rank_edge=0.2,
            momentum_edge=0.5,
            home_advantage=0.15,
        )

        self.assertGreater(xg_home, xg_away)
        self.assertGreater(xg_home, 1.5)
        self.assertGreaterEqual(xg_away, 0.15)

    def test_top_score_probabilities_returns_three_sorted_scores(self) -> None:
        scores = top_score_probabilities(1.7, 0.9)

        self.assertEqual(len(scores), 3)
        self.assertGreaterEqual(scores[0][2], scores[1][2])
        self.assertGreaterEqual(scores[1][2], scores[2][2])

    def test_prediction_score_is_based_on_top_xg_score(self) -> None:
        matches = [
            FootballMatch(dt.date(2026, 1, 10), "Testland", "Examplestan", 2, 0, "Friendly"),
            FootballMatch(dt.date(2026, 1, 9), "Testland", "Examplestan", 2, 1, "Friendly"),
            FootballMatch(dt.date(2026, 1, 8), "Examplestan", "Testland", 0, 1, "Friendly"),
            FootballMatch(dt.date(2026, 1, 7), "Examplestan", "Testland", 1, 1, "Friendly"),
        ]

        result = predict_football_match(
            matches,
            "Testland",
            "Examplestan",
            "WORLD_CUP",
            dt.date(2026, 2, 1),
            "test",
        )
        score_line = next(item for item in result.key_factors if item.startswith("most_likely_scores="))
        top_home, top_away, _ = score_line.split("=", 1)[1].split(",", 1)[0].split(":")

        self.assertIn(f"Testland {top_home} - {top_away} Examplestan", result.predicted_score)


if __name__ == "__main__":
    unittest.main()
