from __future__ import annotations

import unittest

from sports.football.draw_booster import draw_booster


class DrawBoosterTests(unittest.TestCase):
    def test_draw_booster_raises_close_low_goal_match_to_30_percent(self) -> None:
        result = draw_booster(
            {"HOME_WIN": 0.48, "DRAW": 0.20, "AWAY_WIN": 0.32},
            elo_diff=35,
            avg_goals_scored=1.2,
            avg_goals_conceded=1.3,
        )

        self.assertAlmostEqual(result["DRAW"], 0.30, places=6)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertGreater(result["HOME_WIN"], result["AWAY_WIN"])

    def test_draw_booster_keeps_probabilities_when_conditions_do_not_match(self) -> None:
        result = draw_booster(
            {"HOME_WIN": 0.48, "DRAW": 0.20, "AWAY_WIN": 0.32},
            elo_diff=80,
            avg_goals_scored=1.2,
            avg_goals_conceded=1.3,
        )

        self.assertAlmostEqual(result["DRAW"], 0.20, places=6)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
