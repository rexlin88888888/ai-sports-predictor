from __future__ import annotations

import datetime as dt
import unittest

from sports.football.football_data import FootballMatch
from sports.football.football_model import weighted_elo


class WeightedEloTests(unittest.TestCase):
    def test_weighted_elo_recent_match_has_more_influence_than_old_match(self) -> None:
        recent_win_old_loss = [
            FootballMatch(dt.date(2026, 1, 1), "Mexico", "Old Opponent", 0, 3, "Friendly"),
            FootballMatch(dt.date(2026, 1, 10), "Mexico", "Recent Opponent", 3, 0, "Friendly"),
        ]
        recent_loss_old_win = [
            FootballMatch(dt.date(2026, 1, 1), "Mexico", "Old Opponent", 3, 0, "Friendly"),
            FootballMatch(dt.date(2026, 1, 10), "Mexico", "Recent Opponent", 0, 3, "Friendly"),
        ]

        positive_recent = weighted_elo(recent_win_old_loss, "Mexico", half_life=4.0)
        negative_recent = weighted_elo(recent_loss_old_win, "Mexico", half_life=4.0)

        self.assertGreater(positive_recent, 0.0)
        self.assertLess(negative_recent, 0.0)
        self.assertGreater(positive_recent, abs(negative_recent) * 0.99)


if __name__ == "__main__":
    unittest.main()
