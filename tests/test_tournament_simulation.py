from __future__ import annotations

import unittest

from sports.football.tournament_simulation import TeamProfile, TournamentFixture, simulate_tournament


class TournamentSimulationTests(unittest.TestCase):
    def test_simulation_outputs_probabilities_for_full_field(self) -> None:
        groups = "ABCDEFGHIJKL"
        profiles: dict[str, TeamProfile] = {}
        group_fixtures: list[TournamentFixture] = []
        for group_index, group in enumerate(groups):
            teams = [f"Team {group}{seed}" for seed in range(1, 5)]
            for seed, team in enumerate(teams):
                profiles[team] = TeamProfile(
                    team=team,
                    group=group,
                    elo=1800 - group_index * 12 - seed * 22,
                    goals_for=1.45 - seed * 0.05,
                    goals_against=0.95 + seed * 0.05,
                    data_source="test",
                )
            for home_index in range(4):
                for away_index in range(home_index + 1, 4):
                    group_fixtures.append(
                        TournamentFixture(
                            home_team=teams[home_index],
                            away_team=teams[away_index],
                            match_time_utc=f"2026-06-{10 + group_index:02d}T00:00:00",
                            stage="Group Stage",
                            group_name=f"Group {group}",
                        )
                    )
        knockout_fixtures = build_knockout_fixtures()
        frame = simulate_tournament(group_fixtures, knockout_fixtures, profiles, iterations=25, seed=7)
        self.assertEqual(len(frame), 48)
        self.assertAlmostEqual(float(frame["champion_probability"].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(frame["final_probability"].sum()), 2.0, places=6)
        self.assertTrue((frame["round_of_32_probability"] >= 0).all())
        self.assertTrue((frame["round_of_32_probability"] <= 1).all())


def build_knockout_fixtures() -> list[TournamentFixture]:
    def fixture(home: str, away: str, stage: str) -> TournamentFixture:
        return TournamentFixture(home, away, "2026-07-01T00:00:00", stage, "")

    return [
        fixture("2A", "2B", "Round of 32"),
        fixture("1C", "2F", "Round of 32"),
        fixture("1E", "3A/B/C/D/F", "Round of 32"),
        fixture("1F", "2C", "Round of 32"),
        fixture("2E", "2I", "Round of 32"),
        fixture("1I", "3C/D/F/G/H", "Round of 32"),
        fixture("1A", "3C/E/F/H/I", "Round of 32"),
        fixture("1L", "3E/H/I/J/K", "Round of 32"),
        fixture("1G", "3A/E/H/I/J", "Round of 32"),
        fixture("1D", "3B/E/F/I/J", "Round of 32"),
        fixture("1H", "2J", "Round of 32"),
        fixture("2K", "2L", "Round of 32"),
        fixture("1B", "3E/F/G/I/J", "Round of 32"),
        fixture("2D", "2G", "Round of 32"),
        fixture("1J", "2H", "Round of 32"),
        fixture("1K", "3D/E/I/J/L", "Round of 32"),
        fixture("W73", "W75", "Round of 16"),
        fixture("W74", "W77", "Round of 16"),
        fixture("W76", "W78", "Round of 16"),
        fixture("W79", "W80", "Round of 16"),
        fixture("W83", "W84", "Round of 16"),
        fixture("W81", "W82", "Round of 16"),
        fixture("W86", "W88", "Round of 16"),
        fixture("W85", "W87", "Round of 16"),
        fixture("W89", "W90", "Quarter-final"),
        fixture("W93", "W94", "Quarter-final"),
        fixture("W91", "W92", "Quarter-final"),
        fixture("W95", "W96", "Quarter-final"),
        fixture("W97", "W98", "Semi-final"),
        fixture("W99", "W100", "Semi-final"),
        fixture("W101", "W102", "Final"),
    ]


if __name__ == "__main__":
    unittest.main()
