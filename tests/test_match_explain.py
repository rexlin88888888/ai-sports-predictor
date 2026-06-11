from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from core.match_explain import explain_match_json


class MatchExplainTests(unittest.TestCase):
    def test_explain_match_json_returns_top_factors_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["match", "home_team", "away_team", "key_factors", "data_source"])
                writer.writeheader()
                writer.writerow(
                    {
                        "match": "Mexico vs South Africa",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "data_source": "ESPN",
                        "key_factors": (
                            "top_factors=elo_diff:62,recent_attack:23,defence_form:18 | "
                            "data_source_home_elo=ELO, 2026-06-11 | "
                            "data_source_home_recent=ESPN, 5 actual matches"
                        ),
                    }
                )

            result = explain_match_json("Mexico_vs_South_Africa", path)

        self.assertEqual(
            result,
            {
                "top_factors": [
                    {"name": "elo_diff", "value": 62.0},
                    {"name": "recent_attack", "value": 23.0},
                    {"name": "defence_form", "value": 18.0},
                ],
                "data_source": {"elo": "ELO", "form": "ESPN"},
            },
        )

    def test_explain_match_json_defaults_to_estimated_when_match_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = explain_match_json("missing_match", Path(temp_dir) / "predictions.csv")

        self.assertEqual(result["top_factors"], [])
        self.assertEqual(result["data_source"], {"elo": "Estimated", "form": "Estimated"})


if __name__ == "__main__":
    unittest.main()
