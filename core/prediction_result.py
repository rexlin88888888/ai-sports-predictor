from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionResult:
    sport: str
    match: str
    prediction_date: dt.date
    home_team: str
    away_team: str
    predicted_winner: str
    win_probability_home: float | None
    win_probability_away: float | None
    draw_probability: float | None
    predicted_score: str
    confidence: str
    key_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "sport": self.sport,
            "match": self.match,
            "prediction_date": self.prediction_date.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_winner": self.predicted_winner,
            "win_probability_home": "" if self.win_probability_home is None else round(self.win_probability_home, 4),
            "win_probability_away": "" if self.win_probability_away is None else round(self.win_probability_away, 4),
            "draw_probability": "" if self.draw_probability is None else round(self.draw_probability, 4),
            "predicted_score": self.predicted_score,
            "confidence": self.confidence,
            "key_factors": " | ".join(self.key_factors),
            "risk_factors": " | ".join(self.risk_factors),
        }


PREDICTION_FIELDNAMES = [
    "run_timestamp",
    "sport",
    "match",
    "prediction_date",
    "home_team",
    "away_team",
    "predicted_winner",
    "win_probability_home",
    "win_probability_away",
    "draw_probability",
    "predicted_score",
    "confidence",
    "key_factors",
    "risk_factors",
]

