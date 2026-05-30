from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    data_source: str = "unknown"

    def to_row(self) -> dict[str, Any]:
        created_at = dt.datetime.now().isoformat(timespec="seconds")
        return {
            "run_timestamp": created_at,
            "created_at": created_at,
            "sport": self.sport,
            "match": self.match,
            "prediction_date": self.prediction_date.isoformat(),
            "date": self.prediction_date.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_winner": self.predicted_winner,
            "predicted_result": self.predicted_winner,
            "win_probability_home": "" if self.win_probability_home is None else round(self.win_probability_home, 4),
            "win_probability_away": "" if self.win_probability_away is None else round(self.win_probability_away, 4),
            "home_win_probability": "" if self.win_probability_home is None else round(self.win_probability_home, 4),
            "away_win_probability": "" if self.win_probability_away is None else round(self.win_probability_away, 4),
            "draw_probability": "" if self.draw_probability is None else round(self.draw_probability, 4),
            "predicted_score": self.predicted_score,
            "actual_result": "",
            "confidence": self.confidence,
            "key_factors": " | ".join(self.key_factors),
            "risk_factors": " | ".join(self.risk_factors),
            "model_version": current_model_version(),
            "data_source": self.data_source,
        }


PREDICTION_FIELDNAMES = [
    "run_timestamp",
    "created_at",
    "sport",
    "match",
    "prediction_date",
    "date",
    "home_team",
    "away_team",
    "predicted_winner",
    "predicted_result",
    "win_probability_home",
    "win_probability_away",
    "home_win_probability",
    "away_win_probability",
    "draw_probability",
    "predicted_score",
    "actual_result",
    "confidence",
    "key_factors",
    "risk_factors",
    "model_version",
    "data_source",
]


def current_model_version() -> str:
    path = Path(__file__).resolve().parents[1] / "model_version.json"
    if not path.exists():
        return "v1.0.0"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("version") or "v1.0.0")
    except Exception:
        return "v1.0.0"
