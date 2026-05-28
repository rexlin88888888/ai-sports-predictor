from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestSummary:
    sport: str
    games: int
    accuracy: float
    average_score_error: float | None = None
    total_points_error: float | None = None

