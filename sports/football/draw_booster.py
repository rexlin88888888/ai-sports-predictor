from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawBoostResult:
    probabilities: dict[str, float]
    applied: bool
    reason: str


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in probabilities.values())
    if total <= 0:
        raise ValueError("football probabilities must be produced by model features before normalization")
    return {key: max(0.0, value) / total for key, value in probabilities.items()}


def draw_booster(
    probabilities: dict[str, float],
    elo_diff: float,
    avg_goals_scored: float,
    avg_goals_conceded: float,
) -> dict[str, float]:
    normalized = normalize_probabilities(probabilities)
    should_boost = abs(elo_diff) < 50 and avg_goals_scored < 1.5 and avg_goals_conceded < 1.5
    if not should_boost or normalized.get("DRAW", 0.0) >= 0.30:
        return normalized

    target_draw = 0.30
    non_draw_total = max(1e-9, normalized["HOME_WIN"] + normalized["AWAY_WIN"])
    home_share = normalized["HOME_WIN"] / non_draw_total
    return normalize_probabilities(
        {
            "HOME_WIN": (1.0 - target_draw) * home_share,
            "DRAW": target_draw,
            "AWAY_WIN": (1.0 - target_draw) * (1.0 - home_share),
        }
    )


def boost_draw_probability(
    probabilities: dict[str, float],
    elo_diff: float,
    avg_goals_scored: float,
    avg_goals_conceded: float,
    is_knockout: bool = False,
) -> DrawBoostResult:
    """OpenAPI: component DrawBooster applies post-model draw calibration.

    The rule is intentionally deterministic: when teams are close on Elo and both
    teams project under 1.5 goals scored/conceded, group-stage style matches get
    at least a 30% draw probability. Remaining home/away probability is
    redistributed proportionally, which matches the requested logistic-style
    normalization without introducing a separate training dependency.
    """

    normalized = normalize_probabilities(probabilities)
    if is_knockout:
        return DrawBoostResult(normalized, False, "knockout_stage")
    close_low_goal_match = abs(elo_diff) < 50 and avg_goals_scored < 1.5 and avg_goals_conceded < 1.5
    if not close_low_goal_match or normalized.get("DRAW", 0.0) >= 0.30:
        return DrawBoostResult(normalized, False, "not_required")

    return DrawBoostResult(
        draw_booster(normalized, elo_diff, avg_goals_scored, avg_goals_conceded),
        True,
        "close_low_goal_match",
    )
