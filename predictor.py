from __future__ import annotations

from dataclasses import dataclass

try:
    from .core.utils import clamp, names_match
    from .elo import EloSnapshot
    from .fatigue import FatigueProfile
    from .momentum import MomentumProfile
except ImportError:
    from core.utils import clamp, names_match
    from elo import EloSnapshot
    from fatigue import FatigueProfile
    from momentum import MomentumProfile


NBA_HOME_ADVANTAGE = {
    "Denver Nuggets": 4.4,
    "Utah Jazz": 3.4,
    "Golden State Warriors": 3.0,
    "Boston Celtics": 2.9,
    "Los Angeles Lakers": 2.2,
}

FOOTBALL_HOME_ADVANTAGE = {
    "Mexico": 0.28,
    "United States": 0.22,
    "Canada": 0.20,
    "Brazil": 0.18,
    "Spain": 0.16,
    "France": 0.16,
    "Germany": 0.16,
}


@dataclass(frozen=True)
class AdjustmentReport:
    """Unified adjustment payload used by NBA and football plugins."""

    home_probability: float
    away_probability: float
    confidence: str
    key_factors: list[str]
    risk_factors: list[str]


def home_advantage_score(sport: str, home_team: str) -> float:
    table = NBA_HOME_ADVANTAGE if sport == "nba" else FOOTBALL_HOME_ADVANTAGE
    default = 2.1 if sport == "nba" else 0.12
    for team, score in table.items():
        if names_match(team, home_team):
            return score
    return default


def blend_probability(model_probability: float, elo_probability: float, elo_weight: float) -> float:
    return clamp((1.0 - elo_weight) * model_probability + elo_weight * elo_probability, 0.03, 0.97)


def apply_common_nba_adjustments(
    base_home_probability: float,
    confidence: str,
    home_team: str,
    away_team: str,
    elo: EloSnapshot,
    home_momentum: MomentumProfile,
    away_momentum: MomentumProfile,
    home_fatigue: FatigueProfile,
    away_fatigue: FatigueProfile,
    home_injury_penalty: float,
    away_injury_penalty: float,
) -> AdjustmentReport:
    home_adv = home_advantage_score("nba", home_team)
    fatigue_edge = away_fatigue.fatigue_score - home_fatigue.fatigue_score
    momentum_edge = home_momentum.momentum_score - away_momentum.momentum_score
    injury_edge = abs(away_injury_penalty) - abs(home_injury_penalty)
    model_probability = blend_probability(base_home_probability, elo.elo_win_probability, 0.22)
    shift = 0.006 * momentum_edge + 0.007 * fatigue_edge + 0.006 * injury_edge + 0.003 * (home_adv - 2.1)
    raw_home_probability = calibrate_binary_probability(model_probability + shift)
    cap_info = nba_confidence_cap(
        raw_home_probability,
        elo.elo_diff,
        home_team,
        away_team,
        home_momentum,
        away_momentum,
        home_fatigue,
        away_fatigue,
        home_injury_penalty,
        away_injury_penalty,
    )
    home_probability = apply_probability_cap(raw_home_probability, cap_info["confidence_cap"])
    confidence = adjust_confidence(confidence, max(home_probability, 1.0 - home_probability), fatigue_edge, momentum_edge)
    if cap_info["confidence_cap"] <= 0.70 and confidence == "High":
        confidence = "Medium"
    key_factors = [
        f"home_elo={elo.home_elo:.0f}, away_elo={elo.away_elo:.0f}, elo_diff={elo.elo_diff:+.0f}, "
        f"elo_win_probability={elo.elo_win_probability:.3f}",
        f"home_momentum={home_momentum.recent_form}, away_momentum={away_momentum.recent_form}, "
        f"momentum_score_edge={momentum_edge:+.1f}",
        f"home_fatigue_score={home_fatigue.fatigue_score:.1f}, away_fatigue_score={away_fatigue.fatigue_score:.1f}, "
        f"rest_advantage={fatigue_edge:+.1f}, travel_penalty_home={home_fatigue.travel_penalty:.1f}, "
        f"travel_penalty_away={away_fatigue.travel_penalty:.1f}",
        f"home_advantage_score={home_adv:.1f}",
        f"confidence_cap={cap_info['confidence_cap']:.2f}, upset_risk_score={cap_info['upset_risk_score']:.2f}",
    ]
    risk_factors: list[str] = []
    if home_fatigue.warning:
        risk_factors.append(home_fatigue.warning)
    if away_fatigue.warning:
        risk_factors.append(away_fatigue.warning)
    if home_fatigue.back_to_back:
        risk_factors.append(f"{home_team} back-to-back fatigue risk.")
    if away_fatigue.back_to_back:
        risk_factors.append(f"{away_team} back-to-back fatigue risk.")
    if abs(home_injury_penalty) >= 5:
        risk_factors.append(f"{home_team} has high injury impact ({home_injury_penalty:.1f}).")
    if abs(away_injury_penalty) >= 5:
        risk_factors.append(f"{away_team} has high injury impact ({away_injury_penalty:.1f}).")
    risk_factors.extend(cap_info["favorite_risk_reason"])
    return AdjustmentReport(home_probability, 1.0 - home_probability, confidence, key_factors, risk_factors)


def adjust_confidence(confidence: str, top_probability: float, fatigue_edge: float, momentum_edge: float) -> str:
    if top_probability >= 0.86 and abs(fatigue_edge) >= 2.0 and abs(momentum_edge) >= 2.0:
        return "High"
    if top_probability < 0.60:
        return "Low"
    if confidence == "Low" and top_probability >= 0.66:
        return "Medium"
    if confidence == "High" and top_probability < 0.72:
        return "Medium"
    return confidence


def ai_explain(favorite: str, key_factors: list[str], risk_factors: list[str]) -> list[str]:
    explanation = [f"AI analysis: model leans {favorite} after combining Elo, form, availability, rest, and venue context."]
    explanation.extend(key_factors[:4])
    if risk_factors:
        explanation.append(f"Main risk: {risk_factors[0]}")
    return explanation


def calibrate_binary_probability(probability: float) -> float:
    """Shrink extreme live probabilities until enough calibration evidence exists."""

    probability = clamp(probability, 0.05, 0.95)
    return clamp(0.5 + (probability - 0.5) * 0.86, 0.08, 0.92)


def nba_confidence_cap(
    home_probability: float,
    elo_diff: float,
    home_team: str,
    away_team: str,
    home_momentum: MomentumProfile,
    away_momentum: MomentumProfile,
    home_fatigue: FatigueProfile,
    away_fatigue: FatigueProfile,
    home_injury_penalty: float,
    away_injury_penalty: float,
) -> dict[str, object]:
    favorite_is_home = home_probability >= 0.5
    favorite = home_team if favorite_is_home else away_team
    favorite_injury = abs(home_injury_penalty if favorite_is_home else away_injury_penalty)
    favorite_fatigue = home_fatigue if favorite_is_home else away_fatigue
    favorite_momentum = home_momentum if favorite_is_home else away_momentum
    cap = 0.80
    risk = 0.0
    reasons: list[str] = []
    if max(home_probability, 1.0 - home_probability) > 0.75 and favorite_injury >= 3.0:
        cap = min(cap, 0.68)
        risk += 0.24
        reasons.append(f"favorite_risk_reason={favorite} has injury risk despite high favorite probability.")
    if favorite_fatigue.back_to_back:
        cap = min(cap, 0.66)
        risk += 0.22
        reasons.append(f"favorite_risk_reason={favorite} is on a back-to-back.")
    if not favorite_is_home:
        cap = min(cap, 0.72)
        risk += 0.14
        reasons.append(f"favorite_risk_reason={favorite} is an away favorite.")
    if momentum_is_unstable(favorite_momentum):
        cap = min(cap, 0.70)
        risk += 0.16
        reasons.append(f"favorite_risk_reason={favorite} has unstable momentum.")
    if abs(elo_diff) < 90:
        cap = min(cap, 0.70)
        risk += 0.18
        reasons.append("favorite_risk_reason=Elo gap is not large enough for high confidence.")
    return {"confidence_cap": cap, "upset_risk_score": min(1.0, risk), "favorite_risk_reason": reasons}


def apply_probability_cap(home_probability: float, cap: float) -> float:
    if home_probability >= 0.5:
        return min(home_probability, cap)
    return max(home_probability, 1.0 - cap)


def momentum_is_unstable(momentum: MomentumProfile) -> bool:
    return "L" in momentum.recent_form[:2] or abs(momentum.momentum_score) < 1.0
