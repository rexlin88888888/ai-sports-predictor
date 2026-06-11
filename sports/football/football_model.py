from __future__ import annotations

import datetime as dt
import logging
import math
from argparse import Namespace

try:
    from ...config import FOOTBALL_PREDICTIONS_CSV
    from ...core.base_model import SportPredictor
    from ...core.prediction_result import PredictionResult
    from ...core.prediction_store import save_prediction_outputs
    from ...core.utils import append_csv_row, clamp, mean, names_match, parse_target_date
    from ...elo import EloMatch, EloRatingSystem
    from ...momentum import football_momentum
    from ...predictor import ai_explain, blend_probability, home_advantage_score
except ImportError:
    from config import FOOTBALL_PREDICTIONS_CSV
    from core.base_model import SportPredictor
    from core.prediction_result import PredictionResult
    from core.prediction_store import save_prediction_outputs
    from core.utils import append_csv_row, clamp, mean, names_match, parse_target_date
    from elo import EloMatch, EloRatingSystem
    from momentum import football_momentum
    from predictor import ai_explain, blend_probability, home_advantage_score

from .football_data import FootballMatch, load_live_fixtures, load_matches, team_matches


LOGGER = logging.getLogger("sports_predictor")
MAX_EXPECTED_GOALS = 4.0


FIFA_RANKS = {
    "Spain": 1,
    "France": 2,
    "Brazil": 5,
    "Portugal": 6,
    "Belgium": 9,
    "Colombia": 14,
    "Mexico": 16,
    "United States": 17,
    "South Africa": 60,
}


class FootballPredictor(SportPredictor):
    def predict(self, args: Namespace) -> list[PredictionResult]:
        if not args.home or not args.away:
            LOGGER.warning("WARNING: missing data for football prediction: --home and --away are required")
            return []
        matches = load_matches()
        if not matches:
            LOGGER.warning("WARNING: missing data for football historical matches")
            return []
        result = predict_football_match(
            matches,
            args.home,
            args.away,
            args.mode or "FOOTBALL",
            parse_target_date(getattr(args, "date", None)),
            data_source="manual_input",
        )
        self._save_football_prediction(result)
        return [result]

    def _save_football_prediction(self, result: PredictionResult) -> None:
        row = result.to_row()
        append_csv_row(FOOTBALL_PREDICTIONS_CSV, row, list(row.keys()))
        save_prediction_outputs(result)

    def predict_live(self, args: Namespace) -> list[PredictionResult]:
        target_date = parse_target_date(getattr(args, "date", None))
        matches = load_matches()
        if not matches:
            LOGGER.warning("WARNING: missing data for football live mode historical matches")
            return []
        fixtures = load_live_fixtures(target_date)
        if not fixtures:
            print("No Football games scheduled for this date")
            return []
        results = [
            predict_football_match(
                matches,
                fixture.home_team,
                fixture.away_team,
                fixture.mode,
                target_date,
                data_source=fixture.data_source,
            )
            for fixture in fixtures
        ]
        for result in results:
            self._save_football_prediction(result)
        return results

    def backtest(self, args: Namespace) -> dict[str, object]:
        from .football_backtest import run_football_backtest

        return run_football_backtest(args)


def predict_football_match(
    matches: list[FootballMatch],
    home: str,
    away: str,
    mode: str,
    prediction_date: dt.date,
    data_source: str = "unknown",
) -> PredictionResult:
    home_history = team_matches(matches, home, prediction_date)
    away_history = team_matches(matches, away, prediction_date)
    risks: list[str] = []
    if len(home_history) < 10:
        risks.append(f"WARNING: missing data for {home}: only {len(home_history)} historical matches")
    if len(away_history) < 10:
        risks.append(f"WARNING: missing data for {away}: only {len(away_history)} historical matches")
    home_stats = team_stats(home, home_history)
    away_stats = team_stats(away, away_history)
    elo_system = build_football_elo_system(matches, prediction_date)
    elo_snapshot = elo_system.snapshot(home, away, home_advantage=40.0)
    home_momentum = football_momentum(home, home_history)
    away_momentum = football_momentum(away, away_history)
    home_adv = home_advantage_score("football", home)
    momentum_edge = home_momentum.momentum_score - away_momentum.momentum_score
    rank_edge = (rank(away) - rank(home)) / 100.0
    weighted_elo_home = weighted_elo(home_history, home)
    weighted_elo_away = weighted_elo(away_history, away)
    xg_home, xg_away = estimate_xg(
        home_stats=home_stats,
        away_stats=away_stats,
        elo_diff=elo_snapshot.elo_diff,
        weighted_elo_home=weighted_elo_home,
        weighted_elo_away=weighted_elo_away,
        rank_edge=rank_edge,
        momentum_edge=momentum_edge,
        home_advantage=home_adv,
    )
    lambda_home = xg_home
    lambda_away = xg_away
    probs = poisson_probs(xg_home, xg_away)
    probs = apply_football_elo_probability(probs, elo_snapshot.elo_win_probability)
    probs, draw_risk = apply_draw_model(
        probs=probs,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        home_stats=home_stats,
        away_stats=away_stats,
        elo_diff=elo_snapshot.elo_diff,
        mode=mode,
        historical_draw_rate=historical_draw_rate(matches, prediction_date),
    )
    probs = calibrate_football_probabilities(probs)
    likely = max(probs, key=probs.get)
    winner = home if likely == "HOME_WIN" else away if likely == "AWAY_WIN" else "Draw"
    score_probs = top_score_probabilities(xg_home, xg_away)
    projected_home_goals, projected_away_goals = score_probs[0][0], score_probs[0][1]
    confidence = confidence_level(max(probs.values()), risks)
    key_factors = [
        f"{home} recent goals for {home_stats['goals_for']:.2f}, against {home_stats['goals_against']:.2f}",
        f"{away} recent goals for {away_stats['goals_for']:.2f}, against {away_stats['goals_against']:.2f}",
        f"FIFA rank edge feature {rank_edge:+.2f}",
        f"home_elo={elo_snapshot.home_elo:.0f}, away_elo={elo_snapshot.away_elo:.0f}, "
        f"elo_diff={elo_snapshot.elo_diff:+.0f}, elo_win_probability={elo_snapshot.elo_win_probability:.3f}",
        f"weighted_elo_home={weighted_elo_home:+.3f}, weighted_elo_away={weighted_elo_away:+.3f}",
        f"xg_home={xg_home:.2f}, xg_away={xg_away:.2f}",
        "most_likely_scores=" + ",".join(f"{home_goals}:{away_goals}:{probability:.3f}" for home_goals, away_goals, probability in score_probs),
        f"home_momentum={home_momentum.recent_form}, away_momentum={away_momentum.recent_form}, "
        f"momentum_score_edge={momentum_edge:+.1f}",
        f"home_advantage_score={home_adv:.2f}",
        f"draw_probability={probs['DRAW']:.3f}, draw_risk={draw_risk:.3f}, predicted_result={likely}",
    ]
    key_factors.extend(ai_explain(winner, key_factors[-3:], risks))
    return PredictionResult(
        sport="football",
        match=f"{home} vs {away}",
        prediction_date=prediction_date,
        home_team=home,
        away_team=away,
        predicted_winner=winner,
        win_probability_home=probs["HOME_WIN"],
        win_probability_away=probs["AWAY_WIN"],
        draw_probability=probs["DRAW"],
        predicted_score=f"{home} {projected_home_goals} - {projected_away_goals} {away}",
        confidence=confidence,
        key_factors=key_factors,
        risk_factors=risks or ["No major data completeness risks detected."],
        data_source=data_source,
    )


def projected_score(lambda_home: float, lambda_away: float, likely: str) -> tuple[int, int]:
    home_goals = int(round(lambda_home))
    away_goals = int(round(lambda_away))
    if likely == "DRAW":
        goals = int(round((lambda_home + lambda_away) / 2.0))
        return max(0, min(4, goals)), max(0, min(4, goals))
    if likely == "HOME_WIN" and home_goals <= away_goals:
        home_goals = away_goals + 1
    if likely == "AWAY_WIN" and away_goals <= home_goals:
        away_goals = home_goals + 1
    return max(0, min(5, home_goals)), max(0, min(5, away_goals))


def team_stats(team: str, history: list[FootballMatch]) -> dict[str, float]:
    recent = history[:20]
    if not recent:
        return {"goals_for": 1.2, "goals_against": 1.2, "win_rate": 0.5}
    goals_for: list[int] = []
    goals_against: list[int] = []
    wins = 0
    for match in recent:
        if names_match(match.home_team, team):
            gf, ga = match.home_goals, match.away_goals
        else:
            gf, ga = match.away_goals, match.home_goals
        goals_for.append(gf)
        goals_against.append(ga)
        wins += int(gf > ga)
    return {
        "goals_for": mean(goals_for, 1.2),
        "goals_against": mean(goals_against, 1.2),
        "win_rate": wins / len(recent),
    }


def rank(team: str) -> int:
    for key, value in FIFA_RANKS.items():
        if names_match(key, team):
            return value
    return 50


def poisson_probs(lambda_home: float, lambda_away: float, max_goals: int = 7) -> dict[str, float]:
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        hp = poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            p = hp * poisson_pmf(a, lambda_away)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total <= 0:
        return {"HOME_WIN": 0.34, "DRAW": 0.33, "AWAY_WIN": 0.33}
    return {"HOME_WIN": home_win / total, "DRAW": draw / total, "AWAY_WIN": away_win / total}


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def confidence_level(top_probability: float, risks: list[str]) -> str:
    if risks:
        return "Low"
    if top_probability >= 0.70:
        return "High"
    if top_probability >= 0.55:
        return "Medium"
    return "Low"


def apply_football_elo_probability(probs: dict[str, float], elo_home_probability: float) -> dict[str, float]:
    non_draw = probs["HOME_WIN"] + probs["AWAY_WIN"]
    if non_draw <= 0:
        return probs
    current_home_no_draw = probs["HOME_WIN"] / non_draw
    blended_home_no_draw = blend_probability(current_home_no_draw, elo_home_probability, 0.25)
    return {
        "HOME_WIN": non_draw * blended_home_no_draw,
        "DRAW": probs["DRAW"],
        "AWAY_WIN": non_draw * (1.0 - blended_home_no_draw),
    }


def estimate_xg(
    home_stats: dict[str, float],
    away_stats: dict[str, float],
    elo_diff: float,
    weighted_elo_home: float,
    weighted_elo_away: float,
    rank_edge: float = 0.0,
    momentum_edge: float = 0.0,
    home_advantage: float = 0.0,
) -> tuple[float, float]:
    recent_attack_home = home_stats.get("goals_for", 1.2)
    recent_attack_away = away_stats.get("goals_for", 1.2)
    recent_defence_home = home_stats.get("goals_against", 1.2)
    recent_defence_away = away_stats.get("goals_against", 1.2)
    weighted_elo_edge = weighted_elo_home - weighted_elo_away
    xg_home = (
        0.56 * recent_attack_home
        + 0.44 * recent_defence_away
        + 0.16
        + home_advantage
        + 0.0014 * elo_diff
        + 0.050 * weighted_elo_edge
        + 0.060 * momentum_edge
        + 0.25 * rank_edge
    )
    xg_away = (
        0.56 * recent_attack_away
        + 0.44 * recent_defence_home
        - 0.08
        - 0.0010 * elo_diff
        - 0.035 * weighted_elo_edge
        - 0.040 * momentum_edge
        - 0.18 * rank_edge
    )
    return clamp(xg_home, 0.15, MAX_EXPECTED_GOALS), clamp(xg_away, 0.15, MAX_EXPECTED_GOALS)


def apply_draw_model(
    probs: dict[str, float],
    lambda_home: float,
    lambda_away: float,
    home_stats: dict[str, float],
    away_stats: dict[str, float],
    elo_diff: float,
    mode: str,
    historical_draw_rate: float,
) -> tuple[dict[str, float], float]:
    """Independent draw layer for football before final probability calibration."""

    mode_upper = str(mode or "").upper()
    is_knockout = any(token in mode_upper for token in ("KNOCKOUT", "FINAL", "SEMIFINAL", "QUARTER"))
    elo_close = max(0.0, 1.0 - abs(elo_diff) / 260.0)
    expected_total = lambda_home + lambda_away
    low_scoring = max(0.0, 1.0 - expected_total / 3.1)
    defensive_match = max(
        0.0,
        1.0 - ((home_stats["goals_for"] + away_stats["goals_for"]) / 2.8),
    ) + max(
        0.0,
        1.0 - ((home_stats["goals_against"] + away_stats["goals_against"]) / 2.4),
    )
    defensive_match = min(1.0, defensive_match / 2.0)
    score_close = max(0.0, 1.0 - abs(lambda_home - lambda_away) / 0.85)
    rounded_score_draw = round(lambda_home) == round(lambda_away)
    projected_score_gap = abs(lambda_home - lambda_away)
    stage_boost = -0.08 if is_knockout else 0.045
    historical_boost = clamp(historical_draw_rate - 0.22, -0.04, 0.08)
    draw_risk = (
        0.32 * elo_close
        + 0.22 * low_scoring
        + 0.18 * defensive_match
        + 0.22 * score_close
        + (0.12 if rounded_score_draw else 0.0)
        + (0.10 if abs(elo_diff) < 80 else 0.0)
        + (0.12 if projected_score_gap < 0.35 else 0.0)
        + stage_boost
        + historical_boost
    )
    min_draw = 0.10 if is_knockout else 0.18
    max_draw = 0.22 if is_knockout else 0.38
    target_draw = clamp(0.10 + 0.24 * draw_risk + 0.20 * historical_draw_rate, min_draw, max_draw)
    if rounded_score_draw and not is_knockout:
        target_draw = max(target_draw, 0.30)
    if abs(elo_diff) < 80 and projected_score_gap < 0.35 and not is_knockout:
        target_draw = max(target_draw, 0.34)
    non_draw_total = max(1e-9, probs["HOME_WIN"] + probs["AWAY_WIN"])
    home_share = probs["HOME_WIN"] / non_draw_total
    adjusted = {
        "HOME_WIN": (1.0 - target_draw) * home_share,
        "DRAW": target_draw,
        "AWAY_WIN": (1.0 - target_draw) * (1.0 - home_share),
    }
    if not is_knockout:
        adjusted = promote_close_draw(adjusted, rounded_score_draw, projected_score_gap, abs(elo_diff))
    return adjusted, draw_risk


def promote_close_draw(
    probs: dict[str, float],
    rounded_score_draw: bool,
    projected_score_gap: float,
    abs_elo_diff: float,
) -> dict[str, float]:
    max_non_draw = max(probs["HOME_WIN"], probs["AWAY_WIN"])
    close_probability = probs["DRAW"] >= max_non_draw - 0.06
    close_score = projected_score_gap < 0.35
    close_elo = abs_elo_diff < 80
    if not (close_probability and close_score and close_elo):
        return probs
    target_draw = min(0.38, max(probs["DRAW"], max_non_draw + 0.015))
    non_draw_total = max(1e-9, probs["HOME_WIN"] + probs["AWAY_WIN"])
    home_share = probs["HOME_WIN"] / non_draw_total
    return {
        "HOME_WIN": (1.0 - target_draw) * home_share,
        "DRAW": target_draw,
        "AWAY_WIN": (1.0 - target_draw) * (1.0 - home_share),
    }


def calibrate_football_probabilities(probs: dict[str, float]) -> dict[str, float]:
    calibrated = {key: 0.06 + value * 0.88 for key, value in probs.items()}
    total = sum(calibrated.values())
    return {key: value / total for key, value in calibrated.items()}


def weighted_elo(history: list[FootballMatch], team: str | None = None, half_life: float = 4.0) -> float:
    total = 0.0
    total_weight = 0.0
    if half_life <= 0:
        half_life = 4.0
    recent_first = sorted(history, key=lambda match: match.date, reverse=True)[:12]
    for index, match in enumerate(recent_first):
        weight = 0.5 ** (index / half_life)
        if team and names_match(match.away_team, team):
            goal_diff = match.away_goals - match.home_goals
        else:
            goal_diff = match.home_goals - match.away_goals
        total += weight * max(-3.0, min(3.0, goal_diff))
        total_weight += weight
    return total / max(1e-9, total_weight)


def top_score_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 4) -> list[tuple[int, int, float]]:
    outcomes: list[tuple[int, int, float]] = []
    total = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = poisson_pmf(home_goals, lambda_home) * poisson_pmf(away_goals, lambda_away)
            outcomes.append((home_goals, away_goals, probability))
            total += probability
    if total <= 0:
        return [(1, 1, 0.12), (1, 0, 0.10), (2, 1, 0.09)]
    normalized = [(home, away, probability / total) for home, away, probability in outcomes]
    return sorted(normalized, key=lambda item: item[2], reverse=True)[:3]


def historical_draw_rate(matches: list[FootballMatch], prediction_date: dt.date, limit: int = 500) -> float:
    history = [match for match in matches if match.date < prediction_date]
    if not history:
        return 0.26
    recent = history[-limit:]
    if not recent:
        return 0.26
    draws = sum(1 for match in recent if match.home_goals == match.away_goals)
    return clamp(draws / len(recent), 0.18, 0.34)


def build_football_elo_system(matches: list[FootballMatch], prediction_date: dt.date) -> EloRatingSystem:
    elo = EloRatingSystem("football", k_factor=20.0, home_advantage=40.0)
    elo_matches = [
        EloMatch(
            date=match.date,
            sport="football",
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=float(match.home_goals),
            away_score=float(match.away_goals),
        )
        for match in matches
        if match.date < prediction_date
    ]
    if not elo_matches:
        LOGGER.warning("WARNING: no football matches available for Elo before %s", prediction_date)
        return elo
    elo.rebuild(elo_matches, save=True)
    return elo
