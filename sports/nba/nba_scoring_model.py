from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

try:
    from .nba_data import ScheduledGame, TeamGame
    from .nba_utils import clamp, mean
except ImportError:
    from nba_data import ScheduledGame, TeamGame
    from nba_utils import clamp, mean


LOGGER = logging.getLogger("ai_sports_predictor")
LEAGUE_AVG_POINTS = 114.0
DEFAULT_TOTAL_LINE = 224.0


@dataclass(frozen=True)
class TeamMetrics:
    team: str
    games_used: int
    recent5_win_pct: float
    recent10_win_pct: float
    season_win_pct: float
    avg_points_for: float
    avg_points_against: float
    recent_off_eff: float
    recent_def_eff: float
    home_win_pct: float
    away_win_pct: float
    home_avg_margin: float
    away_avg_margin: float
    rest_days: int | None
    back_to_back: bool
    injury_penalty: float
    injuries: list[dict[str, str]]
    warnings: list[str]


@dataclass(frozen=True)
class RatingBreakdown:
    team_strength_score: float
    recent_form_score: float
    offense_score: float
    defense_score: float
    home_advantage_score: float
    rest_advantage_score: float
    injury_penalty: float

    @property
    def total(self) -> float:
        return (
            self.team_strength_score
            + self.recent_form_score
            + self.offense_score
            + self.defense_score
            + self.home_advantage_score
            + self.rest_advantage_score
            + self.injury_penalty
        )


@dataclass(frozen=True)
class GamePrediction:
    game: ScheduledGame
    home_metrics: TeamMetrics
    away_metrics: TeamMetrics
    home_breakdown: RatingBreakdown
    away_breakdown: RatingBreakdown
    home_win_probability: float
    away_win_probability: float
    predicted_home_score: int
    predicted_away_score: int
    predicted_total: int
    total_lean: str
    confidence: str
    predicted_winner: str
    reasons: list[str]
    risks: list[str]


KNOWN_INJURY_PENALTIES = {
    "Milwaukee Bucks": -4.0,
    "Washington Wizards": -3.5,
    "Chicago Bulls": -3.0,
    "Oklahoma City Thunder": -1.0,
}


def build_team_metrics(
    team: str,
    games: list[TeamGame],
    target_date: dt.date,
    injuries: list[dict[str, str]] | None = None,
) -> TeamMetrics:
    injuries = injuries or []
    warnings: list[str] = []
    if len(games) < 10:
        warnings.append(f"Only {len(games)} historical games available for {team}; prediction confidence reduced.")
    if not games:
        warnings.append(f"No historical games available for {team}; using neutral defaults.")
        return TeamMetrics(
            team=team,
            games_used=0,
            recent5_win_pct=0.5,
            recent10_win_pct=0.5,
            season_win_pct=0.5,
            avg_points_for=LEAGUE_AVG_POINTS,
            avg_points_against=LEAGUE_AVG_POINTS,
            recent_off_eff=LEAGUE_AVG_POINTS,
            recent_def_eff=LEAGUE_AVG_POINTS,
            home_win_pct=0.5,
            away_win_pct=0.5,
            home_avg_margin=0.0,
            away_avg_margin=0.0,
            rest_days=None,
            back_to_back=False,
            injury_penalty=injury_penalty(team, injuries),
            injuries=injuries,
            warnings=warnings,
        )

    recent5 = games[:5]
    recent10 = games[:10]
    home_games = [game for game in games if game.is_home]
    away_games = [game for game in games if not game.is_home]
    rest_days = max(0, (target_date - games[0].date).days)
    return TeamMetrics(
        team=team,
        games_used=len(games),
        recent5_win_pct=win_pct(recent5),
        recent10_win_pct=win_pct(recent10),
        season_win_pct=win_pct(games),
        avg_points_for=mean([game.team_score for game in games], LEAGUE_AVG_POINTS),
        avg_points_against=mean([game.opponent_score for game in games], LEAGUE_AVG_POINTS),
        recent_off_eff=mean([game.team_score for game in recent10], LEAGUE_AVG_POINTS),
        recent_def_eff=mean([game.opponent_score for game in recent10], LEAGUE_AVG_POINTS),
        home_win_pct=win_pct(home_games),
        away_win_pct=win_pct(away_games),
        home_avg_margin=mean([game.margin for game in home_games], 0.0),
        away_avg_margin=mean([game.margin for game in away_games], 0.0),
        rest_days=rest_days,
        back_to_back=rest_days <= 1,
        injury_penalty=injury_penalty(team, injuries),
        injuries=injuries,
        warnings=warnings,
    )


def win_pct(games: list[TeamGame]) -> float:
    if not games:
        return 0.5
    return sum(1 for game in games if game.win) / len(games)


def injury_penalty(team: str, injuries: list[dict[str, str]]) -> float:
    penalty = KNOWN_INJURY_PENALTIES.get(team, 0.0)
    for injury in injuries:
        status = str(injury.get("status", "")).lower()
        weighted_impact = injury.get("weighted_impact")
        if weighted_impact is not None:
            try:
                penalty -= float(weighted_impact)
                continue
            except (TypeError, ValueError):
                pass
        impact_score = 2.0
        try:
            impact_score = float(injury.get("impact_score", impact_score))
        except (TypeError, ValueError):
            pass
        if "out" in status:
            penalty -= impact_score
        elif "doubt" in status:
            penalty -= impact_score * 0.8
        elif "questionable" in status:
            penalty -= impact_score * 0.45
        elif "probable" in status:
            penalty -= impact_score * 0.15
    return penalty


def score_team(team: TeamMetrics, opponent: TeamMetrics, is_home: bool) -> RatingBreakdown:
    team_strength_score = 36.0 * (team.season_win_pct - 0.5) + 0.12 * (team.avg_points_for - team.avg_points_against)
    recent_form_score = 24.0 * (team.recent10_win_pct - 0.5) + 18.0 * (team.recent5_win_pct - 0.5)
    offense_score = 0.36 * (team.recent_off_eff - LEAGUE_AVG_POINTS) + 0.16 * (team.recent_off_eff - opponent.recent_def_eff)
    defense_score = 0.36 * (LEAGUE_AVG_POINTS - team.recent_def_eff) + 0.16 * (opponent.recent_off_eff - team.recent_def_eff)
    if is_home:
        home_advantage_score = 2.4 + 10.0 * (team.home_win_pct - 0.5) + 0.08 * team.home_avg_margin
    else:
        home_advantage_score = 10.0 * (team.away_win_pct - 0.5) + 0.08 * team.away_avg_margin
    rest = float(team.rest_days if team.rest_days is not None else 2)
    rest_advantage_score = clamp(rest, 0.0, 4.0) * 0.9 - (2.5 if team.back_to_back else 0.0)
    return RatingBreakdown(
        team_strength_score=team_strength_score,
        recent_form_score=recent_form_score,
        offense_score=offense_score,
        defense_score=defense_score,
        home_advantage_score=home_advantage_score,
        rest_advantage_score=rest_advantage_score,
        injury_penalty=team.injury_penalty,
    )


def predict_game(game: ScheduledGame, home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> GamePrediction:
    home_breakdown = score_team(home_metrics, away_metrics, is_home=True)
    away_breakdown = score_team(away_metrics, home_metrics, is_home=False)
    rating_diff = home_breakdown.total - away_breakdown.total
    home_prob = clamp(1.0 / (1.0 + math.exp(-rating_diff / 13.5)), 0.05, 0.95)
    cap_info = confidence_cap(home_prob, rating_diff, home_metrics, away_metrics)
    home_prob = apply_probability_cap(home_prob, cap_info["confidence_cap"])
    away_prob = 1.0 - home_prob

    base_home = 0.56 * home_metrics.recent_off_eff + 0.44 * away_metrics.recent_def_eff
    base_away = 0.56 * away_metrics.recent_off_eff + 0.44 * home_metrics.recent_def_eff
    home_expected = base_home + 1.8 + 0.08 * rating_diff + rest_margin(home_metrics, away_metrics)
    away_expected = base_away - 0.08 * rating_diff - rest_margin(home_metrics, away_metrics)
    home_score = int(round(clamp(home_expected, 82.0, 145.0)))
    away_score = int(round(clamp(away_expected, 82.0, 145.0)))
    predicted_total = home_score + away_score
    winner = game.home_team if home_prob >= 0.5 else game.away_team
    confidence = confidence_level(max(home_prob, away_prob), home_metrics, away_metrics, cap_info["confidence_cap"])
    total_lean = total_points_lean(predicted_total)
    reasons, risks = explain_prediction(game, home_breakdown, away_breakdown, home_metrics, away_metrics, home_prob)
    risks.extend(cap_info["favorite_risk_reason"])
    risks.append(f"confidence_cap={cap_info['confidence_cap']:.2f}, upset_risk_score={cap_info['upset_risk_score']:.2f}.")
    return GamePrediction(
        game=game,
        home_metrics=home_metrics,
        away_metrics=away_metrics,
        home_breakdown=home_breakdown,
        away_breakdown=away_breakdown,
        home_win_probability=home_prob,
        away_win_probability=away_prob,
        predicted_home_score=home_score,
        predicted_away_score=away_score,
        predicted_total=predicted_total,
        total_lean=total_lean,
        confidence=confidence,
        predicted_winner=winner,
        reasons=reasons,
        risks=risks,
    )


def rest_margin(home: TeamMetrics, away: TeamMetrics) -> float:
    home_rest = home.rest_days if home.rest_days is not None else 2
    away_rest = away.rest_days if away.rest_days is not None else 2
    return clamp(float(home_rest - away_rest), -3.0, 3.0) * 0.55


def confidence_level(win_prob: float, home: TeamMetrics, away: TeamMetrics, confidence_cap_value: float = 1.0) -> str:
    if home.games_used < 10 or away.games_used < 10:
        return "Low"
    if confidence_cap_value <= 0.70:
        return "Medium" if win_prob >= 0.66 else "Low"
    if win_prob >= 0.86:
        return "High"
    if win_prob >= 0.63:
        return "Medium"
    return "Low"


def confidence_cap(home_prob: float, rating_diff: float, home: TeamMetrics, away: TeamMetrics) -> dict[str, object]:
    favorite = home if home_prob >= 0.5 else away
    underdog = away if home_prob >= 0.5 else home
    favorite_is_away = home_prob < 0.5
    cap = 0.80
    risk = 0.0
    reasons: list[str] = []
    favorite_probability = max(home_prob, 1.0 - home_prob)
    if favorite_probability > 0.75 and abs(favorite.injury_penalty) >= 3.0:
        cap = min(cap, 0.68)
        risk += 0.24
        reasons.append(f"favorite_risk_reason={favorite.team} has injury risk despite high favorite probability.")
    if favorite.back_to_back:
        cap = min(cap, 0.66)
        risk += 0.22
        reasons.append(f"favorite_risk_reason={favorite.team} is on a back-to-back.")
    if favorite_is_away:
        cap = min(cap, 0.72)
        risk += 0.14
        reasons.append(f"favorite_risk_reason={favorite.team} is an away favorite.")
    if momentum_unstable(favorite):
        cap = min(cap, 0.70)
        risk += 0.16
        reasons.append(f"favorite_risk_reason={favorite.team} has unstable recent momentum.")
    if abs(rating_diff) < 9.0:
        cap = min(cap, 0.70)
        risk += 0.18
        reasons.append("favorite_risk_reason=rating edge is not large enough for high confidence.")
    if abs(underdog.injury_penalty) < 2.0 and underdog.recent5_win_pct >= 0.6:
        cap = min(cap, 0.72)
        risk += 0.12
        reasons.append(f"favorite_risk_reason={underdog.team} has upset profile: healthy and recent form is strong.")
    return {
        "confidence_cap": cap,
        "upset_risk_score": min(1.0, risk),
        "favorite_risk_reason": reasons,
    }


def apply_probability_cap(home_prob: float, cap: float) -> float:
    if home_prob >= 0.5:
        return min(home_prob, cap)
    return max(home_prob, 1.0 - cap)


def momentum_unstable(team: TeamMetrics) -> bool:
    if 0.4 <= team.recent5_win_pct <= 0.6:
        return True
    return abs(team.recent5_win_pct - team.recent10_win_pct) >= 0.35


def total_points_lean(predicted_total: int) -> str:
    if predicted_total >= DEFAULT_TOTAL_LINE + 5:
        return "Over lean"
    if predicted_total <= DEFAULT_TOTAL_LINE - 5:
        return "Under lean"
    return "Neutral"


def explain_prediction(
    game: ScheduledGame,
    home: RatingBreakdown,
    away: RatingBreakdown,
    home_metrics: TeamMetrics,
    away_metrics: TeamMetrics,
    home_prob: float,
) -> tuple[list[str], list[str]]:
    component_names = [
        ("team strength", home.team_strength_score - away.team_strength_score),
        ("recent form", home.recent_form_score - away.recent_form_score),
        ("offense", home.offense_score - away.offense_score),
        ("defense", home.defense_score - away.defense_score),
        ("home/away split", home.home_advantage_score - away.home_advantage_score),
        ("rest", home.rest_advantage_score - away.rest_advantage_score),
        ("injury availability", home.injury_penalty - away.injury_penalty),
    ]
    favorite = game.home_team if home_prob >= 0.5 else game.away_team
    direction = 1 if home_prob >= 0.5 else -1
    sorted_factors = sorted(component_names, key=lambda item: abs(item[1]), reverse=True)
    reasons = [f"Model favors {favorite} mainly because of {name} ({value * direction:+.1f} rating edge)." for name, value in sorted_factors[:3]]
    risks: list[str] = []
    risks.extend(home_metrics.warnings)
    risks.extend(away_metrics.warnings)
    if home_metrics.back_to_back:
        risks.append(f"{game.home_team} is on a back-to-back or one-day rest.")
    if away_metrics.back_to_back:
        risks.append(f"{game.away_team} is on a back-to-back or one-day rest.")
    if home_metrics.injuries:
        risks.append(f"{game.home_team} injury penalty {home_metrics.injury_penalty:.1f}.")
    if away_metrics.injuries:
        risks.append(f"{game.away_team} injury penalty {away_metrics.injury_penalty:.1f}.")
    if not home_metrics.injuries and not away_metrics.injuries:
        risks.append("Injury feed is not configured; injury penalties use only the local placeholder list.")
    if abs(home_prob - 0.5) < 0.08:
        risks.append("Win probability is close to 50/50, so confidence is naturally limited.")
    return reasons, risks[:5]
