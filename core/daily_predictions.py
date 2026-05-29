from __future__ import annotations

import datetime as dt
import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from ..config import (
        DAILY_PREDICTIONS_CSV,
        DAILY_PREDICTIONS_TXT,
        DAILY_SHORT_SCRIPT_TXT,
        DAILY_SOCIAL_POSTS_TXT,
        MODEL_VERSION_JSON,
        OUTPUT_PREDICTIONS_CSV,
        WEIGHT_TUNING_JSON,
        ensure_project_dirs,
        project_relative,
    )
    from .automation_status import update_automation_status
    from .prediction_result import PredictionResult, current_model_version
    from .utils import append_csv_row
    from ..sports.football.football_model import FootballPredictor
    from ..sports.nba.nba_model import NBAPredictor
except ImportError:
    from config import (
        DAILY_PREDICTIONS_CSV,
        DAILY_PREDICTIONS_TXT,
        DAILY_SHORT_SCRIPT_TXT,
        DAILY_SOCIAL_POSTS_TXT,
        MODEL_VERSION_JSON,
        OUTPUT_PREDICTIONS_CSV,
        WEIGHT_TUNING_JSON,
        ensure_project_dirs,
        project_relative,
    )
    from core.automation_status import update_automation_status
    from core.prediction_result import PredictionResult, current_model_version
    from core.utils import append_csv_row
    from sports.football.football_model import FootballPredictor
    from sports.nba.nba_model import NBAPredictor


CONFIDENCE_SCORES = {"Low": 0.42, "Medium": 0.62, "High": 0.78}


@dataclass(frozen=True)
class DailyPredictionPackage:
    generated_at: dt.datetime
    predictions: list[PredictionResult]
    highest_confidence: list[PredictionResult] = field(default_factory=list)
    upset_watch: list[PredictionResult] = field(default_factory=list)
    draw_watch: list[PredictionResult] = field(default_factory=list)
    injury_risk_games: list[PredictionResult] = field(default_factory=list)
    best_value: list[PredictionResult] = field(default_factory=list)
    model_version: str = "v1.0.0"
    csv_path: Path = DAILY_PREDICTIONS_CSV
    txt_path: Path = DAILY_PREDICTIONS_TXT
    short_script_path: Path = DAILY_SHORT_SCRIPT_TXT
    social_posts_path: Path = DAILY_SOCIAL_POSTS_TXT


def build_daily_prediction_package(predictions: Iterable[PredictionResult]) -> DailyPredictionPackage:
    ensure_project_dirs()
    results = list(predictions)
    version = ensure_model_version()
    package = DailyPredictionPackage(
        generated_at=dt.datetime.now(),
        predictions=results,
        highest_confidence=highest_confidence_picks(results),
        upset_watch=upset_watch(results),
        draw_watch=draw_watch(results),
        injury_risk_games=injury_risk_games(results),
        best_value=best_value_picks(results),
        model_version=version,
    )
    write_daily_outputs(package)
    write_content_outputs(package)
    save_tracking_rows(package)
    update_automation_status(
        last_daily_run=package.generated_at.isoformat(timespec="seconds"),
        last_daily_status="success",
        daily_prediction_count=len(package.predictions),
        daily_predictions_csv=project_relative(package.csv_path),
        daily_predictions_txt=project_relative(package.txt_path),
    )
    return package


def generate_daily_predictions() -> DailyPredictionPackage:
    target_date = dt.date.today()
    predictions: list[PredictionResult] = []
    nba_args = Namespace(
        sport="nba",
        date=target_date.isoformat(),
        home="",
        away="",
        mode="",
        backtest=False,
        evaluate=False,
        injuries=False,
        season="2025-26",
        limit=100,
        verbose=False,
    )
    football_args = Namespace(
        sport="football",
        date=target_date.isoformat(),
        home="",
        away="",
        mode="WORLD_CUP",
        backtest=False,
        evaluate=False,
        injuries=False,
        season="2025-26",
        limit=100,
        verbose=False,
    )
    try:
        predictions.extend(NBAPredictor().predict(nba_args))
    except Exception as exc:
        update_automation_status(last_daily_status=f"nba_failed: {exc}")
    try:
        predictions.extend(FootballPredictor().predict_live(football_args))
    except Exception as exc:
        update_automation_status(last_daily_status=f"football_failed: {exc}")
    return build_daily_prediction_package(predictions)


def ensure_model_version() -> str:
    ensure_project_dirs()
    if MODEL_VERSION_JSON.exists():
        try:
            payload = json.loads(MODEL_VERSION_JSON.read_text(encoding="utf-8"))
            if payload.get("version"):
                return str(payload["version"])
        except Exception:
            pass
    payload = {
        "version": "v1.0.0",
        "weights": load_current_weights(),
        "last_updated": dt.datetime.now().isoformat(timespec="seconds"),
        "calibration_status": {
            "nba": "confidence calibrated",
            "football": "draw probability calibrated",
        },
    }
    MODEL_VERSION_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(payload["version"])


def load_current_weights() -> dict[str, object]:
    if not WEIGHT_TUNING_JSON.exists():
        return {}
    try:
        return json.loads(WEIGHT_TUNING_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_daily_outputs(package: DailyPredictionPackage) -> None:
    rows = [daily_row(result, package) for result in package.predictions]
    pd.DataFrame(rows).to_csv(package.csv_path, index=False)
    package.txt_path.write_text(daily_report_text(package), encoding="utf-8")


def write_content_outputs(package: DailyPredictionPackage) -> None:
    package.short_script_path.parent.mkdir(parents=True, exist_ok=True)
    package.short_script_path.write_text(shorts_script(package), encoding="utf-8")
    package.social_posts_path.write_text(social_posts(package), encoding="utf-8")


def save_tracking_rows(package: DailyPredictionPackage) -> None:
    rows = [daily_row(result, package) for result in package.predictions]
    if not rows:
        return
    if OUTPUT_PREDICTIONS_CSV.exists():
        try:
            existing = pd.read_csv(OUTPUT_PREDICTIONS_CSV).fillna("")
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    incoming = pd.DataFrame(rows).fillna("")
    if not existing.empty:
        for column in incoming.columns:
            if column not in existing.columns:
                existing[column] = ""
        for _, row in incoming.iterrows():
            duplicate = (
                (existing.get("date", "").astype(str) == str(row.get("date", "")))
                & (existing.get("sport", "").astype(str) == str(row.get("sport", "")))
                & (existing.get("home_team", "").astype(str) == str(row.get("home_team", "")))
                & (existing.get("away_team", "").astype(str) == str(row.get("away_team", "")))
            )
            existing = existing[~duplicate]
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    OUTPUT_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)


def daily_row(result: PredictionResult, package: DailyPredictionPackage) -> dict[str, object]:
    base = result.to_row()
    base.update(
        {
            "created_at": package.generated_at.isoformat(timespec="seconds"),
            "model_version": package.model_version,
            "top_probability": round(top_probability(result), 4),
            "pick_category": pick_category(result, package),
        }
    )
    return base


def pick_category(result: PredictionResult, package: DailyPredictionPackage) -> str:
    categories: list[str] = []
    if result in package.highest_confidence:
        categories.append("highest_confidence")
    if result in package.best_value:
        categories.append("best_value")
    if result in package.upset_watch:
        categories.append("upset_watch")
    if result in package.draw_watch:
        categories.append("draw_watch")
    if result in package.injury_risk_games:
        categories.append("injury_risk")
    return "|".join(categories) or "standard"


def daily_report_text(package: DailyPredictionPackage) -> str:
    lines = [
        "AI Sports Predictor Daily Report",
        f"Generated: {package.generated_at.isoformat(timespec='seconds')}",
        f"Model version: {package.model_version}",
        "",
        "Today's NBA Predictions",
        *prediction_lines([item for item in package.predictions if item.sport == "nba"]),
        "",
        "Today's Football Predictions",
        *prediction_lines([item for item in package.predictions if item.sport == "football"]),
        "",
        "Highest Confidence Picks",
        *prediction_lines(package.highest_confidence),
        "",
        "Upset Watch",
        *prediction_lines(package.upset_watch),
        "",
        "Draw Watch",
        *prediction_lines(package.draw_watch),
        "",
        "Injury Risk Games",
        *prediction_lines(package.injury_risk_games),
    ]
    return "\n".join(lines).strip() + "\n"


def prediction_lines(results: Iterable[PredictionResult]) -> list[str]:
    rows = []
    for result in results:
        draw = f", draw {result.draw_probability:.1%}" if result.draw_probability is not None else ""
        rows.append(
            f"- {result.match}: {result.predicted_winner}, {result.predicted_score}, "
            f"home {nullable_pct(result.win_probability_home)}, away {nullable_pct(result.win_probability_away)}{draw}, "
            f"confidence {result.confidence}"
        )
    return rows or ["- No games available."]


def shorts_script(package: DailyPredictionPackage) -> str:
    lead = first_or_none(package.highest_confidence) or first_or_none(package.predictions)
    upset = first_or_none(package.upset_watch)
    draw = first_or_none(package.draw_watch)
    if not lead:
        return "No games are available today. Check back after the next schedule update.\n"
    lines = [
        "TikTok / YouTube Shorts Script",
        "",
        "Hook: Today's AI sports board is live, and one matchup stands out.",
        f"Pick: {lead.predicted_winner} in {lead.match}.",
        f"Projection: {lead.predicted_score} with {lead.confidence.lower()} confidence.",
        f"Why: {lead.key_factors[0] if lead.key_factors else 'The model edge comes from form, Elo, and matchup data.'}",
    ]
    if upset:
        lines.append(f"Upset watch: {upset.predicted_winner} has a live path in {upset.match}.")
    if draw:
        lines.append(f"Draw watch: {draw.match} carries a {nullable_pct(draw.draw_probability)} draw signal.")
    lines.append("Close: Follow for the daily prediction card and postgame tracking.")
    return "\n".join(lines) + "\n"


def social_posts(package: DailyPredictionPackage) -> str:
    lead = first_or_none(package.highest_confidence) or first_or_none(package.predictions)
    draw = first_or_none(package.draw_watch)
    if not lead:
        return "No daily posts generated because no games are available today.\n"
    x_post = (
        f"Daily AI pick: {lead.predicted_winner} in {lead.match}. "
        f"Projection: {lead.predicted_score}. Confidence: {lead.confidence}. "
        f"Model version: {package.model_version}. #SportsAI #Predictions"
    )
    title = f"{lead.predicted_winner} is today's top AI sports pick"
    instagram = (
        f"Today's model card is live. Top pick: {lead.predicted_winner} in {lead.match}. "
        f"Projected score: {lead.predicted_score}. "
        f"{'Draw alert: ' + draw.match + ' at ' + nullable_pct(draw.draw_probability) if draw else 'No major draw alert today.'}"
    )
    return "\n".join(
        [
            "Twitter/X Post",
            x_post,
            "",
            "YouTube Shorts Title",
            title,
            "",
            "Instagram Caption",
            instagram,
        ]
    ) + "\n"


def highest_confidence_picks(results: list[PredictionResult]) -> list[PredictionResult]:
    return sorted(results, key=lambda item: (confidence_score(item), top_probability(item)), reverse=True)[:3]


def best_value_picks(results: list[PredictionResult]) -> list[PredictionResult]:
    candidates = [
        item for item in results
        if top_probability(item) >= 0.52 and (item.win_probability_away or 0.0) >= (item.win_probability_home or 0.0)
    ]
    return sorted(candidates or results, key=lambda item: abs(top_probability(item) - 0.58))[:2]


def upset_watch(results: list[PredictionResult]) -> list[PredictionResult]:
    candidates = []
    for result in results:
        text = " ".join(result.key_factors + result.risk_factors).lower()
        away_pick = result.predicted_winner == result.away_team
        upset_signal = "upset" in text or "favorite_risk_reason" in text or away_pick
        if upset_signal and top_probability(result) <= 0.68:
            candidates.append(result)
    return sorted(candidates, key=top_probability, reverse=True)[:3]


def draw_watch(results: list[PredictionResult]) -> list[PredictionResult]:
    candidates = [item for item in results if item.draw_probability is not None and item.draw_probability >= 0.25]
    return sorted(candidates, key=lambda item: item.draw_probability or 0.0, reverse=True)[:3]


def injury_risk_games(results: list[PredictionResult]) -> list[PredictionResult]:
    candidates = []
    for result in results:
        text = f" {' '.join(result.key_factors + result.risk_factors).lower()} "
        if (
            "injury" in text
            or "missing_starters" in text
            or "questionable" in text
            or " doubtful " in text
            or " probable " in text
            or " out " in text
        ):
            candidates.append(result)
    return candidates[:4]


def top_probability(result: PredictionResult) -> float:
    values = [value for value in (result.win_probability_home, result.win_probability_away, result.draw_probability) if value is not None]
    return max(values) if values else confidence_score(result)


def confidence_score(result: PredictionResult) -> float:
    return CONFIDENCE_SCORES.get(result.confidence, 0.5)


def nullable_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def first_or_none(items: Iterable[PredictionResult]) -> PredictionResult | None:
    return next(iter(items), None)


if __name__ == "__main__":
    ensure_project_dirs()
    ensure_model_version()
    empty = build_daily_prediction_package([])
    print(f"Wrote {empty.csv_path}")
    print(f"Wrote {empty.txt_path}")
