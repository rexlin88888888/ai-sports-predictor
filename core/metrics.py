from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..config import BACKTEST_REPORT_TXT, FOOTBALL_DATA_DIR, MODELS_DIR, NBA_DATA_DIR, REPORTS_DIR, project_relative
    from .tuning import tune_model_weights
    from .utils import mean, safe_float, write_csv
except ImportError:
    from config import BACKTEST_REPORT_TXT, FOOTBALL_DATA_DIR, MODELS_DIR, NBA_DATA_DIR, REPORTS_DIR, project_relative
    from core.tuning import tune_model_weights
    from core.utils import mean, safe_float, write_csv


def evaluate_sport(sport: str) -> dict[str, Any]:
    if sport == "nba":
        result = evaluate_nba(NBA_DATA_DIR / "nba_backtest_results.csv")
    elif sport == "football":
        result = evaluate_football(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    else:
        raise ValueError(f"Unsupported sport: {sport}")
    save_model_version(sport, result)
    return result


def evaluate_nba(path: Path) -> dict[str, Any]:
    frame = read_backtest(path)
    if frame.empty:
        return {"sport": "nba", "games": 0, "weakness": "missing backtest data"}
    score_errors = numeric(frame, "score_error")
    total_errors = numeric(frame, "total_points_error")
    residuals = []
    if {"predicted_home_score", "predicted_away_score", "actual_home_score", "actual_away_score"} <= set(frame.columns):
        for _, row in frame.iterrows():
            residuals.extend([
                safe_float(row["predicted_home_score"]) - safe_float(row["actual_home_score"]),
                safe_float(row["predicted_away_score"]) - safe_float(row["actual_away_score"]),
            ])
    metrics = {
        "sport": "nba",
        "games": len(frame),
        "accuracy": accuracy(frame),
        "average_score_error": mean(score_errors),
        "average_total_points_error": mean(total_errors),
        "MAE": mean([abs(value) for value in residuals]) if residuals else mean(score_errors),
        "RMSE": math.sqrt(mean([value * value for value in residuals])) if residuals else 0.0,
        "average_confidence": average_confidence(frame),
        "favorite_accuracy": boolean_group_accuracy(frame, "predicted_is_favorite"),
        "underdog_accuracy": boolean_group_accuracy(frame, "predicted_is_underdog"),
    }
    calibration = calibration_report(frame)
    confidence = confidence_report(frame)
    confidence_distribution = confidence_distribution_report(frame)
    upset = upset_report(frame)
    roi = roi_simulation(frame)
    tuning = tune_model_weights(frame, "nba")
    feature_importance = feature_importance_report(frame, ["recent_form_edge", "home_advantage_edge", "offense_edge", "defense_edge", "fatigue_edge", "injury_edge", "elo_difference"])
    drift = drift_report(frame)
    weakness = weakness_nba(metrics, confidence, upset)
    write_csv(NBA_DATA_DIR / "calibration_report.csv", calibration)
    report_path = write_text_report("nba", metrics, confidence, upset, feature_importance, drift, weakness)
    historical_report_path = write_historical_backtest_report("nba", metrics, confidence_distribution, upset, roi, tuning, weakness)
    return {
        **metrics,
        "calibration_report": project_relative(NBA_DATA_DIR / "calibration_report.csv"),
        "confidence_validation": confidence,
        "confidence_distribution": confidence_distribution,
        "upset_analysis": upset,
        "roi_simulation": roi,
        "weight_tuning": tuning,
        "feature_importance": feature_importance,
        "prediction_drift": drift,
        "report": project_relative(report_path),
        "historical_report": project_relative(historical_report_path),
        "weakness": weakness,
    }


def evaluate_football(path: Path) -> dict[str, Any]:
    frame = read_backtest(path)
    if frame.empty:
        return {"sport": "football", "games": 0, "weakness": "missing backtest data"}
    goal_errors = football_goal_errors(frame)
    log_losses = football_log_losses(frame)
    brier_scores = football_brier_scores(frame)
    metrics = {
        "sport": "football",
        "games": len(frame),
        "accuracy": accuracy(frame),
        "draw_accuracy": draw_accuracy(frame),
        "over_under_accuracy": over_under_accuracy(frame),
        "average_goal_error": mean(goal_errors),
        "log_loss": mean(log_losses),
        "brier_score": mean(brier_scores),
        "average_confidence": average_confidence(frame),
    }
    calibration = calibration_report(frame)
    confidence = confidence_report(frame)
    confidence_distribution = confidence_distribution_report(frame)
    upset = upset_report(frame)
    roi = roi_simulation(frame)
    tuning = tune_model_weights(frame, "football")
    feature_importance = feature_importance_report(frame, ["recent_form_edge", "home_advantage_edge", "elo_difference"])
    if not feature_importance:
        feature_importance = [{"feature": "probability_model", "importance": round(abs(metrics["accuracy"] - 0.5), 4)}]
    drift = drift_report(frame)
    weakness = weakness_football(metrics, confidence, upset)
    write_csv(FOOTBALL_DATA_DIR / "calibration_report.csv", calibration)
    report_path = write_text_report("football", metrics, confidence, upset, feature_importance, drift, weakness)
    historical_report_path = write_historical_backtest_report("football", metrics, confidence_distribution, upset, roi, tuning, weakness)
    return {
        **metrics,
        "calibration_report": project_relative(FOOTBALL_DATA_DIR / "calibration_report.csv"),
        "confidence_validation": confidence,
        "confidence_distribution": confidence_distribution,
        "upset_analysis": upset,
        "roi_simulation": roi,
        "weight_tuning": tuning,
        "feature_importance": feature_importance,
        "prediction_drift": drift,
        "report": project_relative(report_path),
        "historical_report": project_relative(historical_report_path),
        "weakness": weakness,
    }


def read_backtest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def accuracy(frame: pd.DataFrame) -> float:
    if "correct" not in frame or frame.empty:
        return 0.0
    return float(frame["correct"].astype(str).str.lower().isin(["true", "1"]).mean())


def numeric(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame:
        return []
    return [safe_float(value) for value in frame[column].tolist()]


def calibration_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "predicted_probability" not in frame or "actual_result" not in frame:
        return []
    rows: list[dict[str, Any]] = []
    for low in [i / 10 for i in range(0, 10)]:
        high = low + 0.1
        bucket = frame[(frame["predicted_probability"] >= low) & (frame["predicted_probability"] < high)]
        if bucket.empty:
            continue
        rows.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "games": len(bucket),
                "avg_predicted_probability": round(float(bucket["predicted_probability"].mean()), 4),
                "actual_win_rate": round(float(bucket["actual_result"].mean()), 4),
                "calibration_error": round(abs(float(bucket["predicted_probability"].mean()) - float(bucket["actual_result"].mean())), 4),
            }
        )
    return rows


def confidence_report(frame: pd.DataFrame) -> dict[str, float | int]:
    if "confidence" not in frame:
        return {}
    rows: dict[str, float | int] = {}
    for level in ("High", "Medium", "Low"):
        bucket = frame[frame["confidence"] == level]
        rows[f"{level.lower()}_games"] = len(bucket)
        rows[f"{level.lower()}_accuracy"] = accuracy(bucket) if not bucket.empty else 0.0
    return rows


def confidence_distribution_report(frame: pd.DataFrame) -> dict[str, int]:
    if "confidence" not in frame:
        return {}
    return {level: int((frame["confidence"] == level).sum()) for level in ("High", "Medium", "Low")}


def average_confidence(frame: pd.DataFrame) -> float:
    if "confidence_value" in frame:
        return float(frame["confidence_value"].astype(float).mean())
    if "predicted_probability" in frame:
        return float(frame["predicted_probability"].astype(float).mean())
    return 0.0


def boolean_group_accuracy(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or "correct" not in frame:
        return 0.0
    bucket = frame[frame[column].astype(str).str.lower().isin(["true", "1"])]
    return accuracy(bucket) if not bucket.empty else 0.0


def upset_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    predicted_prob = numeric(frame, "predicted_probability")
    actual = numeric(frame, "actual_result")
    upsets = [1 for p, a in zip(predicted_prob, actual) if p >= 0.65 and a == 0]
    max_error = None
    incorrect = frame[~frame["correct"].astype(str).str.lower().isin(["true", "1"])] if "correct" in frame else frame
    error_frame = incorrect if not incorrect.empty else frame
    if "score_error" in error_frame:
        idx = error_frame["score_error"].astype(float).idxmax()
        max_error = error_frame.loc[idx].to_dict()
    elif "predicted_probability" in error_frame and "actual_result" in error_frame:
        idx = (error_frame["predicted_probability"].astype(float) - error_frame["actual_result"].astype(float)).abs().idxmax()
        max_error = error_frame.loc[idx].to_dict()
    return {
        "upset_rate": len(upsets) / len(frame),
        "high_probability_failures": len(upsets),
        "largest_prediction_error": compact_error(max_error),
    }


def roi_simulation(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "predicted_probability" not in frame or "actual_result" not in frame:
        return {"bets": 0, "roi": 0.0}
    profit = 0.0
    bets = 0
    for _, row in frame.iterrows():
        probability = safe_float(row.get("predicted_probability"), 0.0)
        if probability < 0.55:
            continue
        decimal_odds = max(1.2, min(5.0, 1.0 / max(probability, 0.01)))
        bets += 1
        profit += (decimal_odds - 1.0) if safe_float(row.get("actual_result"), 0.0) >= 1.0 else -1.0
    return {
        "bets": bets,
        "profit_units": round(profit, 3),
        "roi": round(profit / bets, 4) if bets else 0.0,
    }


def compact_error(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    predicted = row.get("predicted_winner", "")
    actual = row.get("actual_winner", "")
    return f"{away} at {home}: predicted {predicted}, actual {actual}"


def feature_importance_report(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "correct" not in frame:
        return rows
    y = frame["correct"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    for column in feature_columns:
        if column not in frame:
            continue
        values = frame[column].astype(float)
        if values.nunique() <= 1:
            corr = 0.0
        else:
            corr = float(values.corr(y))
            if math.isnan(corr):
                corr = 0.0
        rows.append({"feature": readable_feature(column), "importance": round(abs(corr), 4), "direction": round(corr, 4)})
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def readable_feature(column: str) -> str:
    return {
        "recent_form_edge": "recent_form",
        "home_advantage_edge": "home_advantage",
        "offense_edge": "offensive_rating",
        "defense_edge": "defensive_rating",
        "fatigue_edge": "fatigue",
        "elo_difference": "elo_difference",
    }.get(column, column)


def drift_report(frame: pd.DataFrame) -> dict[str, Any]:
    if "date" not in frame or len(frame) < 10:
        return {"status": "not_enough_data"}
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"])
    if dated.empty:
        return {"status": "not_enough_data"}
    last_date = dated["date"].max()
    recent = dated[dated["date"] >= last_date - pd.Timedelta(days=30)]
    older = dated[dated["date"] < last_date - pd.Timedelta(days=30)]
    return {
        "recent_30d_games": len(recent),
        "recent_30d_accuracy": accuracy(recent) if not recent.empty else 0.0,
        "previous_accuracy": accuracy(older) if not older.empty else 0.0,
        "trend": "worse" if not older.empty and accuracy(recent) < accuracy(older) else "stable_or_better",
    }


def football_goal_errors(frame: pd.DataFrame) -> list[float]:
    errors: list[float] = []
    for _, row in frame.iterrows():
        predicted = parse_score_pair(str(row.get("predicted_score", "")))
        actual = parse_score_pair(str(row.get("actual_score", "")))
        if predicted and actual:
            errors.append((abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])) / 2.0)
    return errors


def parse_score_pair(value: str) -> tuple[int, int] | None:
    import re

    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    return numbers[-2], numbers[-1]


def football_log_losses(frame: pd.DataFrame) -> list[float]:
    losses: list[float] = []
    for _, row in frame.iterrows():
        label = row.get("actual_label")
        if label == "HOME_WIN":
            p = safe_float(row.get("home_win_probability"), 1e-9)
        elif label == "AWAY_WIN":
            p = safe_float(row.get("away_win_probability"), 1e-9)
        else:
            p = safe_float(row.get("draw_probability"), 1e-9)
        losses.append(-math.log(max(min(p, 1 - 1e-9), 1e-9)))
    return losses


def football_brier_scores(frame: pd.DataFrame) -> list[float]:
    scores: list[float] = []
    for _, row in frame.iterrows():
        label = row.get("actual_label")
        probs = {
            "HOME_WIN": safe_float(row.get("home_win_probability")),
            "DRAW": safe_float(row.get("draw_probability")),
            "AWAY_WIN": safe_float(row.get("away_win_probability")),
        }
        scores.append(sum((probs[key] - (1.0 if key == label else 0.0)) ** 2 for key in probs) / 3.0)
    return scores


def draw_accuracy(frame: pd.DataFrame) -> float:
    draws = frame[frame.get("actual_label", "") == "DRAW"] if "actual_label" in frame else pd.DataFrame()
    return accuracy(draws) if not draws.empty else 0.0


def over_under_accuracy(frame: pd.DataFrame) -> float:
    if "over_under_correct" not in frame or frame.empty:
        return 0.0
    return float(frame["over_under_correct"].astype(str).str.lower().isin(["true", "1"]).mean())


def weakness_nba(metrics: dict[str, Any], confidence: dict[str, Any], upset: dict[str, Any]) -> str:
    if confidence.get("high_games", 0) and confidence.get("high_accuracy", 0.0) < metrics.get("accuracy", 0.0):
        return "High confidence predictions are not more reliable than the average."
    if upset.get("upset_rate", 0.0) > 0.2:
        return "High-probability favorites are failing too often."
    if metrics.get("average_total_points_error", 0.0) > 14:
        return "Total points prediction is noisy."
    return "Largest current weakness is score precision, not winner selection."


def weakness_football(metrics: dict[str, Any], confidence: dict[str, Any], upset: dict[str, Any]) -> str:
    if metrics.get("draw_accuracy", 0.0) < 0.25:
        return "draw prediction weak"
    if metrics.get("log_loss", 0.0) > 1.2:
        return "probability calibration is weak"
    return "Football model is most limited by simple team-strength features."


def write_text_report(
    sport: str,
    metrics: dict[str, Any],
    confidence: dict[str, Any],
    upset: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    drift: dict[str, Any],
    weakness: str,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{sport}_report.txt"
    lines = [
        f"{sport.upper()} MODEL REPORT",
        "=" * 60,
        "Metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Confidence validation:"])
    for key, value in confidence.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Upset analysis:"])
    for key, value in upset.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Feature importance:"])
    for item in feature_importance[:8]:
        lines.append(f"- {item}")
    lines.extend(["", "Prediction drift:"])
    for key, value in drift.items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "Most stable teams: requires longer team-level history in future versions.",
        "Hardest teams to predict: inspect largest_prediction_error and team-level residuals.",
        "",
        f"Biggest weakness: {weakness}",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_historical_backtest_report(
    sport: str,
    metrics: dict[str, Any],
    confidence_distribution: dict[str, int],
    upset: dict[str, Any],
    roi: dict[str, Any],
    tuning: dict[str, Any],
    weakness: str,
) -> Path:
    BACKTEST_REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    existing = BACKTEST_REPORT_TXT.read_text(encoding="utf-8") if BACKTEST_REPORT_TXT.exists() else ""
    lines = [
        f"{sport.upper()} HISTORICAL BACKTEST",
        "=" * 60,
        f"total_predictions: {metrics.get('games', 0)}",
        f"accuracy: {metrics.get('accuracy', 0.0):.4f}",
        f"average_confidence: {metrics.get('average_confidence', 0.0):.4f}",
        f"ROI simulation: {roi}",
        f"confidence_distribution: {confidence_distribution}",
        f"upset_analysis: {upset}",
        f"auto_tuned_weights: {tuning.get('weights', {})}",
        f"current_biggest_weakness: {weakness}",
        "",
    ]
    blocks = [
        block.strip()
        for block in existing.split("\n\n")
        if block.strip() and not block.strip().startswith(f"{sport.upper()} HISTORICAL BACKTEST")
    ]
    blocks.append("\n".join(lines).strip())
    BACKTEST_REPORT_TXT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return BACKTEST_REPORT_TXT


def save_model_version(sport: str, metrics: dict[str, Any]) -> Path:
    model_dir = MODELS_DIR / sport
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(model_dir.glob("model_v*.json"))
    version = len(existing) + 1
    path = model_dir / f"model_v{version}.json"
    payload = {
        "sport": sport,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
