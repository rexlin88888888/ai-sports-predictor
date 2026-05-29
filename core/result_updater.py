from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..config import (
        DAILY_PREDICTIONS_CSV,
        DAILY_RESULT_RECAP_TXT,
        FOOTBALL_DATA_DIR,
        NBA_DATA_DIR,
        OUTPUT_PREDICTIONS_CSV,
        PERFORMANCE_REPORT_TXT,
        ensure_project_dirs,
        project_relative,
    )
    from .automation_status import update_automation_status
    from .utils import names_match
except ImportError:
    from config import (
        DAILY_PREDICTIONS_CSV,
        DAILY_RESULT_RECAP_TXT,
        FOOTBALL_DATA_DIR,
        NBA_DATA_DIR,
        OUTPUT_PREDICTIONS_CSV,
        PERFORMANCE_REPORT_TXT,
        ensure_project_dirs,
        project_relative,
    )
    from core.automation_status import update_automation_status
    from core.utils import names_match


NBA_RESULTS_CSV = NBA_DATA_DIR / "nba_games.csv"
FOOTBALL_RESULTS_CSV = FOOTBALL_DATA_DIR / "international_football.csv"


@dataclass(frozen=True)
class ActualResult:
    actual_score: str
    actual_result: str
    home_score: int
    away_score: int


def update_results() -> dict[str, Any]:
    ensure_project_dirs()
    daily = read_csv(DAILY_PREDICTIONS_CSV)
    if daily.empty:
        PERFORMANCE_REPORT_TXT.write_text("No daily predictions found.\n", encoding="utf-8")
        DAILY_RESULT_RECAP_TXT.write_text("No predictions are available for result recap.\n", encoding="utf-8")
        return {"updated": 0, "settled": 0, "pending": 0, "performance_report": str(PERFORMANCE_REPORT_TXT)}

    daily = normalize_prediction_frame(daily)
    updated = update_prediction_frame(daily)
    write_csv(DAILY_PREDICTIONS_CSV, updated)

    master = read_csv(OUTPUT_PREDICTIONS_CSV)
    if not master.empty:
        master = normalize_prediction_frame(master)
        master = update_prediction_frame(master)
        write_csv(OUTPUT_PREDICTIONS_CSV, master)

    report = build_performance_report(updated)
    PERFORMANCE_REPORT_TXT.write_text(report, encoding="utf-8")
    DAILY_RESULT_RECAP_TXT.write_text(build_result_recap(updated), encoding="utf-8")
    settled = settled_frame(updated)
    summary = {
        "updated": int((updated["actual_result"].astype(str) != "").sum()),
        "settled": len(settled),
        "pending": int((updated["actual_result"].astype(str) == "").sum()),
        "performance_report": project_relative(PERFORMANCE_REPORT_TXT),
        "recap": project_relative(DAILY_RESULT_RECAP_TXT),
    }
    update_automation_status(
        last_result_update=dt.datetime.now().isoformat(timespec="seconds"),
        last_result_status="success",
        settled_predictions=summary["settled"],
        pending_predictions=summary["pending"],
        performance_report=summary["performance_report"],
        result_recap=summary["recap"],
    )
    return summary


def update_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    local = normalize_prediction_frame(frame)
    for idx, row in local.iterrows():
        actual = fetch_actual_result(row)
        if actual is None:
            continue
        predicted = str(row.get("predicted_result") or row.get("predicted_winner") or "")
        local.at[idx, "actual_score"] = actual.actual_score
        local.at[idx, "actual_result"] = actual.actual_result
        local.at[idx, "prediction_correct"] = str(prediction_matches(predicted, actual.actual_result))
        local.at[idx, "result_updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    return local


def fetch_actual_result(row: pd.Series) -> ActualResult | None:
    sport = str(row.get("sport") or "").lower()
    if sport == "nba":
        return fetch_nba_result(row)
    if sport == "football":
        return fetch_football_result(row)
    return None


def fetch_nba_result(row: pd.Series) -> ActualResult | None:
    frame = read_csv(NBA_RESULTS_CSV)
    if frame.empty:
        return None
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(frame.columns):
        return None
    target_date = row_date(row)
    if target_date is None:
        return None
    for _, game in frame.iterrows():
        if parse_date(game.get("date")) != target_date:
            continue
        if names_match(str(game.get("home_team")), str(row.get("home_team"))) and names_match(str(game.get("away_team")), str(row.get("away_team"))):
            home_score = safe_int(game.get("home_score"))
            away_score = safe_int(game.get("away_score"))
            if home_score is None or away_score is None:
                return None
            actual_result = str(row.get("home_team")) if home_score > away_score else str(row.get("away_team"))
            return ActualResult(
                actual_score=f"{row.get('home_team')} {home_score} - {away_score} {row.get('away_team')}",
                actual_result=actual_result,
                home_score=home_score,
                away_score=away_score,
            )
    return None


def fetch_football_result(row: pd.Series) -> ActualResult | None:
    frame = read_csv(FOOTBALL_RESULTS_CSV)
    if frame.empty:
        return None
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(frame.columns):
        return None
    target_date = row_date(row)
    if target_date is None:
        return None
    for _, game in frame.iterrows():
        if parse_date(game.get("date")) != target_date:
            continue
        if names_match(str(game.get("home_team")), str(row.get("home_team"))) and names_match(str(game.get("away_team")), str(row.get("away_team"))):
            home_score = safe_int(game.get("home_score"))
            away_score = safe_int(game.get("away_score"))
            if home_score is None or away_score is None:
                return None
            if home_score > away_score:
                actual_result = str(row.get("home_team"))
            elif away_score > home_score:
                actual_result = str(row.get("away_team"))
            else:
                actual_result = "Draw"
            return ActualResult(
                actual_score=f"{row.get('home_team')} {home_score} - {away_score} {row.get('away_team')}",
                actual_result=actual_result,
                home_score=home_score,
                away_score=away_score,
            )
    return None


def build_performance_report(frame: pd.DataFrame) -> str:
    local = normalize_prediction_frame(frame)
    settled = settled_frame(local)
    pending = local[local["actual_result"].astype(str) == ""]
    lines = [
        "AI Sports Predictor Performance Report",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"total predictions: {len(local)}",
        f"settled predictions: {len(settled)}",
        f"pending predictions: {len(pending)}",
        f"overall accuracy: {format_pct(accuracy(settled))}",
        f"NBA accuracy: {format_pct(accuracy(settled[settled['sport'].astype(str).str.lower() == 'nba']))}",
        f"Football accuracy: {format_pct(accuracy(settled[settled['sport'].astype(str).str.lower() == 'football']))}",
        f"draw accuracy: {format_pct(draw_accuracy(settled))}",
        f"high confidence accuracy: {format_pct(category_accuracy(settled, confidence='High'))}",
        f"upset alert accuracy: {format_pct(category_accuracy(settled, category='upset_watch'))}",
        f"best value accuracy: {format_pct(category_accuracy(settled, category='best_value'))}",
    ]
    return "\n".join(lines) + "\n"


def build_result_recap(frame: pd.DataFrame) -> str:
    local = normalize_prediction_frame(frame)
    settled = settled_frame(local)
    hits = settled[settled["prediction_correct"].astype(str).str.lower() == "true"]
    misses = settled[settled["prediction_correct"].astype(str).str.lower() == "false"]
    best = best_call(hits)
    upset_missed = biggest_upset_missed(misses)
    lines = [
        "Daily Result Recap",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Yesterday's hits",
        *row_lines(hits),
        "",
        "Missed predictions",
        *row_lines(misses),
        "",
        f"Best call: {best}",
        f"Biggest upset missed: {upset_missed}",
        "Model learning notes: Keep tracking settled games before changing model weights. Pending games are excluded from accuracy.",
    ]
    return "\n".join(lines).strip() + "\n"


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    if "date" not in local and "prediction_date" in local:
        local["date"] = local["prediction_date"]
    if "prediction_date" not in local and "date" in local:
        local["prediction_date"] = local["date"]
    if "predicted_result" not in local and "predicted_winner" in local:
        local["predicted_result"] = local["predicted_winner"]
    for column in ("actual_score", "actual_result", "prediction_correct", "result_updated_at"):
        if column not in local:
            local[column] = ""
    return local.fillna("")


def settled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "actual_result" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["actual_result"].astype(str) != ""].copy()


def accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty or "prediction_correct" not in frame:
        return None
    values = frame["prediction_correct"].astype(str).str.lower()
    values = values[values.isin(["true", "false"])]
    if values.empty:
        return None
    return float((values == "true").mean())


def draw_accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty or "actual_result" not in frame:
        return None
    draws = frame[frame["actual_result"].astype(str).str.lower() == "draw"]
    return accuracy(draws)


def category_accuracy(frame: pd.DataFrame, category: str | None = None, confidence: str | None = None) -> float | None:
    local = frame
    if category and "pick_category" in local:
        local = local[local["pick_category"].astype(str).str.contains(category, case=False, na=False)]
    if confidence and "confidence" in local:
        local = local[local["confidence"].astype(str).str.lower() == confidence.lower()]
    return accuracy(local)


def prediction_matches(predicted: str, actual: str) -> bool:
    if not predicted or not actual:
        return False
    if actual.lower() == "draw":
        return predicted.lower() == "draw"
    return names_match(predicted, actual)


def row_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- None yet."]
    lines = []
    for _, row in frame.tail(8).iterrows():
        lines.append(
            f"- {row.get('match') or str(row.get('home_team')) + ' vs ' + str(row.get('away_team'))}: "
            f"predicted {row.get('predicted_result')}, actual {row.get('actual_result')} ({row.get('actual_score')})"
        )
    return lines


def best_call(hits: pd.DataFrame) -> str:
    if hits.empty:
        return "No settled winning calls yet."
    local = hits.copy()
    local["_prob"] = pd.to_numeric(local.get("top_probability", 0), errors="coerce").fillna(0)
    row = local.sort_values("_prob", ascending=False).iloc[0]
    return f"{row.get('predicted_result')} in {row.get('match')} ({row.get('actual_score')})"


def biggest_upset_missed(misses: pd.DataFrame) -> str:
    if misses.empty:
        return "No missed upsets yet."
    candidates = misses[misses.get("pick_category", "").astype(str).str.contains("upset", case=False, na=False)] if "pick_category" in misses else misses
    if candidates.empty:
        candidates = misses
    row = candidates.iloc[0]
    return f"{row.get('match')}: predicted {row.get('predicted_result')}, actual {row.get('actual_result')}"


def row_date(row: pd.Series) -> dt.date | None:
    return parse_date(row.get("date") or row.get("prediction_date"))


def parse_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    print(update_results())
