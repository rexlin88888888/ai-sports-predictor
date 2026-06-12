from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from config import DAILY_PREDICTIONS_CSV, OUTPUTS_DIR, ensure_project_dirs, project_relative
    from core.prediction_result import current_model_version
    from core.team_names import normalize_team_name, normalized_team_key
    from data_pipeline.db import fetch_all, initialize_database
    from data_pipeline.fetch_espn import fetch_espn_live_matches, persist_espn_match
    from data_pipeline.fetch_schedule import fetch_openfootball_schedule
    from sports.football.football_data import load_matches
    from sports.football.football_model import predict_football_match
except ImportError:  # pragma: no cover
    from ..config import DAILY_PREDICTIONS_CSV, OUTPUTS_DIR, ensure_project_dirs, project_relative
    from .prediction_result import current_model_version
    from .team_names import normalize_team_name, normalized_team_key
    from ..data_pipeline.db import fetch_all, initialize_database
    from ..data_pipeline.fetch_espn import fetch_espn_live_matches, persist_espn_match
    from ..data_pipeline.fetch_schedule import fetch_openfootball_schedule
    from ..sports.football.football_data import load_matches
    from ..sports.football.football_model import predict_football_match


TRACKER_CSV = OUTPUTS_DIR / "prediction_tracker.csv"
TRACKER_REPORT_TXT = OUTPUTS_DIR / "prediction_tracker_report.txt"


DAILY_FIELDNAMES = [
    "prediction_generated_at",
    "match_id",
    "match_time",
    "home_team",
    "away_team",
    "predicted_score",
    "second_predicted_score",
    "third_predicted_score",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "model_odds",
    "model_odds_home",
    "model_odds_draw",
    "model_odds_away",
    "xg_home",
    "xg_away",
    "data_source",
    "data_updated_at",
    "model_version",
]


TRACKER_FIELDNAMES = [
    "prediction_time",
    "match_id",
    "match_time",
    "sport",
    "home_team",
    "away_team",
    "prediction_content",
    "predicted_result",
    "predicted_score",
    "top_score_1",
    "top_score_2",
    "top_score_3",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "xg_home",
    "xg_away",
    "actual_score",
    "actual_result",
    "win_draw_loss_hit",
    "top1_score_hit",
    "top3_score_hit",
    "result_updated_at",
    "data_source",
    "model_version",
]


@dataclass(frozen=True)
class RealFixture:
    match_id: str
    match_time: str
    home_team: str
    away_team: str
    status: str
    home_score: int | None
    away_score: int | None
    data_source: str
    data_updated_at: str
    mode: str = "WORLD_CUP"

    @property
    def match_date(self) -> dt.date:
        try:
            return dt.date.fromisoformat(self.match_time[:10])
        except ValueError:
            return dt.date.today()


def run_real_match_validation(days_forward: int = 7) -> dict[str, Any]:
    """Generate real fixture predictions and update the validation tracker.

    The football model is treated as read-only here. This module only handles
    real fixture collection, CSV output, and result tracking.
    """

    ensure_project_dirs()
    initialize_database()
    start_date = dt.date.today()
    fetched = refresh_real_sources(start_date, days_forward)
    fixtures = collect_real_fixtures(start_date, days_forward)
    prediction_rows = build_prediction_rows(fixtures)
    write_csv(DAILY_PREDICTIONS_CSV, DAILY_FIELDNAMES, prediction_rows)
    upsert_tracker(prediction_rows)
    updated_results = update_tracker_results(start_date, days_forward)
    stats = build_tracker_report()
    write_tracker_report(stats, fixtures, fetched, updated_results)
    return {
        "fixtures": fixtures,
        "predictions": prediction_rows,
        "stats": stats,
        "espn_fetched": fetched["espn_matches"],
        "openfootball_fetched": fetched["openfootball_matches"],
        "tracker_updates": updated_results,
        "daily_csv": DAILY_PREDICTIONS_CSV,
        "tracker_csv": TRACKER_CSV,
        "report_txt": TRACKER_REPORT_TXT,
    }


def refresh_real_sources(start_date: dt.date, days_forward: int) -> dict[str, int]:
    espn_count = 0
    openfootball_count = 0
    for offset in range(days_forward + 1):
        day = start_date + dt.timedelta(days=offset)
        try:
            matches = fetch_espn_live_matches(day.strftime("%Y%m%d"))
            for match in matches:
                persist_espn_match(match)
            espn_count += len(matches)
        except Exception:
            continue
    try:
        openfootball_count = len(fetch_openfootball_schedule())
    except Exception:
        openfootball_count = 0
    return {"espn_matches": espn_count, "openfootball_matches": openfootball_count}


def collect_real_fixtures(start_date: dt.date, days_forward: int) -> list[RealFixture]:
    end_date = start_date + dt.timedelta(days=days_forward)
    rows = fetch_all(
        """
        SELECT match_id, match_time_utc, home_team, away_team, status, home_score, away_score,
               stage, data_source, data_timestamp
        FROM matches
        WHERE substr(match_time_utc, 1, 10) >= ?
          AND substr(match_time_utc, 1, 10) <= ?
          AND lower(coalesce(status, '')) <> 'cancelled'
        ORDER BY match_time_utc, home_team, away_team
        """,
        (start_date.isoformat(), end_date.isoformat()),
    )
    fixtures_by_key: dict[tuple[str, str, str], RealFixture] = {}
    for row in rows:
        home = normalize_team_name(row.get("home_team"))
        away = normalize_team_name(row.get("away_team"))
        match_time = str(row.get("match_time_utc") or "")
        if not home or not away or not match_time:
            continue
        key = (match_time[:10], normalized_team_key(home), normalized_team_key(away))
        fixture = RealFixture(
            match_id=str(row.get("match_id") or stable_match_id(home, away, match_time[:10])),
            match_time=match_time,
            home_team=home,
            away_team=away,
            status=str(row.get("status") or "scheduled"),
            home_score=to_int(row.get("home_score")),
            away_score=to_int(row.get("away_score")),
            data_source=clean_source(row.get("data_source")),
            data_updated_at=str(row.get("data_timestamp") or dt.datetime.now(dt.UTC).isoformat(timespec="seconds")),
            mode=str(row.get("stage") or "WORLD_CUP"),
        )
        existing = fixtures_by_key.get(key)
        if existing is None or source_priority(fixture.data_source) > source_priority(existing.data_source):
            fixtures_by_key[key] = fixture
    return list(fixtures_by_key.values())


def build_prediction_rows(fixtures: list[RealFixture]) -> list[dict[str, Any]]:
    matches = load_matches()
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    version = current_model_version()
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        prediction = predict_football_match(
            matches,
            fixture.home_team,
            fixture.away_team,
            fixture.mode or "WORLD_CUP",
            fixture.match_date,
            data_source=fixture.data_source,
        )
        top_scores = extract_top_scores(prediction.key_factors)
        xg_home, xg_away = extract_xg(prediction.key_factors)
        odds_home = probability_to_odds(prediction.win_probability_home)
        odds_draw = probability_to_odds(prediction.draw_probability)
        odds_away = probability_to_odds(prediction.win_probability_away)
        predicted_score = top_scores[0]["score"] if top_scores else compact_score(prediction.predicted_score)
        second_score = top_scores[1]["score_with_probability"] if len(top_scores) > 1 else ""
        third_score = top_scores[2]["score_with_probability"] if len(top_scores) > 2 else ""
        row = {
            "prediction_generated_at": generated_at,
            "match_id": fixture.match_id,
            "match_time": fixture.match_time,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "predicted_score": predicted_score,
            "second_predicted_score": second_score,
            "third_predicted_score": third_score,
            "home_win_probability": format_probability(prediction.win_probability_home),
            "draw_probability": format_probability(prediction.draw_probability),
            "away_win_probability": format_probability(prediction.win_probability_away),
            "model_odds": f"H {odds_home} / D {odds_draw} / A {odds_away}",
            "model_odds_home": odds_home,
            "model_odds_draw": odds_draw,
            "model_odds_away": odds_away,
            "xg_home": "" if xg_home is None else f"{xg_home:.2f}",
            "xg_away": "" if xg_away is None else f"{xg_away:.2f}",
            "data_source": fixture.data_source,
            "data_updated_at": fixture.data_updated_at,
            "model_version": version,
        }
        rows.append(row)
    return rows


def upsert_tracker(prediction_rows: list[dict[str, Any]]) -> None:
    existing_rows = read_csv_rows(TRACKER_CSV)
    existing_by_id = {str(row.get("match_id") or ""): row for row in existing_rows if row.get("match_id")}
    for prediction in prediction_rows:
        match_id = str(prediction["match_id"])
        current = existing_by_id.get(match_id, {})
        content = (
            f"{prediction['home_team']} vs {prediction['away_team']}: "
            f"{prediction['predicted_score']} "
            f"(top3 {prediction['predicted_score']}, {prediction['second_predicted_score']}, {prediction['third_predicted_score']}); "
            f"H {prediction['home_win_probability']} D {prediction['draw_probability']} A {prediction['away_win_probability']}"
        )
        tracker_row = {
            **{field: "" for field in TRACKER_FIELDNAMES},
            **current,
            "prediction_time": current.get("prediction_time") or prediction["prediction_generated_at"],
            "match_id": match_id,
            "match_time": prediction["match_time"],
            "sport": "football",
            "home_team": prediction["home_team"],
            "away_team": prediction["away_team"],
            "prediction_content": content,
            "predicted_result": predicted_result_from_probabilities(prediction),
            "predicted_score": prediction["predicted_score"],
            "top_score_1": prediction["predicted_score"],
            "top_score_2": score_without_probability(str(prediction["second_predicted_score"])),
            "top_score_3": score_without_probability(str(prediction["third_predicted_score"])),
            "home_win_probability": prediction["home_win_probability"],
            "draw_probability": prediction["draw_probability"],
            "away_win_probability": prediction["away_win_probability"],
            "xg_home": prediction["xg_home"],
            "xg_away": prediction["xg_away"],
            "data_source": prediction["data_source"],
            "model_version": prediction["model_version"],
        }
        existing_by_id[match_id] = tracker_row
    ordered = sorted(existing_by_id.values(), key=lambda row: str(row.get("match_time") or ""))
    write_csv(TRACKER_CSV, TRACKER_FIELDNAMES, ordered)


def update_tracker_results(start_date: dt.date, days_forward: int) -> int:
    rows = read_csv_rows(TRACKER_CSV)
    if not rows:
        return 0
    actuals = actual_results_by_match_id(start_date - dt.timedelta(days=30), days_forward + 30)
    updated = 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    for row in rows:
        match_id = str(row.get("match_id") or "")
        actual = actuals.get(match_id)
        if not actual:
            continue
        old_actual = row.get("actual_score")
        row["actual_score"] = actual["actual_score"]
        row["actual_result"] = actual["actual_result"]
        row["win_draw_loss_hit"] = "1" if str(row.get("predicted_result")) == actual["actual_result"] else "0"
        row["top1_score_hit"] = "1" if str(row.get("top_score_1")) == actual["actual_score"] else "0"
        top_scores = {str(row.get("top_score_1")), str(row.get("top_score_2")), str(row.get("top_score_3"))}
        row["top3_score_hit"] = "1" if actual["actual_score"] in top_scores else "0"
        row["result_updated_at"] = now
        if old_actual != row["actual_score"]:
            updated += 1
    write_csv(TRACKER_CSV, TRACKER_FIELDNAMES, rows)
    return updated


def actual_results_by_match_id(start_date: dt.date, days_forward: int) -> dict[str, dict[str, str]]:
    actuals: dict[str, dict[str, str]] = {}
    for offset in range(days_forward + 1):
        day = start_date + dt.timedelta(days=offset)
        try:
            matches = fetch_espn_live_matches(day.strftime("%Y%m%d"))
        except Exception:
            matches = []
        for match in matches:
            if str(match.get("status") or "").lower() != "finished":
                continue
            home_score = to_int(match.get("home_score"))
            away_score = to_int(match.get("away_score"))
            if home_score is None or away_score is None:
                continue
            actuals[str(match["match_id"])] = {
                "actual_score": f"{home_score}:{away_score}",
                "actual_result": result_from_score(home_score, away_score),
            }
    db_rows = fetch_all(
        """
        SELECT match_id, home_score, away_score
        FROM matches
        WHERE lower(coalesce(status, '')) = 'finished'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        """
    )
    for row in db_rows:
        home_score = to_int(row.get("home_score"))
        away_score = to_int(row.get("away_score"))
        if home_score is None or away_score is None:
            continue
        actuals.setdefault(
            str(row.get("match_id") or ""),
            {"actual_score": f"{home_score}:{away_score}", "actual_result": result_from_score(home_score, away_score)},
        )
    return actuals


def build_tracker_report() -> dict[str, Any]:
    rows = read_csv_rows(TRACKER_CSV)
    now = dt.date.today()
    return {
        "last_7_days": tracker_stats(rows, now - dt.timedelta(days=7), now + dt.timedelta(days=1)),
        "last_30_days": tracker_stats(rows, now - dt.timedelta(days=30), now + dt.timedelta(days=1)),
        "all_time": tracker_stats(rows, None, None),
    }


def tracker_stats(rows: list[dict[str, str]], start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        try:
            match_date = dt.date.fromisoformat(str(row.get("match_time") or "")[:10])
        except ValueError:
            continue
        if start and match_date < start:
            continue
        if end and match_date >= end:
            continue
        if not row.get("actual_result"):
            continue
        filtered.append(row)
    return {
        "settled_predictions": len(filtered),
        "win_draw_loss_accuracy": ratio(filtered, "win_draw_loss_hit"),
        "top1_score_accuracy": ratio(filtered, "top1_score_hit"),
        "top3_score_accuracy": ratio(filtered, "top3_score_hit"),
    }


def write_tracker_report(stats: dict[str, Any], fixtures: list[RealFixture], fetched: dict[str, int], updated_results: int) -> None:
    lines = [
        "Real Match Prediction Tracker",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Future window fixtures: {len(fixtures)}",
        f"ESPN matches fetched: {fetched['espn_matches']}",
        f"openfootball matches fetched: {fetched['openfootball_matches']}",
        f"Result rows updated: {updated_results}",
        "",
    ]
    for label in ("last_7_days", "last_30_days", "all_time"):
        item = stats[label]
        lines.extend(
            [
                label.replace("_", " ").title(),
                f"Settled predictions: {item['settled_predictions']}",
                f"Win/Draw/Loss accuracy: {format_metric(item['win_draw_loss_accuracy'])}",
                f"Top1 score accuracy: {format_metric(item['top1_score_accuracy'])}",
                f"Top3 score accuracy: {format_metric(item['top3_score_accuracy'])}",
                "",
            ]
        )
    TRACKER_REPORT_TXT.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def extract_top_scores(key_factors: list[str]) -> list[dict[str, str]]:
    text = " ".join(key_factors)
    match = re.search(r"most_likely_scores=([^ |]+)", text)
    if not match:
        match = re.search(r"score_probabilities=([^ |]+)", text)
    if not match:
        return []
    scores: list[dict[str, str]] = []
    for raw in match.group(1).split(",")[:3]:
        parts = raw.split(":")
        if len(parts) != 3:
            continue
        try:
            home_goals = int(parts[0])
            away_goals = int(parts[1])
            probability = float(parts[2])
        except ValueError:
            continue
        score = f"{home_goals}:{away_goals}"
        scores.append({"score": score, "score_with_probability": f"{score} ({probability:.1%})"})
    return scores


def extract_xg(key_factors: list[str]) -> tuple[float | None, float | None]:
    text = " ".join(key_factors)
    match = re.search(r"xg_home=([0-9.]+),\s*xg_away=([0-9.]+)", text)
    if not match:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def predicted_result_from_probabilities(row: dict[str, Any]) -> str:
    home = parse_probability(row.get("home_win_probability"))
    draw = parse_probability(row.get("draw_probability"))
    away = parse_probability(row.get("away_win_probability"))
    if draw >= home and draw >= away:
        return "DRAW"
    return "HOME" if home >= away else "AWAY"


def result_from_score(home_score: int, away_score: int) -> str:
    if home_score == away_score:
        return "DRAW"
    return "HOME" if home_score > away_score else "AWAY"


def compact_score(value: str) -> str:
    match = re.search(r"(\d+)\s*-\s*(\d+)", value)
    if match:
        return f"{int(match.group(1))}:{int(match.group(2))}"
    return value


def score_without_probability(value: str) -> str:
    match = re.search(r"(\d+)\s*:\s*(\d+)", value)
    return f"{match.group(1)}:{match.group(2)}" if match else ""


def probability_to_odds(value: float | None) -> str:
    if not value or value <= 0:
        return ""
    return f"{1.0 / value:.2f}"


def format_probability(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def parse_probability(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ratio(rows: list[dict[str, str]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) in {"0", "1"}]
    if not values:
        return None
    return sum(int(value) for value in values) / len(values)


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def source_priority(source: str) -> int:
    if "ESPN" in source:
        return 30
    if "openfootball" in source:
        return 20
    if "database" in source:
        return 10
    return 0


def clean_source(value: Any) -> str:
    source = str(value or "").strip()
    if "ESPN" in source:
        return "ESPN"
    if "openfootball" in source:
        return "openfootball"
    if source.startswith("database"):
        return source
    return source or "database"


def stable_match_id(home: str, away: str, match_date: str) -> str:
    safe_home = re.sub(r"[^A-Za-z0-9]+", "_", normalize_team_name(home)).strip("_")
    safe_away = re.sub(r"[^A-Za-z0-9]+", "_", normalize_team_name(away)).strip("_")
    return f"wc_{match_date}_{safe_home}_vs_{safe_away}"


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Daily predictions: {project_relative(summary['daily_csv'])}")
    print(f"Prediction tracker: {project_relative(summary['tracker_csv'])}")
    print(f"Tracker report: {project_relative(summary['report_txt'])}")
    print(f"ESPN matches fetched: {summary['espn_fetched']}")
    print(f"openfootball matches fetched: {summary['openfootball_fetched']}")
    print(f"Future window predictions: {len(summary['predictions'])}")
    print("")
    for row in summary["predictions"]:
        print(
            f"{row['match_time']} | {row['home_team']} vs {row['away_team']} | "
            f"{row['predicted_score']} | H {row['home_win_probability']} "
            f"D {row['draw_probability']} A {row['away_win_probability']} | {row['data_source']}"
        )


def main() -> None:
    summary = run_real_match_validation(days_forward=7)
    print_summary(summary)


if __name__ == "__main__":
    main()
