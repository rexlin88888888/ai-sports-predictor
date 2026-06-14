from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR, ensure_project_dirs, project_relative
from core.team_names import normalized_team_key
from data_pipeline.fetch_espn import fetch_espn_live_matches


TRACKER_CSV = OUTPUTS_DIR / "prediction_tracker.csv"
SCORE_REPORT_TXT = OUTPUTS_DIR / "prediction_tracker_score_report.txt"
SCORE_STATUS_JSON = OUTPUTS_DIR / "prediction_tracker_score_status.json"

TRACKER_SCORE_FIELDS = [
    "actual_score",
    "actual_result",
    "win_draw_loss_hit",
    "top1_score_hit",
    "top3_score_hit",
    "result_updated_at",
]


def score_prediction_tracker(
    tracker_path: Path = TRACKER_CSV,
    report_path: Path = SCORE_REPORT_TXT,
    lookback_days: int = 2,
) -> dict[str, Any]:
    """Update settled Prediction Tracker rows from ESPN final scores."""

    ensure_project_dirs()
    rows, fieldnames = read_tracker_rows(tracker_path)
    if not rows:
        report = {
            "tracker_rows": 0,
            "updated_rows": 0,
            "settled_rows": 0,
            "pending_rows": 0,
            "diagnostics": {"其它原因": 1},
            "message": "prediction_tracker.csv is empty or missing.",
        }
        write_score_report(report, [], report_path)
        return report

    for field in TRACKER_SCORE_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    target_dates = tracker_query_dates(rows, lookback_days)
    espn_rows, fetch_errors = fetch_espn_actual_rows(target_dates)
    indices = build_actual_indices(espn_rows)

    diagnostics: Counter[str] = Counter()
    updated = 0
    now_text = dt.datetime.now().isoformat(timespec="seconds")
    scored_rows: list[dict[str, str]] = []

    for row in rows:
        reason = ""
        actual = find_actual_for_tracker_row(row, indices)
        if actual is None:
            reason = diagnose_unscored_row(row, indices, fetch_errors)
            diagnostics[reason] += 1
            continue
        if actual["status"] != "finished":
            diagnostics["比赛未结束"] += 1
            continue

        old_score = str(row.get("actual_score") or "")
        row["actual_score"] = actual["actual_score"]
        row["actual_result"] = actual["actual_result"]
        row["win_draw_loss_hit"] = "1" if str(row.get("predicted_result") or "") == actual["actual_result"] else "0"
        row["top1_score_hit"] = "1" if clean_score(row.get("top_score_1") or row.get("predicted_score")) == actual["actual_score"] else "0"
        top_scores = {
            clean_score(row.get("top_score_1") or row.get("predicted_score")),
            clean_score(row.get("top_score_2")),
            clean_score(row.get("top_score_3")),
        }
        row["top3_score_hit"] = "1" if actual["actual_score"] in top_scores else "0"
        row["result_updated_at"] = now_text
        scored_rows.append(row)
        if old_score != actual["actual_score"]:
            updated += 1

    write_tracker_rows(tracker_path, rows, fieldnames)

    settled = [row for row in rows if str(row.get("actual_result") or "").strip()]
    pending = len(rows) - len(settled)
    report = {
        "tracker_rows": len(rows),
        "espn_dates_checked": len(target_dates),
        "espn_matches_fetched": len(espn_rows),
        "updated_rows": updated,
        "settled_rows": len(settled),
        "pending_rows": pending,
        "win_draw_loss_accuracy": ratio(settled, "win_draw_loss_hit"),
        "top1_score_accuracy": ratio(settled, "top1_score_hit"),
        "top3_score_accuracy": ratio(settled, "top3_score_hit"),
        "diagnostics": dict(diagnostics),
        "fetch_errors": fetch_errors,
        "generated_at": now_text,
    }
    write_score_report(report, settled, report_path)
    SCORE_STATUS_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def score_prediction_tracker_if_due(max_age_minutes: int = 10) -> dict[str, Any] | None:
    """Run tracker scoring at app startup, but avoid repeated network work on reruns."""

    if SCORE_STATUS_JSON.exists():
        try:
            payload = json.loads(SCORE_STATUS_JSON.read_text(encoding="utf-8"))
            generated_at = dt.datetime.fromisoformat(str(payload.get("generated_at") or ""))
            if dt.datetime.now() - generated_at < dt.timedelta(minutes=max_age_minutes):
                return None
        except Exception:
            pass
    return score_prediction_tracker()


def read_tracker_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_tracker_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def tracker_query_dates(rows: list[dict[str, str]], lookback_days: int) -> list[dt.date]:
    dates: set[dt.date] = set()
    today = dt.date.today()
    for row in rows:
        parsed = parse_date(row.get("match_time"))
        if parsed:
            dates.add(parsed)
    for offset in range(lookback_days + 2):
        dates.add(today - dt.timedelta(days=offset))
    return sorted(dates)


def fetch_espn_actual_rows(dates: list[dt.date]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for day in dates:
        try:
            rows.extend(fetch_espn_live_matches(day.strftime("%Y%m%d")))
        except Exception as exc:
            errors[day.isoformat()] = str(exc)
    return rows, errors


def build_actual_indices(rows: list[dict[str, Any]]) -> dict[str, dict[Any, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_team_date: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_team: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        actual = normalize_actual_row(row)
        if not actual:
            continue
        by_id[actual["match_id"]] = actual
        by_team_date[(actual["date"], normalized_team_key(actual["home_team"]), normalized_team_key(actual["away_team"]))] = actual
        by_team.setdefault((normalized_team_key(actual["home_team"]), normalized_team_key(actual["away_team"])), []).append(actual)
        by_date.setdefault(actual["date"], []).append(actual)
    return {"by_id": by_id, "by_team_date": by_team_date, "by_team": by_team, "by_date": by_date}


def normalize_actual_row(row: dict[str, Any]) -> dict[str, Any] | None:
    home_score = safe_int(row.get("home_score"))
    away_score = safe_int(row.get("away_score"))
    match_id = str(row.get("match_id") or "")
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    date_text = str(row.get("match_time_utc") or "")[:10]
    if not match_id or not home or not away or not date_text:
        return None
    status = str(row.get("status") or "").lower()
    actual: dict[str, Any] = {
        "match_id": match_id,
        "date": date_text,
        "home_team": home,
        "away_team": away,
        "status": status,
        "home_score": home_score,
        "away_score": away_score,
    }
    if status == "finished" and home_score is not None and away_score is not None:
        actual["actual_score"] = f"{home_score}:{away_score}"
        actual["actual_result"] = result_from_score(home_score, away_score)
    return actual


def find_actual_for_tracker_row(row: dict[str, str], indices: dict[str, dict[Any, Any]]) -> dict[str, Any] | None:
    match_id = str(row.get("match_id") or "")
    if match_id and match_id in indices["by_id"]:
        return indices["by_id"][match_id]

    match_date = parse_date(row.get("match_time"))
    home_key = normalized_team_key(row.get("home_team"))
    away_key = normalized_team_key(row.get("away_team"))
    if match_date:
        actual = indices["by_team_date"].get((match_date.isoformat(), home_key, away_key))
        if actual:
            return actual

    candidates = indices["by_team"].get((home_key, away_key), [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def diagnose_unscored_row(
    row: dict[str, str],
    indices: dict[str, dict[Any, Any]],
    fetch_errors: dict[str, str],
) -> str:
    match_date = parse_date(row.get("match_time"))
    if match_date and match_date.isoformat() in fetch_errors:
        return "ESPN查询失败"
    if match_date and match_date > dt.date.today() + dt.timedelta(days=1):
        return "比赛未结束"
    home_key = normalized_team_key(row.get("home_team"))
    away_key = normalized_team_key(row.get("away_team"))
    if indices["by_team"].get((home_key, away_key)):
        return "match_id不匹配"
    if match_date is None:
        return "日期格式错误"
    return "其它原因"


def parse_date(value: Any) -> dt.date | None:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def clean_score(value: Any) -> str:
    text = str(value or "")
    import re

    match = re.search(r"(\d+)\s*[:\-]\s*(\d+)", text)
    return f"{int(match.group(1))}:{int(match.group(2))}" if match else ""


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def result_from_score(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "HOME"
    if away_score > home_score:
        return "AWAY"
    return "DRAW"


def ratio(rows: list[dict[str, str]], column: str) -> float | None:
    values = [str(row.get(column) or "") for row in rows if str(row.get(column) or "") in {"0", "1"}]
    if not values:
        return None
    return sum(1 for value in values if value == "1") / len(values)


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def write_score_report(report: dict[str, Any], settled: list[dict[str, str]], path: Path) -> None:
    lines = [
        "Prediction Tracker Score Report",
        f"Generated: {report.get('generated_at') or dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Tracker rows: {report.get('tracker_rows', 0)}",
        f"ESPN dates checked: {report.get('espn_dates_checked', 0)}",
        f"ESPN matches fetched: {report.get('espn_matches_fetched', 0)}",
        f"Updated rows: {report.get('updated_rows', 0)}",
        f"Settled rows: {report.get('settled_rows', 0)}",
        f"Pending rows: {report.get('pending_rows', 0)}",
        f"Win/Draw/Loss accuracy: {pct(report.get('win_draw_loss_accuracy'))}",
        f"Top1 score accuracy: {pct(report.get('top1_score_accuracy'))}",
        f"Top3 score accuracy: {pct(report.get('top3_score_accuracy'))}",
        "",
        "Diagnostics",
    ]
    diagnostics = report.get("diagnostics") or {}
    if diagnostics:
        for reason, count in sorted(diagnostics.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No unscored diagnostics.")
    if report.get("fetch_errors"):
        lines.append("")
        lines.append("ESPN Fetch Errors")
        for day, error in sorted(report["fetch_errors"].items()):
            lines.append(f"- {day}: {error}")
    lines.append("")
    lines.append("Recent Settled Matches")
    for row in settled[-20:]:
        lines.append(
            f"- {row.get('match_time')} | {row.get('home_team')} vs {row.get('away_team')} | "
            f"pred {row.get('predicted_score')} | actual {row.get('actual_score')} | "
            f"WDL {row.get('win_draw_loss_hit')} | Top3 {row.get('top3_score_hit')}"
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    report = score_prediction_tracker()
    print("Prediction Tracker scoring complete")
    print(f"tracker={project_relative(TRACKER_CSV)}")
    print(f"report={project_relative(SCORE_REPORT_TXT)}")
    print(f"updated={report.get('updated_rows', 0)} settled={report.get('settled_rows', 0)} pending={report.get('pending_rows', 0)}")
    print(f"diagnostics={report.get('diagnostics', {})}")


if __name__ == "__main__":
    main()
