from __future__ import annotations

import datetime as dt
import logging
import sys
from argparse import Namespace
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BACKTEST_REPORT_TXT, DATA_DIR, FOOTBALL_DATA_DIR, NBA_DATA_DIR, OUTPUTS_DIR, ensure_project_dirs, load_environment, project_relative  # noqa: E402
from core.prediction_result import PredictionResult  # noqa: E402
from core.utils import configure_logging  # noqa: E402
from sports.football.football_model import FootballPredictor  # noqa: E402
from sports.nba.nba_model import NBAPredictor  # noqa: E402


LOGGER = logging.getLogger("sports_predictor")


def main() -> None:
    ensure_project_dirs()
    load_environment()
    configure_logging(False)
    st.set_page_config(page_title="AI Sports Predictor", layout="wide")
    apply_theme()

    st.sidebar.markdown("## AI Sports Predictor")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Live Prediction", "Backtest Report", "Team Analysis", "Model Settings"],
        label_visibility="collapsed",
    )

    st.markdown("<h1>AI Sports Predictor</h1>", unsafe_allow_html=True)
    st.caption("Prediction dashboard for NBA and Football World Cup with Elo, momentum, fatigue, injuries, and calibration reports.")

    if page == "Dashboard":
        render_dashboard()
    elif page == "Live Prediction":
        render_live_prediction()
    elif page == "Backtest Report":
        render_backtest_report()
    elif page == "Team Analysis":
        render_team_analysis()
    elif page == "Model Settings":
        render_model_settings()


def render_dashboard() -> None:
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    master = read_csv(DATA_DIR / "predictions_master.csv")

    nba_accuracy = accuracy(nba)
    football_accuracy = accuracy(football)
    draw_accuracy = football_draw_accuracy(football)
    avg_confidence = average_confidence([nba, football])
    total_predictions = len(master) if not master.empty else len(nba) + len(football)

    cols = st.columns(5)
    metric_card(cols[0], "NBA Accuracy", percent(nba_accuracy), "Last backtest sample")
    metric_card(cols[1], "Football Accuracy", percent(football_accuracy), "Last backtest sample")
    metric_card(cols[2], "Football Draw Accuracy", percent(draw_accuracy), "Draw recall")
    metric_card(cols[3], "Average Confidence", percent(avg_confidence), "Across backtests")
    metric_card(cols[4], "Total Predictions", f"{total_predictions:,}", "Saved history")

    st.markdown("### Model Overview")
    left, right = st.columns(2)
    with left:
        render_accuracy_trend(nba, football)
    with right:
        render_confidence_distribution(nba, football)

    st.markdown("### Prediction History")
    render_prediction_history()

    st.markdown("### Today's NBA Predictions")
    render_today_predictions("nba")
    st.markdown("### Today's Football Predictions")
    render_today_predictions("football")


def render_live_prediction() -> None:
    st.markdown("### Live Prediction")
    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns([1.1, 1.2, 1.2, 1.0])
        sport = col1.selectbox("Sport Type", ["nba", "football"], format_func=lambda value: "NBA" if value == "nba" else "Football World Cup")
        date_value = col2.text_input("Date", "tomorrow")
        if sport == "football":
            home = col3.text_input("Home Team", "Mexico")
            away = col4.text_input("Away Team", "South Africa")
            mode = st.text_input("Mode", "WORLD_CUP")
        else:
            home = ""
            away = ""
            mode = ""
        show_injuries = st.checkbox("Include injury impact", value=True)
        submitted = st.form_submit_button("Run Prediction", type="primary")

    if not submitted:
        st.info("Choose a sport and run a prediction. NBA will use the selected date schedule; football requires home and away teams.")
        return

    with st.spinner("Running prediction..."):
        results = run_prediction(sport, date_value, home, away, mode, show_injuries)
    if not results:
        st.warning("No prediction was generated. If NBA has no games on that date, try another date such as 2026-04-12.")
        return

    for result in results:
        render_prediction_card(result)
        csv_path, txt_path = export_prediction(result)
        export_cols = st.columns(3)
        export_cols[0].download_button("Export prediction as CSV", csv_path.read_text(encoding="utf-8"), file_name=csv_path.name, mime="text/csv")
        export_cols[1].download_button("Export prediction as TXT", txt_path.read_text(encoding="utf-8"), file_name=txt_path.name, mime="text/plain")
        export_cols[2].success(f"Saved to outputs/{csv_path.name} and outputs/{txt_path.name}")


def render_prediction_card(result: PredictionResult) -> None:
    st.markdown(f"## {result.match}")
    cols = st.columns(5)
    metric_card(cols[0], "Home Win", percent(result.win_probability_home), result.home_team)
    metric_card(cols[1], "Away Win", percent(result.win_probability_away), result.away_team)
    metric_card(cols[2], "Draw", percent(result.draw_probability), "Football only")
    metric_card(cols[3], "Score", result.predicted_score, "Projected")
    metric_card(cols[4], "Confidence", result.confidence, "Calibrated")

    factor_groups = split_factors(result)
    group_cols = st.columns(4)
    mini_panel(group_cols[0], "Injury Impact", factor_groups["injury"])
    mini_panel(group_cols[1], "Elo Difference", factor_groups["elo"])
    mini_panel(group_cols[2], "Momentum", factor_groups["momentum"])
    mini_panel(group_cols[3], "Fatigue", factor_groups["fatigue"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Key Factors")
        render_factor_list(result.key_factors)
    with right:
        st.markdown("### Risk Factors")
        render_factor_list(result.risk_factors)


def render_backtest_report() -> None:
    st.markdown("### Backtest Report")
    report = BACKTEST_REPORT_TXT.read_text(encoding="utf-8") if BACKTEST_REPORT_TXT.exists() else ""
    if not report:
        st.info("No backtest report is available yet. Run the backtest commands from the terminal first.")
        return
    st.download_button("Export backtest report", report, file_name="backtest_report.txt", mime="text/plain")
    st.text_area("Historical performance report", report, height=360)

    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    col1, col2 = st.columns(2)
    with col1:
        render_accuracy_trend(nba, football)
    with col2:
        render_draw_calibration()


def render_team_analysis() -> None:
    st.markdown("### Team Analysis")
    query = st.text_input("Search by team name", "Mexico")
    frames = [
        read_csv(NBA_DATA_DIR / "nba_backtest_results.csv"),
        read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv"),
        read_csv(DATA_DIR / "predictions_master.csv"),
    ]
    history = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    if history.empty:
        st.info("No prediction or backtest history is available yet.")
        return
    if query:
        mask = history.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        history = history[mask]
    if history.empty:
        st.info("No matching team records found.")
        return

    col1, col2, col3 = st.columns(3)
    metric_card(col1, "Matched Records", f"{len(history):,}", "Filtered history")
    metric_card(col2, "Accuracy", percent(accuracy(history)), "Rows with result")
    metric_card(col3, "Avg Confidence", percent(frame_average_confidence(history)), "Predicted probability")
    st.dataframe(history.tail(200), use_container_width=True, hide_index=True)


def render_model_settings() -> None:
    st.markdown("### Model Settings")
    tuning_path = OUTPUTS_DIR / "model_weight_tuning.json"
    tuning = tuning_path.read_text(encoding="utf-8") if tuning_path.exists() else "{}"
    st.markdown("#### Current tuned weights")
    st.code(tuning, language="json")

    st.markdown("#### Data status")
    status_rows = [
        {"File": "NBA backtest", "Path": project_relative(NBA_DATA_DIR / "nba_backtest_results.csv"), "Available": (NBA_DATA_DIR / "nba_backtest_results.csv").exists()},
        {"File": "Football backtest", "Path": project_relative(FOOTBALL_DATA_DIR / "football_backtest_results.csv"), "Available": (FOOTBALL_DATA_DIR / "football_backtest_results.csv").exists()},
        {"File": "Elo ratings", "Path": project_relative(DATA_DIR / "elo_ratings.csv"), "Available": (DATA_DIR / "elo_ratings.csv").exists()},
        {"File": "Backtest report", "Path": project_relative(BACKTEST_REPORT_TXT), "Available": BACKTEST_REPORT_TXT.exists()},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Useful commands")
    st.code(
        "streamlit run app.py",
        language="powershell",
    )


def run_prediction(sport: str, date_value: str, home: str, away: str, mode: str, show_injuries: bool) -> list[PredictionResult]:
    args = Namespace(
        sport=sport,
        date=date_value,
        home=home,
        away=away,
        mode=mode,
        backtest=False,
        evaluate=False,
        injuries=show_injuries,
        season="2025-26",
        limit=100,
        verbose=False,
    )
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_prediction_request sport=%s date=%s", sport, date_value)
    return predictor.predict(args)


def run_live_prediction_for_ui(sport: str) -> list[PredictionResult]:
    args = Namespace(
        sport=sport,
        date=dt.date.today().isoformat(),
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
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_live_prediction_request sport=%s", sport)
    if sport == "football":
        return predictor.predict_live(args)
    return predictor.predict(args)


def render_today_predictions(sport: str) -> None:
    try:
        results = run_live_prediction_for_ui(sport)
    except Exception as exc:
        LOGGER.exception("streamlit_live_prediction_error sport=%s error=%s", sport, exc)
        st.warning("Today's predictions are temporarily unavailable. Cached reports and manual predictions are still available.")
        return
    if not results:
        st.info("No games found today. Live mode will use API/cache data when available.")
        return
    for result in results[:4]:
        cols = st.columns([1.6, 1, 1, 1])
        cols[0].markdown(f"**{result.match}**")
        cols[1].write(f"Home: {percent(result.win_probability_home)}")
        cols[2].write(f"Away: {percent(result.win_probability_away)}")
        cols[3].write(f"Draw: {percent(result.draw_probability)}")


def export_prediction(result: PredictionResult) -> tuple[Path, Path]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_match = safe_filename(f"{result.sport}_{result.home_team}_vs_{result.away_team}_{timestamp}")
    csv_path = OUTPUTS_DIR / f"{safe_match}.csv"
    txt_path = OUTPUTS_DIR / f"{safe_match}.txt"
    row = result.to_row()
    buffer = StringIO()
    pd.DataFrame([row]).to_csv(buffer, index=False)
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")
    txt_path.write_text(prediction_text(result), encoding="utf-8")
    LOGGER.info("export_history match=%s csv=%s txt=%s", result.match, csv_path.name, txt_path.name)
    return csv_path, txt_path


def prediction_text(result: PredictionResult) -> str:
    return "\n".join(
        [
            f"Match: {result.match}",
            f"Date: {result.prediction_date}",
            f"Predicted winner: {result.predicted_winner}",
            f"Predicted score: {result.predicted_score}",
            f"Home win probability: {percent(result.win_probability_home)}",
            f"Away win probability: {percent(result.win_probability_away)}",
            f"Draw probability: {percent(result.draw_probability)}",
            f"Confidence: {result.confidence}",
            "",
            "Key factors:",
            *[f"- {item}" for item in result.key_factors],
            "",
            "Risk factors:",
            *[f"- {item}" for item in result.risk_factors],
        ]
    )


def render_accuracy_trend(nba: pd.DataFrame, football: pd.DataFrame) -> None:
    st.markdown("#### Accuracy Trend")
    rows = []
    for sport, frame in (("NBA", nba), ("Football", football)):
        if frame.empty or "date" not in frame or "correct" not in frame:
            continue
        local = frame.copy()
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
        local["correct_num"] = local["correct"].astype(str).str.lower().isin(["true", "1"]).astype(float)
        local = local.dropna(subset=["date"]).sort_values("date")
        local["rolling_accuracy"] = local["correct_num"].rolling(12, min_periods=3).mean()
        local["sport"] = sport
        rows.append(local[["date", "sport", "rolling_accuracy"]])
    if not rows:
        st.info("Run backtests to populate the accuracy trend.")
        return
    trend = pd.concat(rows, ignore_index=True)
    st.line_chart(trend, x="date", y="rolling_accuracy", color="sport")


def render_confidence_distribution(nba: pd.DataFrame, football: pd.DataFrame) -> None:
    st.markdown("#### Confidence Distribution")
    rows = []
    for sport, frame in (("NBA", nba), ("Football", football)):
        if frame.empty or "confidence" not in frame:
            continue
        counts = frame["confidence"].value_counts().rename_axis("confidence").reset_index(name="games")
        counts["sport"] = sport
        rows.append(counts)
    if not rows:
        st.info("No confidence data available.")
        return
    st.bar_chart(pd.concat(rows, ignore_index=True), x="confidence", y="games", color="sport")


def render_prediction_history() -> None:
    frame = read_csv(OUTPUTS_DIR / "predictions.csv")
    if frame.empty:
        frame = read_csv(DATA_DIR / "predictions_master.csv")
    if frame.empty:
        st.info("No saved live predictions yet.")
        return
    st.dataframe(frame.tail(100), use_container_width=True, hide_index=True)


def render_draw_calibration() -> None:
    st.markdown("#### Draw Probability Calibration")
    frame = read_csv(FOOTBALL_DATA_DIR / "calibration_report.csv")
    if frame.empty:
        st.info("Football calibration data is not available.")
        return
    st.line_chart(frame, x="bucket", y=["avg_predicted_probability", "actual_win_rate"])


def split_factors(result: PredictionResult) -> dict[str, list[str]]:
    factors = result.key_factors + result.risk_factors
    return {
        "injury": filter_factors(factors, ["injury", "missing_starters"]),
        "elo": filter_factors(factors, ["elo_diff", "home_elo", "Elo"]),
        "momentum": filter_factors(factors, ["momentum", "recent"]),
        "fatigue": filter_factors(factors, ["fatigue", "rest_advantage", "travel_penalty", "back-to-back"]),
    }


def filter_factors(factors: list[str], needles: list[str]) -> list[str]:
    return [factor for factor in factors if any(needle.lower() in factor.lower() for needle in needles)][:3]


def render_factor_list(items: list[str]) -> None:
    if not items:
        st.write("No major factors available.")
        return
    for item in items:
        st.write(f"- {item}")


def mini_panel(container, title: str, items: list[str]) -> None:
    with container:
        st.markdown(f"<div class='mini-card'><b>{title}</b>", unsafe_allow_html=True)
        if items:
            for item in items[:2]:
                st.caption(item)
        else:
            st.caption("No major signal.")
        st.markdown("</div>", unsafe_allow_html=True)


def metric_card(container, label: str, value: str, caption: str) -> None:
    with container:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def accuracy(frame: pd.DataFrame) -> float:
    if frame.empty or "correct" not in frame:
        return 0.0
    return float(frame["correct"].astype(str).str.lower().isin(["true", "1"]).mean())


def football_draw_accuracy(frame: pd.DataFrame) -> float:
    if frame.empty or "actual_label" not in frame or "correct" not in frame:
        return 0.0
    draws = frame[frame["actual_label"] == "DRAW"]
    return accuracy(draws) if not draws.empty else 0.0


def average_confidence(frames: list[pd.DataFrame]) -> float:
    values = [frame_average_confidence(frame) for frame in frames if not frame.empty]
    values = [value for value in values if value > 0]
    return sum(values) / len(values) if values else 0.0


def frame_average_confidence(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    if "confidence_value" in frame:
        return float(pd.to_numeric(frame["confidence_value"], errors="coerce").dropna().mean())
    if "predicted_probability" in frame:
        return float(pd.to_numeric(frame["predicted_probability"], errors="coerce").dropna().mean())
    return 0.0


def percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return "_".join(cleaned.split("_"))


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1 {
            color: #111827;
            letter-spacing: 0;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 18px 18px 14px 18px;
            min-height: 118px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .metric-label {
            color: #6b7280;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .metric-value {
            color: #111827;
            font-size: 1.75rem;
            font-weight: 750;
            margin-top: 8px;
            overflow-wrap: anywhere;
        }
        .metric-caption {
            color: #6b7280;
            font-size: 0.86rem;
            margin-top: 6px;
        }
        .mini-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px;
            min-height: 132px;
        }
        div.stButton > button,
        div[data-testid="stDownloadButton"] > button {
            width: 100%;
            border-radius: 8px;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.75rem 1.5rem 0.75rem;
            }
            .metric-card {
                min-height: auto;
                padding: 14px;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 1.35rem;
            }
            .mini-card {
                min-height: auto;
                margin-bottom: 8px;
            }
            h1 {
                font-size: 1.75rem;
            }
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
