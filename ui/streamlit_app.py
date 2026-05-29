from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import sys
from argparse import Namespace
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BACKTEST_REPORT_TXT, CACHE_DIR, DAILY_RESULT_RECAP_TXT, DAILY_SHORT_SCRIPT_TXT, DAILY_SOCIAL_POSTS_TXT, DATA_DIR, FOOTBALL_DATA_DIR, MODEL_VERSION_JSON, NBA_DATA_DIR, OUTPUTS_DIR, PERFORMANCE_REPORT_TXT, ensure_project_dirs, load_environment, project_relative  # noqa: E402
from core.automation_status import read_automation_status  # noqa: E402
from core.daily_predictions import DailyPredictionPackage, build_daily_prediction_package, generate_daily_predictions  # noqa: E402
from core.prediction_result import PredictionResult  # noqa: E402
from core.result_updater import update_results  # noqa: E402
from core.utils import configure_logging  # noqa: E402
from sports.football.football_model import FootballPredictor  # noqa: E402
from sports.nba.nba_model import NBAPredictor  # noqa: E402


LOGGER = logging.getLogger("sports_predictor")
NAV_ITEMS = [
    "Dashboard",
    "Live Predictions",
    "Content Studio",
    "Install App",
    "NBA",
    "Football",
    "Team Analysis",
    "Results Tracker",
    "Prediction History",
    "Backtest Reports",
    "Settings",
]


def main() -> None:
    ensure_project_dirs()
    load_environment()
    configure_logging(False)
    st.set_page_config(page_title="AI Sports Predictor", layout="wide", page_icon="AI")

    theme_mode = st.sidebar.selectbox("Display Mode", ["Dark", "Light", "Auto"], index=0)
    apply_theme(theme_mode)
    inject_pwa_assets()
    st.sidebar.markdown(
        """
        <div class="brand-block">
            <div class="brand-mark">AI</div>
            <div>
                <div class="brand-title">AI Sports Predictor</div>
                <div class="brand-subtitle">Live model dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sync_quick_navigation()
    page = st.sidebar.radio("Navigation", NAV_ITEMS, index=1, key="nav_choice", label_visibility="collapsed")
    render_sidebar_status()
    render_header(page)

    if page == "Dashboard":
        render_dashboard()
    elif page == "Live Predictions":
        render_live_predictions_page()
    elif page == "Content Studio":
        render_content_studio()
    elif page == "Install App":
        render_install_app_page()
    elif page == "NBA":
        render_nba_page()
    elif page == "Football":
        render_football_page()
    elif page == "Team Analysis":
        render_team_analysis()
    elif page == "Results Tracker":
        render_results_tracker()
    elif page == "Prediction History":
        render_prediction_history_page()
    elif page == "Backtest Reports":
        render_backtest_report()
    elif page == "Settings":
        render_model_settings()
    render_footer()


def sync_quick_navigation() -> None:
    target = st.session_state.pop("quick_nav_target", None)
    if target in NAV_ITEMS:
        st.session_state["nav_choice"] = target


def go_to_page(page: str) -> None:
    st.session_state["quick_nav_target"] = page
    st.rerun()


def inject_pwa_assets() -> None:
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        const manifestHref = "/app/static/manifest.json";
        const icon192 = "/app/static/assets/icon-192.png";
        const icon512 = "/app/static/assets/icon-512.png";

        function upsertLink(selector, attrs) {
          let el = parentDoc.querySelector(selector);
          if (!el) {
            el = parentDoc.createElement("link");
            parentDoc.head.appendChild(el);
          }
          Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
        }

        function upsertMeta(name, content) {
          let el = parentDoc.querySelector(`meta[name="${name}"]`);
          if (!el) {
            el = parentDoc.createElement("meta");
            el.setAttribute("name", name);
            parentDoc.head.appendChild(el);
          }
          el.setAttribute("content", content);
        }

        parentDoc.title = "AI Sports Predictor";
        upsertLink('link[rel="manifest"]', { rel: "manifest", href: manifestHref });
        upsertLink('link[rel="apple-touch-icon"]', { rel: "apple-touch-icon", href: icon192 });
        upsertLink('link[rel="icon"]', { rel: "icon", type: "image/png", sizes: "192x192", href: icon192 });
        upsertMeta("theme-color", "#0b1f3a");
        upsertMeta("apple-mobile-web-app-capable", "yes");
        upsertMeta("apple-mobile-web-app-title", "Sports AI");
        upsertMeta("apple-mobile-web-app-status-bar-style", "black-translucent");

        if ("serviceWorker" in window.parent.navigator) {
          window.parent.navigator.serviceWorker.register("/app/static/service-worker.js").catch(() => undefined);
        }
        </script>
        """,
        height=0,
    )


def render_header(page: str) -> None:
    updated = dt.datetime.now().strftime("%b %d, %Y %H:%M")
    live_status = "Live APIs enabled" if (os.getenv("NEWS_API_KEY") or os.getenv("FOOTBALL_DATA_KEY")) else "Fallback mode ready"
    st.markdown(
        f"""
        <section class="top-header">
            <div class="header-left">
                <div class="app-logo">A</div>
                <div>
                    <h1>AI Sports Predictor</h1>
                    <p>{html.escape(page)} · NBA and Football forecasts with Elo, momentum, fatigue and injury signals.</p>
                </div>
            </div>
            <div class="header-meta">
                <div class="status-pill live-dot">{html.escape(live_status)}</div>
                <div class="updated-pill">Last updated {html.escape(updated)}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status() -> None:
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    automation = read_automation_status()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="panel-label">Model status</div>
            <div class="sidebar-row"><span>NBA accuracy</span><b>{percent(accuracy(nba))}</b></div>
            <div class="sidebar-row"><span>Football accuracy</span><b>{percent(accuracy(football))}</b></div>
            <div class="sidebar-row"><span>Draw model</span><b>{percent(football_draw_accuracy(football))}</b></div>
            <div class="sidebar-row"><span>Automation</span><b>{html.escape(str(automation.get("automation_status") or automation.get("last_daily_status") or "ready"))}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    master = read_prediction_history()
    cols = st.columns(5)
    metric_card(cols[0], "NBA Accuracy", percent(accuracy(nba)), "Last backtest sample", "positive")
    metric_card(cols[1], "Football Accuracy", percent(accuracy(football)), "Last backtest sample", "positive")
    metric_card(cols[2], "Football Draw Accuracy", percent(football_draw_accuracy(football)), "Draw recall", "neutral")
    metric_card(cols[3], "Average Confidence", percent(average_confidence([nba, football])), "Across backtests", "neutral")
    metric_card(cols[4], "Total Predictions", f"{len(master):,}", "Saved history", "accent")
    render_quick_actions()
    render_automation_overview()

    st.markdown("### Model Overview")
    left, right = st.columns(2)
    with left:
        render_accuracy_trend(nba, football)
    with right:
        render_confidence_distribution(nba, football)
    st.markdown("### Latest Predictions")
    render_history_table(master.tail(12), compact=True)


def render_quick_actions() -> None:
    st.markdown("### Quick Actions")
    cols = st.columns(4)
    actions = [
        ("Today's Picks", "Live Predictions", "Open today's model board"),
        ("Highest Confidence", "Live Predictions", "Review top confidence cards"),
        ("Content Studio", "Content Studio", "Copy social posts"),
        ("Results Tracker", "Results Tracker", "Check settled picks"),
    ]
    for col, (label, target, caption) in zip(cols, actions):
        with col:
            st.markdown(
                f"""
                <div class="quick-card">
                    <div class="quick-title">{html.escape(label)}</div>
                    <div class="quick-caption">{html.escape(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"quick_{target}_{label}"):
                go_to_page(target)


def render_automation_overview() -> None:
    automation = read_automation_status()
    cols = st.columns(3)
    metric_card(cols[0], "Last Daily Run", short_datetime(automation.get("last_daily_run")), "Prediction generation", "neutral")
    metric_card(cols[1], "Last Result Update", short_datetime(automation.get("last_result_update")), "Actual result sync", "neutral")
    metric_card(cols[2], "Automation Status", str(automation.get("automation_status") or automation.get("last_daily_status") or "ready"), "GitHub Actions / local", "accent")


def render_live_predictions_page() -> None:
    enable_auto_refresh()
    nba_results = safe_live_results("nba")
    football_results = safe_live_results("football")
    package = build_daily_prediction_package(nba_results + football_results)
    st.markdown(
        """
        <div class="section-intro">
            <h2>Live Predictions</h2>
            <p>Auto-refreshes every 5 minutes. Cards show model probability, score projection, confidence and strongest model signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_daily_spotlights(package)
    st.markdown("### Today's NBA Predictions")
    render_live_cards(nba_results, "NBA")
    st.markdown("### Today's Football Predictions")
    render_live_cards(football_results, "Football")
    if st.button("Generate Today's Content", type="primary"):
        refreshed = generate_daily_predictions()
        st.success(f"Generated content for {len(refreshed.predictions)} prediction(s). Open Content Studio to copy posts.")
    render_daily_exports(package)


def render_content_studio() -> None:
    st.markdown(
        """
        <div class="section-intro">
            <h2>Content Studio</h2>
            <p>Copy-ready social content built from today's model board. Posts avoid guarantee language and stay framed as model predictions and watchlists.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Regenerate today's content", type="primary"):
        package = generate_daily_predictions()
        st.success(f"Content regenerated for {len(package.predictions)} prediction(s).")

    short_script = read_text_file(DAILY_SHORT_SCRIPT_TXT)
    social_posts = read_text_file(DAILY_SOCIAL_POSTS_TXT)
    result_recap = read_text_file(DAILY_RESULT_RECAP_TXT)
    social = parse_social_posts(social_posts)
    generated = latest_generated_time([DAILY_SHORT_SCRIPT_TXT, DAILY_SOCIAL_POSTS_TXT, DAILY_RESULT_RECAP_TXT])

    st.markdown("### Copy-Ready Posts")
    left, right = st.columns(2)
    with left:
        render_content_card("Daily Short Script", content_title(short_script, "TikTok / Shorts script"), content_body(short_script), content_hashtags(short_script), generated, "TikTok / Shorts")
        render_content_card("Twitter/X Post", "Daily model pick post", social.get("Twitter/X Post", ""), hashtags_from_text(social.get("Twitter/X Post", "")), generated, "Caption")
    with right:
        render_content_card("YouTube Shorts Title", social.get("YouTube Shorts Title", "Daily model pick to watch"), "", "#SportsAI #ModelPick #Shorts", generated, "Title")
        render_content_card("Instagram Caption", "Daily model board caption", social.get("Instagram Caption", ""), hashtags_from_text(social.get("Instagram Caption", "")), generated, "Caption")
    st.markdown("### Yesterday Result Recap")
    render_content_card("Yesterday Result Recap", "Yesterday's hits and misses", result_recap, "#SportsAI #PredictionRecap", generated, "Recap")

    all_content = all_content_text(short_script, social_posts, result_recap)
    social_csv = social_posts_csv(social, short_script, result_recap, generated)
    export_cols = st.columns(2)
    export_cols[0].download_button("Export all content as TXT", all_content, file_name="daily_content_pack.txt", mime="text/plain")
    export_cols[1].download_button("Export social posts as CSV", social_csv, file_name="daily_social_posts.csv", mime="text/csv")


def render_install_app_page() -> None:
    st.markdown(
        """
        <div class="section-intro">
            <h2>Install App</h2>
            <p>Add AI Sports Predictor to your phone home screen for a standalone app-style experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    install_cards = [
        (
            "iPhone",
            "Safari",
            [
                "Open this site in Safari.",
                "Tap the Share button.",
                "Choose Add to Home Screen.",
                "Confirm the AI Sports Predictor icon.",
            ],
        ),
        (
            "Android",
            "Chrome",
            [
                "Open this site in Chrome.",
                "Tap the menu button.",
                "Choose Add to Home Screen or Install app.",
                "Confirm the Sports AI shortcut.",
            ],
        ),
        (
            "PWA Status",
            "Mobile Ready",
            [
                "Standalone display mode is enabled.",
                "Dark navy theme color is configured.",
                "Home screen icons are included.",
                "Offline cache support is registered when the browser allows it.",
            ],
        ),
    ]
    for col, (title, subtitle, steps) in zip(cols, install_cards):
        with col:
            st.markdown(
                f"""
                <div class="install-card">
                    <div class="install-icon">AI</div>
                    <div class="install-kicker">{html.escape(subtitle)}</div>
                    <h3>{html.escape(title)}</h3>
                    {html_list(steps)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info("Tip: if your browser does not show Install immediately, refresh once after the latest deployment finishes.")


def render_nba_page() -> None:
    st.markdown("### NBA")
    render_sport_summary("nba")
    st.markdown("#### Today's NBA Predictions")
    render_live_cards(safe_live_results("nba"), "NBA")
    st.markdown("#### Run NBA Date Prediction")
    with st.form("nba_prediction_form"):
        date_value = st.text_input("Date", "tomorrow")
        show_injuries = st.checkbox("Include injury impact", value=True)
        submitted = st.form_submit_button("Run NBA Prediction", type="primary")
    if submitted:
        results = run_prediction("nba", date_value, "", "", "", show_injuries)
        if results:
            for result in results:
                render_prediction_card(result)
        else:
            st.warning("No NBA prediction was generated for that date. Try 2026-04-12 for a seeded historical test.")


def render_football_page() -> None:
    st.markdown("### Football")
    render_sport_summary("football")
    st.markdown("#### Today's Football Predictions")
    render_live_cards(safe_live_results("football"), "Football")
    st.markdown("#### Run Football Match Prediction")
    with st.form("football_prediction_form"):
        cols = st.columns(4)
        home = cols[0].text_input("Home Team", "Mexico")
        away = cols[1].text_input("Away Team", "South Africa")
        date_value = cols[2].text_input("Date", "today")
        mode = cols[3].text_input("Mode", "WORLD_CUP")
        submitted = st.form_submit_button("Run Football Prediction", type="primary")
    if submitted:
        for result in run_prediction("football", date_value, home, away, mode, False):
            render_prediction_card(result)


def render_sport_summary(sport: str) -> None:
    frame = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv") if sport == "nba" else read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    cols = st.columns(4)
    metric_card(cols[0], "Accuracy", percent(accuracy(frame)), "Backtest sample", "positive")
    metric_card(cols[1], "Avg Confidence", percent(frame_average_confidence(frame)), "Calibration input", "neutral")
    metric_card(cols[2], "Games Tested", f"{len(frame):,}", "Historical rows", "accent")
    if sport == "football":
        metric_card(cols[3], "Draw Accuracy", percent(football_draw_accuracy(frame)), "Football only", "neutral")
    else:
        metric_card(cols[3], "Score Error", f"{safe_mean_column(frame, 'score_error'):.1f}", "Average points", "neutral")


def render_live_cards(results: list[PredictionResult], sport_label: str) -> None:
    if not results:
        st.info(f"No {sport_label} games found today. Cached reports and manual predictions are still available.")
        return
    for row in chunked(results[:8], 2):
        cols = st.columns(len(row))
        for col, result in zip(cols, row):
            with col:
                render_match_card(result)


def render_daily_spotlights(package: DailyPredictionPackage) -> None:
    st.markdown("### Daily Board")
    cards = [
        ("Highest Confidence Pick", first_result(package.highest_confidence), "confidence"),
        ("Best Value Pick", first_result(package.best_value), "value"),
        ("Upset Alert", first_result(package.upset_watch), "upset"),
        ("Draw Alert", first_result(package.draw_watch), "draw"),
        ("Injury Watch", first_result(package.injury_risk_games), "injury"),
    ]
    cols = st.columns(5)
    for col, (label, result, tone) in zip(cols, cards):
        with col:
            render_spotlight_card(label, result, tone)


def render_spotlight_card(label: str, result: PredictionResult | None, tone: str) -> None:
    if result is None:
        st.markdown(
            f"""
            <div class="spotlight-card empty">
                <div class="spotlight-label">{html.escape(label)}</div>
                <div class="spotlight-main">No signal</div>
                <div class="spotlight-sub">Waiting for today's schedule.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    probability = result.draw_probability if tone == "draw" and result.draw_probability is not None else top_probability(result)
    st.markdown(
        f"""
        <div class="spotlight-card {tone}">
            <div class="spotlight-label">{html.escape(label)}</div>
            <div class="spotlight-main">{html.escape(result.predicted_winner)}</div>
            <div class="spotlight-sub">{html.escape(result.match)}</div>
            <div class="spotlight-score">{html.escape(result.predicted_score)}</div>
            <div class="spotlight-prob">{percent(probability)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_daily_exports(package: DailyPredictionPackage) -> None:
    st.markdown("### Daily Exports")
    cols = st.columns(4)
    export_file_button(cols[0], "Download daily CSV", package.csv_path, "text/csv")
    export_file_button(cols[1], "Download daily TXT", package.txt_path, "text/plain")
    export_file_button(cols[2], "Shorts script", package.short_script_path, "text/plain")
    export_file_button(cols[3], "Social posts", package.social_posts_path, "text/plain")
    st.caption(f"Daily outputs saved with model version {package.model_version}.")


def render_content_card(platform: str, title: str, body: str, hashtags: str, generated: str, body_label: str) -> None:
    st.markdown(
        f"""
        <article class="content-card">
            <div class="content-meta"><span>{html.escape(platform)}</span><span>Generated {html.escape(generated)}</span></div>
            <h3>{html.escape(title or platform)}</h3>
            <div class="content-chip">{html.escape(body_label)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    if body:
        st.text_area(f"{platform} {body_label}", body, height=180 if len(body) > 220 else 120, label_visibility="collapsed")
    else:
        st.text_input(f"{platform} title", title, label_visibility="collapsed")
    st.caption(hashtags or "#SportsAI #ModelPick")


def export_file_button(container, label: str, path: Path, mime: str) -> None:
    with container:
        if path.exists():
            st.download_button(label, path.read_text(encoding="utf-8"), file_name=path.name, mime=mime)
        else:
            st.button(label, disabled=True)


def first_result(results: list[PredictionResult]) -> PredictionResult | None:
    return results[0] if results else None


def top_probability(result: PredictionResult) -> float:
    values = [value for value in (result.win_probability_home, result.win_probability_away, result.draw_probability) if value is not None]
    return max(values) if values else 0.0


def render_match_card(result: PredictionResult) -> None:
    home_prob = result.win_probability_home or 0.0
    away_prob = result.win_probability_away or 0.0
    home_width = max(4, min(96, int(home_prob * 100)))
    away_width = max(4, min(96, int(away_prob * 100)))
    confidence_class = result.confidence.lower() if result.confidence else "low"
    factors = split_factors(result)
    st.markdown(
        f"""
        <article class="match-card">
            <div class="match-topline">
                <span>{html.escape(result.sport.upper())}</span>
                <span class="confidence {confidence_class}">{html.escape(result.confidence)}</span>
            </div>
            <div class="teams-row">
                <div class="team-block">
                    <div class="team-logo">{team_initials(result.home_team)}</div>
                    <div><div class="team-name">{html.escape(result.home_team)}</div><div class="team-role">Home</div></div>
                </div>
                <div class="score-box">{html.escape(result.predicted_score)}</div>
                <div class="team-block right">
                    <div><div class="team-name">{html.escape(result.away_team)}</div><div class="team-role">Away</div></div>
                    <div class="team-logo">{team_initials(result.away_team)}</div>
                </div>
            </div>
            <div class="probability-grid">
                <div><div class="prob-label">Home win</div><div class="prob-value green">{percent(home_prob)}</div><div class="prob-track"><span style="width:{home_width}%"></span></div></div>
                <div><div class="prob-label">Away win</div><div class="prob-value blue">{percent(away_prob)}</div><div class="prob-track away"><span style="width:{away_width}%"></span></div></div>
                <div><div class="prob-label">Draw</div><div class="prob-value">{percent(result.draw_probability)}</div></div>
            </div>
            <div class="signal-grid">
                <div><b>Injury</b><span>{html.escape(short_signal(factors["injury"]))}</span></div>
                <div><b>Momentum</b><span>{html.escape(short_signal(factors["momentum"]))}</span></div>
                <div><b>Elo</b><span>{html.escape(short_signal(factors["elo"]))}</span></div>
                <div><b>Fatigue</b><span>{html.escape(short_signal(factors["fatigue"]))}</span></div>
            </div>
            <div class="factor-columns">
                <div><b>Key factors</b>{html_list(factor_preview(result.key_factors, 3))}</div>
                <div><b>Risk factors</b>{html_list(factor_preview(result.risk_factors, 3))}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(result: PredictionResult) -> None:
    render_match_card(result)
    csv_path, txt_path = export_prediction(result)
    export_cols = st.columns(3)
    export_cols[0].download_button("Export prediction as CSV", csv_path.read_text(encoding="utf-8"), file_name=csv_path.name, mime="text/csv")
    export_cols[1].download_button("Export prediction as TXT", txt_path.read_text(encoding="utf-8"), file_name=txt_path.name, mime="text/plain")
    export_cols[2].success(f"Saved to outputs/{csv_path.name} and outputs/{txt_path.name}")


def render_backtest_report() -> None:
    st.markdown("### Backtest Reports")
    report = BACKTEST_REPORT_TXT.read_text(encoding="utf-8") if BACKTEST_REPORT_TXT.exists() else ""
    if not report:
        st.info("No backtest report is available yet. Run the backtest commands from the terminal first.")
        return
    st.download_button("Export backtest report", report, file_name="backtest_report.txt", mime="text/plain")
    with st.expander("Historical performance report", expanded=True):
        st.text_area("Report text", report, height=320, label_visibility="collapsed")
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    col1, col2 = st.columns(2)
    with col1:
        render_accuracy_trend(nba, football)
    with col2:
        render_draw_calibration()


def render_team_analysis() -> None:
    st.markdown("### Team Analysis")
    query = st.text_input("Search team", "Mexico")
    history = all_history_frames()
    if history.empty:
        st.info("No prediction or backtest history is available yet.")
        return
    filtered = filter_team_history(history, query)
    if filtered.empty:
        st.info("No matching team records found.")
        return
    elo = read_csv(DATA_DIR / "elo_ratings.csv")
    team_elo = team_elo_value(elo, query)
    cols = st.columns(5)
    metric_card(cols[0], "Matched Records", f"{len(filtered):,}", "History rows", "accent")
    metric_card(cols[1], "Win Rate", percent(accuracy(filtered)), "Rows with result", "positive")
    metric_card(cols[2], "Avg Confidence", percent(frame_average_confidence(filtered)), "Model confidence", "neutral")
    metric_card(cols[3], "Current Elo", f"{team_elo:.0f}" if team_elo else "N/A", "Latest rating", "accent")
    metric_card(cols[4], "Injury Impact", injury_summary(query), "NBA cache", "neutral")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confidence Trend")
        render_confidence_trend(filtered)
    with right:
        st.markdown("#### Recent Form")
        render_recent_form_panel(filtered)
    st.markdown("#### Historical Predictions")
    render_history_table(filtered.tail(100), compact=False)


def render_prediction_history_page() -> None:
    st.markdown("### Prediction History")
    frame = read_prediction_history()
    if frame.empty:
        st.info("No saved prediction history yet.")
        return
    controls = st.columns([1.4, 1.0, 1.0, 1.0])
    search = controls[0].text_input("Search team or match", "")
    sport_options = ["All"] + sorted(frame["sport"].dropna().astype(str).unique().tolist()) if "sport" in frame else ["All"]
    sport = controls[1].selectbox("Sport", sport_options)
    confidence_options = ["All"] + sorted(frame["confidence"].dropna().astype(str).unique().tolist()) if "confidence" in frame else ["All"]
    confidence = controls[2].selectbox("Confidence", confidence_options)
    date_filter = controls[3].text_input("Date contains", "")
    filtered = frame.copy()
    if search:
        filtered = filtered[filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]
    if sport != "All" and "sport" in filtered:
        filtered = filtered[filtered["sport"].astype(str) == sport]
    if confidence != "All" and "confidence" in filtered:
        filtered = filtered[filtered["confidence"].astype(str) == confidence]
    if date_filter:
        filtered = filtered[filtered.astype(str).apply(lambda col: col.str.contains(date_filter, case=False, na=False)).any(axis=1)]
    st.caption(f"{len(filtered):,} matching predictions")
    render_history_table(filtered.tail(250), compact=False)


def render_results_tracker() -> None:
    st.markdown("### Results Tracker")
    render_automation_overview()
    frame = read_prediction_history()
    if frame.empty:
        st.info("No saved predictions are available yet.")
        return
    frame = normalize_history_columns(frame)
    settled = settled_predictions(frame)
    pending = frame[frame.get("actual_result", "").astype(str) == ""] if "actual_result" in frame else frame
    today = dt.date.today()
    cols = st.columns(5)
    metric_card(cols[0], "Pending Predictions", f"{len(pending):,}", "Waiting for result", "neutral")
    metric_card(cols[1], "Settled Predictions", f"{len(settled):,}", "Actual result found", "accent")
    metric_card(cols[2], "Accuracy Today", percent(period_accuracy(settled, today, today)), "Settled today", "positive")
    metric_card(cols[3], "Accuracy This Week", percent(period_accuracy(settled, today - dt.timedelta(days=7), today)), "Last 7 days", "positive")
    metric_card(cols[4], "Accuracy This Month", percent(period_accuracy(settled, today.replace(day=1), today)), "Current month", "positive")
    if st.button("Update actual results now", type="primary"):
        summary = update_results()
        st.success(f"Results updated. Settled: {summary.get('settled', 0)} · Pending: {summary.get('pending', 0)}")
    st.markdown("#### Recent Wins / Losses")
    render_recent_result_form(settled)
    st.markdown("#### Performance Report")
    report = PERFORMANCE_REPORT_TXT.read_text(encoding="utf-8") if PERFORMANCE_REPORT_TXT.exists() else "Run update-results to generate a performance report."
    st.text_area("Performance report", report, height=260, label_visibility="collapsed")
    st.markdown("#### Settled Predictions")
    render_history_table(settled.tail(100), compact=False)


def render_model_settings() -> None:
    st.markdown("### Settings")
    tuning_path = OUTPUTS_DIR / "model_weight_tuning.json"
    tuning = tuning_path.read_text(encoding="utf-8") if tuning_path.exists() else "{}"
    model_version = MODEL_VERSION_JSON.read_text(encoding="utf-8") if MODEL_VERSION_JSON.exists() else "{}"
    automation = read_automation_status()
    api_rows = [
        {"Service": "News API", "Mode": "Live" if os.getenv("NEWS_API_KEY") else "Fallback"},
        {"Service": "Football Data", "Mode": "Live" if os.getenv("FOOTBALL_DATA_KEY") else "Fallback"},
        {"Service": "Odds API", "Mode": "Live" if os.getenv("ODDS_API_KEY") else "Fallback"},
    ]
    cache_rows = [
        {"File": "NBA backtest", "Path": project_relative(NBA_DATA_DIR / "nba_backtest_results.csv"), "Available": (NBA_DATA_DIR / "nba_backtest_results.csv").exists()},
        {"File": "Football backtest", "Path": project_relative(FOOTBALL_DATA_DIR / "football_backtest_results.csv"), "Available": (FOOTBALL_DATA_DIR / "football_backtest_results.csv").exists()},
        {"File": "Elo ratings", "Path": project_relative(DATA_DIR / "elo_ratings.csv"), "Available": (DATA_DIR / "elo_ratings.csv").exists()},
        {"File": "Cache directory", "Path": project_relative(CACHE_DIR), "Available": CACHE_DIR.exists()},
        {"File": "Backtest report", "Path": project_relative(BACKTEST_REPORT_TXT), "Available": BACKTEST_REPORT_TXT.exists()},
    ]
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### API Mode")
        st.dataframe(pd.DataFrame(api_rows), use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("#### Cache Status")
        st.dataframe(pd.DataFrame(cache_rows), use_container_width=True, hide_index=True)
    st.markdown("#### Current Model Weights")
    try:
        st.code(json.dumps(json.loads(tuning), indent=2), language="json")
    except Exception:
        st.code(tuning, language="json")
    st.markdown("#### Model Version")
    try:
        st.code(json.dumps(json.loads(model_version), indent=2), language="json")
    except Exception:
        st.code(model_version, language="json")
    st.markdown("#### Automation Status")
    st.code(json.dumps(automation, indent=2, ensure_ascii=False), language="json")
    st.markdown("#### Calibration Status")
    render_draw_calibration()


def run_prediction(sport: str, date_value: str, home: str, away: str, mode: str, show_injuries: bool) -> list[PredictionResult]:
    args = Namespace(sport=sport, date=date_value, home=home, away=away, mode=mode, backtest=False, evaluate=False, injuries=show_injuries, season="2025-26", limit=100, verbose=False)
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_prediction_request sport=%s date=%s", sport, date_value)
    return predictor.predict(args)


def run_live_prediction_for_ui(sport: str) -> list[PredictionResult]:
    args = Namespace(sport=sport, date=dt.date.today().isoformat(), home="", away="", mode="WORLD_CUP", backtest=False, evaluate=False, injuries=False, season="2025-26", limit=100, verbose=False)
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_live_prediction_request sport=%s", sport)
    return predictor.predict_live(args) if sport == "football" else predictor.predict(args)


def safe_live_results(sport: str) -> list[PredictionResult]:
    try:
        return run_live_prediction_for_ui(sport)
    except Exception as exc:
        LOGGER.exception("streamlit_live_prediction_error sport=%s error=%s", sport, exc)
        return []


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
    return "\n".join([
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
    ])


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
    if px:
        fig = px.line(trend, x="date", y="rolling_accuracy", color="sport", template="plotly_dark")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
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
    data = pd.concat(rows, ignore_index=True)
    if px:
        fig = px.bar(data, x="confidence", y="games", color="sport", barmode="group", template="plotly_dark")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data, x="confidence", y="games", color="sport")


def render_draw_calibration() -> None:
    st.markdown("#### Draw Probability Calibration")
    frame = read_csv(FOOTBALL_DATA_DIR / "calibration_report.csv")
    if frame.empty:
        st.info("Football calibration data is not available.")
        return
    if px:
        fig = px.line(frame, x="bucket", y=["avg_predicted_probability", "actual_win_rate"], template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(frame, x="bucket", y=["avg_predicted_probability", "actual_win_rate"])


def render_confidence_trend(frame: pd.DataFrame) -> None:
    if frame.empty or "date" not in frame:
        st.info("No confidence trend available.")
        return
    local = frame.copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    value_col = "confidence_value" if "confidence_value" in local else "predicted_probability"
    if value_col not in local:
        st.info("No confidence values available.")
        return
    local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
    local = local.dropna(subset=["date", value_col]).sort_values("date")
    if local.empty:
        st.info("No confidence trend available.")
        return
    st.line_chart(local.tail(50), x="date", y=value_col)


def render_recent_form_panel(frame: pd.DataFrame) -> None:
    if "correct" not in frame or frame.empty:
        st.info("Recent form is unavailable.")
        return
    recent = frame.tail(10)
    form = "".join("W" if str(value).lower() in ("true", "1") else "L" for value in recent["correct"].tolist())
    st.markdown(f"<div class='form-strip'>{' '.join(form)}</div>", unsafe_allow_html=True)
    render_history_table(recent, compact=True)


def render_history_table(frame: pd.DataFrame, compact: bool) -> None:
    if frame.empty:
        st.info("No rows available.")
        return
    preferred = ["date", "prediction_date", "sport", "match", "home_team", "away_team", "predicted_winner", "predicted_result", "predicted_score", "confidence", "actual_score", "actual_winner", "actual_result", "prediction_correct", "correct", "result_updated_at"]
    cols = [col for col in preferred if col in frame.columns]
    st.dataframe((frame[cols] if cols else frame).tail(100 if compact else 250), use_container_width=True, hide_index=True)


def read_prediction_history() -> pd.DataFrame:
    frames = [read_csv(OUTPUTS_DIR / "daily_predictions.csv"), read_csv(OUTPUTS_DIR / "predictions.csv"), read_csv(DATA_DIR / "predictions_master.csv")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat([normalize_history_columns(frame) for frame in frames], ignore_index=True).drop_duplicates()


def normalize_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    if "prediction_date" in local and "date" not in local:
        local["date"] = local["prediction_date"]
    if "predicted_winner" in local and "predicted_result" not in local:
        local["predicted_result"] = local["predicted_winner"]
    if "run_timestamp" in local and "created_at" not in local:
        local["created_at"] = local["run_timestamp"]
    for column in ("actual_score", "actual_result", "prediction_correct", "result_updated_at"):
        if column not in local:
            local[column] = ""
    return local


def settled_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "actual_result" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["actual_result"].astype(str) != ""].copy()


def period_accuracy(frame: pd.DataFrame, start: dt.date, end: dt.date) -> float | None:
    if frame.empty or "date" not in frame or "prediction_correct" not in frame:
        return None
    local = frame.copy()
    local["date_obj"] = pd.to_datetime(local["date"], errors="coerce").dt.date
    local = local[(local["date_obj"] >= start) & (local["date_obj"] <= end)]
    values = local["prediction_correct"].astype(str).str.lower()
    values = values[values.isin(["true", "false"])]
    if values.empty:
        return None
    return float((values == "true").mean())


def render_recent_result_form(frame: pd.DataFrame) -> None:
    if frame.empty or "prediction_correct" not in frame:
        st.info("No settled predictions yet.")
        return
    recent = frame.tail(12)
    form = " ".join("W" if str(value).lower() == "true" else "L" for value in recent["prediction_correct"].tolist())
    st.markdown(f"<div class='form-strip'>{html.escape(form)}</div>", unsafe_allow_html=True)


def all_history_frames() -> pd.DataFrame:
    frames = [read_prediction_history(), read_csv(NBA_DATA_DIR / "nba_backtest_results.csv"), read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")]
    frames = [normalize_history_columns(frame) for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_team_history(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return frame
    return frame[frame.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]


def team_elo_value(elo: pd.DataFrame, query: str) -> float | None:
    if elo.empty or not query or "team" not in elo or "elo" not in elo:
        return None
    match = elo[elo["team"].astype(str).str.contains(query, case=False, na=False)]
    if match.empty:
        return None
    return float(pd.to_numeric(match.iloc[0]["elo"], errors="coerce"))


def injury_summary(query: str) -> str:
    if not query:
        return "N/A"
    injuries = read_csv(NBA_DATA_DIR / "injuries.csv")
    if injuries.empty:
        return "Fallback"
    matches = injuries[injuries.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    return f"{len(matches)} records" if not matches.empty else "No major record"


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


def factor_preview(items: list[str], limit: int) -> list[str]:
    return [item for item in items if item][:limit] or ["No major signal."]


def short_signal(items: list[str]) -> str:
    if not items:
        return "No major signal"
    return items[0][:92] + "..." if len(items[0]) > 92 else items[0]


def html_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def read_text_file(path: Path) -> str:
    if not path.exists():
        return "Content has not been generated yet. Use Generate Today's Content first."
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "Content could not be loaded."


def parse_social_posts(text: str) -> dict[str, str]:
    headings = ["Twitter/X Post", "YouTube Shorts Title", "Instagram Caption"]
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in headings:
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = stripped
            buffer = []
        elif current:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def content_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return fallback


def content_body(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.lower().startswith("title:") and not line.lower().startswith("hashtags:")]
    return "\n".join(lines).strip()


def content_hashtags(text: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith("hashtags:"):
            return line.split(":", 1)[1].strip()
    return hashtags_from_text(text)


def hashtags_from_text(text: str) -> str:
    tags = [part for part in text.replace("\n", " ").split(" ") if part.startswith("#")]
    return " ".join(tags) if tags else "#SportsAI #ModelPick #SportsPredictions"


def latest_generated_time(paths: list[Path]) -> str:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "Not generated"
    latest = max(path.stat().st_mtime for path in existing)
    return dt.datetime.fromtimestamp(latest).strftime("%b %d %H:%M")


def all_content_text(short_script: str, social_posts: str, result_recap: str) -> str:
    return "\n\n".join(
        [
            "DAILY SHORT SCRIPT",
            short_script.strip(),
            "SOCIAL POSTS",
            social_posts.strip(),
            "RESULT RECAP",
            result_recap.strip(),
        ]
    ).strip() + "\n"


def social_posts_csv(social: dict[str, str], short_script: str, result_recap: str, generated: str) -> str:
    rows = [
        {"platform": "TikTok / Shorts", "title": content_title(short_script, "Daily short script"), "content": content_body(short_script), "hashtags": content_hashtags(short_script), "generated_time": generated},
        {"platform": "Twitter/X", "title": "Daily model pick post", "content": social.get("Twitter/X Post", ""), "hashtags": hashtags_from_text(social.get("Twitter/X Post", "")), "generated_time": generated},
        {"platform": "YouTube Shorts", "title": social.get("YouTube Shorts Title", ""), "content": "", "hashtags": "#SportsAI #ModelPick #Shorts", "generated_time": generated},
        {"platform": "Instagram", "title": "Daily model board caption", "content": social.get("Instagram Caption", ""), "hashtags": hashtags_from_text(social.get("Instagram Caption", "")), "generated_time": generated},
        {"platform": "Result Recap", "title": "Yesterday's hits and misses", "content": result_recap, "hashtags": "#SportsAI #PredictionRecap", "generated_time": generated},
    ]
    output = StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)
    return output.getvalue()


def app_version() -> str:
    try:
        data = json.loads(MODEL_VERSION_JSON.read_text(encoding="utf-8"))
        return str(data.get("version") or "v1.0.0")
    except Exception:
        return "v1.0.0"


def render_footer() -> None:
    st.markdown(
        f"""
        <footer class="app-footer">
            <div>AI Sports Predictor</div>
            <div>Version {html.escape(app_version())} · PWA mobile build</div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def metric_card(container, label: str, value: str, caption: str, tone: str = "neutral") -> None:
    with container:
        st.markdown(f"""<div class="metric-card {tone}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div><div class="metric-caption">{html.escape(caption)}</div></div>""", unsafe_allow_html=True)


def enable_auto_refresh() -> None:
    components.html("<script>window.setTimeout(function(){window.parent.location.reload();},300000);</script>", height=0)


def chunked(items: list[PredictionResult], size: int) -> list[list[PredictionResult]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def team_initials(name: str) -> str:
    words = [word for word in str(name).replace("-", " ").split() if word]
    if not words:
        return "TM"
    return html.escape(("".join(word[0] for word in words[:3]) if len(words) > 1 else words[0][:3]).upper())


def safe_mean_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


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
    column = "confidence_value" if "confidence_value" in frame else "predicted_probability"
    if column not in frame:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(value.mean()) if not value.empty else 0.0


def percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def short_datetime(value) -> str:
    if not value:
        return "Not run"
    text = str(value)
    try:
        parsed = dt.datetime.fromisoformat(text)
        return parsed.strftime("%b %d %H:%M")
    except ValueError:
        return text[:16]


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


def apply_theme(theme_mode: str) -> None:
    light_scope = "" if theme_mode == "Dark" else ":root {--bg:#f4f7fb;--panel:#ffffff;--panel-2:#f8fafc;--border:#dbe3ef;--text:#0f172a;--muted:#64748b;}"
    auto_scope = "" if theme_mode != "Auto" else "@media (prefers-color-scheme: light) {:root {--bg:#f4f7fb;--panel:#ffffff;--panel-2:#f8fafc;--border:#dbe3ef;--text:#0f172a;--muted:#64748b;}}"
    st.markdown(
        f"""
        <style>
        :root {{--bg:#07111f;--panel:#0d1b2f;--panel-2:#10243d;--border:rgba(148,163,184,.22);--text:#f8fafc;--muted:#94a3b8;--green:#35d46f;--blue:#3b82f6;--red:#ff5f64;--gold:#fbbf24;}}
        {light_scope}{auto_scope}
        .stApp {{background:radial-gradient(circle at top left,rgba(37,99,235,.18),transparent 28rem),radial-gradient(circle at top right,rgba(34,197,94,.09),transparent 24rem),var(--bg);color:var(--text);}}
        header[data-testid="stHeader"] {{background:transparent;}}
        .block-container {{padding-top:1.25rem;padding-bottom:2rem;max-width:1440px;}}
        section[data-testid="stSidebar"] {{background:linear-gradient(180deg,#06101e 0%,#09182c 100%);border-right:1px solid rgba(148,163,184,.14);}}
        section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span {{color:#dbeafe;}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{border:1px solid transparent;border-radius:12px;margin:.12rem 0;padding:.45rem .5rem;transition:background .15s ease,border-color .15s ease;}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{background:rgba(59,130,246,.14);border-color:rgba(96,165,250,.24);}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {{font-weight:700;color:#e5eefc !important;}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{background:linear-gradient(90deg,rgba(37,99,235,.26),rgba(34,197,94,.1));border-color:rgba(96,165,250,.36);}}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{border-radius:10px;border-color:rgba(148,163,184,.22);}}
        .brand-block,.header-left,.match-topline,.teams-row,.sidebar-row {{display:flex;align-items:center;}}
        .brand-block {{gap:.8rem;padding:.85rem .4rem 1.2rem;}}
        .brand-mark,.app-logo {{width:42px;height:42px;display:grid;place-items:center;border-radius:12px;color:white;font-weight:800;background:linear-gradient(135deg,#2563eb,#22c55e);box-shadow:0 10px 30px rgba(37,99,235,.28);}}
        .brand-title {{color:white;font-weight:800;line-height:1.05;}}
        .brand-subtitle,.panel-label {{color:#8aa0bd;font-size:.78rem;margin-top:.25rem;}}
        .sidebar-panel {{margin-top:1rem;padding:.9rem;border:1px solid rgba(148,163,184,.18);background:rgba(15,23,42,.55);border-radius:14px;}}
        .sidebar-row {{justify-content:space-between;color:#cbd5e1;padding-top:.55rem;font-size:.86rem;}}
        .sidebar-row b {{color:var(--green);}}
        .top-header {{display:flex;justify-content:space-between;gap:1rem;align-items:center;margin-bottom:1.4rem;padding:1rem 1.1rem;border:1px solid var(--border);border-radius:18px;background:linear-gradient(135deg,rgba(15,31,53,.94),rgba(9,20,36,.82));box-shadow:0 18px 50px rgba(0,0,0,.22);}}
        .header-left {{gap:1rem;}}
        .top-header h1 {{color:white;font-size:1.55rem;margin:0;letter-spacing:0;}}
        .top-header p,.section-intro p {{margin:.25rem 0 0;color:#9fb0c6;}}
        .header-meta {{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap;justify-content:flex-end;}}
        .status-pill,.updated-pill {{border-radius:999px;padding:.48rem .72rem;background:rgba(15,23,42,.7);border:1px solid var(--border);color:#dbeafe;font-size:.85rem;}}
        .live-dot::before {{content:"";display:inline-block;width:8px;height:8px;margin-right:7px;background:var(--green);border-radius:50%;box-shadow:0 0 16px var(--green);}}
        h2,h3,h4 {{color:var(--text);letter-spacing:0;}}
        .section-intro {{margin:.5rem 0 1.2rem;}}
        .section-intro h2 {{margin:0;color:var(--text);}}
        .metric-card,.match-card,.mini-card,.spotlight-card,.content-card,.quick-card,.install-card {{background:linear-gradient(180deg,rgba(17,34,57,.98),rgba(11,26,45,.96));border:1px solid var(--border);border-radius:18px;box-shadow:0 16px 42px rgba(0,0,0,.22);}}
        .metric-card {{padding:1rem;min-height:116px;transition:transform .16s ease,border-color .16s ease;}}
        .metric-card:hover,.match-card:hover,.spotlight-card:hover,.content-card:hover,.quick-card:hover,.install-card:hover {{transform:translateY(-2px);border-color:rgba(59,130,246,.52);}}
        .metric-label {{color:var(--muted);font-size:.78rem;font-weight:700;text-transform:uppercase;}}
        .metric-value {{color:var(--text);font-size:1.8rem;font-weight:800;margin-top:.45rem;overflow-wrap:anywhere;}}
        .metric-card.positive .metric-value {{color:var(--green);}} .metric-card.accent .metric-value {{color:var(--blue);}}
        .metric-caption {{color:var(--muted);font-size:.84rem;margin-top:.35rem;}}
        .spotlight-card {{padding:1rem;min-height:174px;margin-bottom:1rem;position:relative;overflow:hidden;transition:transform .16s ease,border-color .16s ease;}}
        .spotlight-card::before {{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:linear-gradient(90deg,var(--blue),var(--green));}}
        .spotlight-card.value::before {{background:linear-gradient(90deg,#fbbf24,#22c55e);}}
        .spotlight-card.upset::before {{background:linear-gradient(90deg,#ff5f64,#fbbf24);}}
        .spotlight-card.draw::before {{background:linear-gradient(90deg,#a78bfa,#60a5fa);}}
        .spotlight-card.injury::before {{background:linear-gradient(90deg,#fb7185,#f97316);}}
        .spotlight-card.empty {{opacity:.72;}}
        .spotlight-label {{color:var(--muted);font-size:.72rem;text-transform:uppercase;font-weight:800;letter-spacing:.04em;}}
        .spotlight-main {{color:var(--text);font-size:1.05rem;font-weight:900;margin-top:.55rem;line-height:1.15;}}
        .spotlight-sub {{color:#9fb0c6;font-size:.78rem;margin-top:.35rem;line-height:1.25;}}
        .spotlight-score {{color:#dbeafe;font-size:.82rem;margin-top:.7rem;line-height:1.25;}}
        .spotlight-prob {{color:var(--green);font-size:1.45rem;font-weight:900;margin-top:.55rem;}}
        .content-card {{padding:1rem;margin:.85rem 0 .45rem;transition:transform .16s ease,border-color .16s ease;}}
        .content-card h3 {{margin:.5rem 0;color:var(--text);font-size:1.08rem;line-height:1.2;}}
        .content-meta {{display:flex;justify-content:space-between;gap:.7rem;color:var(--muted);font-size:.74rem;text-transform:uppercase;font-weight:800;letter-spacing:.04em;}}
        .content-chip {{display:inline-flex;margin-top:.2rem;padding:.25rem .5rem;border-radius:999px;background:rgba(59,130,246,.16);color:#bfdbfe;font-size:.76rem;font-weight:800;}}
        .quick-card {{padding:.9rem;margin:.7rem 0 .45rem;min-height:92px;transition:transform .16s ease,border-color .16s ease;}}
        .quick-title {{font-size:1rem;font-weight:900;color:var(--text);line-height:1.15;}}
        .quick-caption {{font-size:.82rem;color:var(--muted);margin-top:.45rem;line-height:1.28;}}
        .install-card {{padding:1.15rem;min-height:290px;transition:transform .16s ease,border-color .16s ease;}}
        .install-icon {{width:54px;height:54px;display:grid;place-items:center;border-radius:16px;color:white;font-weight:900;background:linear-gradient(135deg,#2563eb,#22c55e);box-shadow:0 10px 30px rgba(37,99,235,.3);margin-bottom:1rem;}}
        .install-kicker {{color:var(--green);font-size:.76rem;text-transform:uppercase;font-weight:900;letter-spacing:.06em;}}
        .install-card h3 {{margin:.45rem 0 .7rem;color:var(--text);}}
        .install-card ul {{margin:.3rem 0 0;padding-left:1.1rem;color:var(--muted);line-height:1.45;font-size:.9rem;}}
        .match-card {{padding:1rem;margin-bottom:1rem;}}
        .match-topline {{justify-content:space-between;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:1rem;}}
        .confidence {{border-radius:999px;padding:.28rem .55rem;background:rgba(59,130,246,.16);color:#93c5fd;}}
        .confidence.high {{background:rgba(34,197,94,.14);color:var(--green);}} .confidence.medium {{background:rgba(251,191,36,.14);color:var(--gold);}} .confidence.low {{background:rgba(148,163,184,.14);color:#cbd5e1;}}
        .teams-row {{justify-content:space-between;gap:.8rem;}}
        .team-block {{display:flex;gap:.7rem;align-items:center;min-width:0;}} .team-block.right {{text-align:right;}}
        .team-logo {{flex:0 0 auto;width:48px;height:48px;display:grid;place-items:center;border-radius:50%;background:radial-gradient(circle at 25% 20%,rgba(34,197,94,.45),rgba(37,99,235,.35) 48%,rgba(15,23,42,.95));border:1px solid rgba(148,163,184,.24);color:white;font-weight:900;}}
        .team-name {{color:var(--text);font-size:1rem;font-weight:800;line-height:1.15;}} .team-role {{color:var(--muted);font-size:.78rem;margin-top:.15rem;}}
        .score-box {{text-align:center;color:var(--text);min-width:130px;border-radius:14px;padding:.75rem;background:rgba(2,8,23,.35);border:1px solid rgba(148,163,184,.18);font-weight:800;}}
        .probability-grid {{display:grid;grid-template-columns:1fr 1fr .7fr;gap:.75rem;margin:1rem 0;}}
        .prob-label {{color:var(--muted);font-size:.75rem;}} .prob-value {{color:var(--text);font-weight:800;font-size:1.3rem;}} .prob-value.green {{color:var(--green);}} .prob-value.blue {{color:#60a5fa;}}
        .prob-track {{height:7px;border-radius:999px;background:rgba(148,163,184,.25);overflow:hidden;}} .prob-track span {{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#86efac);}} .prob-track.away span {{background:linear-gradient(90deg,#3b82f6,#93c5fd);}}
        .signal-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-bottom:.9rem;}}
        .signal-grid div {{padding:.65rem;border:1px solid rgba(148,163,184,.16);border-radius:12px;background:rgba(15,23,42,.38);min-height:88px;}}
        .signal-grid b,.factor-columns b {{display:block;color:var(--text);font-size:.78rem;margin-bottom:.35rem;}} .signal-grid span {{color:var(--muted);font-size:.78rem;line-height:1.25;}}
        .factor-columns {{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}} .factor-columns ul {{margin:.2rem 0 0;padding-left:1rem;color:var(--muted);font-size:.8rem;line-height:1.35;}}
        .form-strip {{font-size:1.4rem;letter-spacing:.35rem;color:var(--green);background:rgba(15,23,42,.4);border:1px solid var(--border);border-radius:16px;padding:1rem;margin-bottom:1rem;}}
        div.stButton>button,div[data-testid="stDownloadButton"]>button {{width:100%;border-radius:10px;border:1px solid rgba(59,130,246,.35);}}
        div[data-testid="stDataFrame"] {{border-radius:14px;overflow:hidden;overflow-x:auto;}} div[data-testid="stAlert"] {{border-radius:14px;}}
        .app-footer {{display:flex;justify-content:space-between;gap:1rem;margin:2rem 0 .5rem;padding:1rem 0;color:var(--muted);font-size:.82rem;border-top:1px solid var(--border);}}
        @media (max-width:768px) {{
            .block-container{{padding:.65rem .55rem 1.25rem;max-width:100%;}}
            section[data-testid="stSidebar"] div[role="radiogroup"] label{{padding:.5rem .55rem;margin:.08rem 0;}}
            .brand-block{{padding:.6rem .25rem .8rem;}}
            .top-header,.teams-row,.factor-columns,.app-footer{{flex-direction:column;display:flex;align-items:flex-start;}}
            .top-header{{padding:.9rem;border-radius:16px;margin-bottom:1rem;}}
            .header-left{{gap:.75rem;}}
            .header-meta{{justify-content:flex-start;gap:.45rem;}}
            .status-pill,.updated-pill{{font-size:.76rem;padding:.4rem .55rem;}}
            .probability-grid,.signal-grid{{grid-template-columns:1fr;gap:.5rem;margin:.7rem 0;}}
            .score-box{{width:100%;padding:.65rem;}}
            .team-logo{{width:40px;height:40px;}}
            .team-name{{font-size:.94rem;}}
            .team-block.right{{text-align:left;flex-direction:row-reverse;}}
            .metric-card,.spotlight-card,.content-card,.install-card,.quick-card{{min-height:auto;margin-bottom:.65rem;border-radius:14px;box-shadow:0 10px 26px rgba(0,0,0,.18);}}
            .match-card{{padding:.85rem;margin-bottom:.75rem;border-radius:14px;}}
            .content-card{{padding:.85rem;margin:.65rem 0 .35rem;}}
            .content-meta{{font-size:.68rem;align-items:flex-start;}}
            textarea,input{{font-size:16px !important;}}
            .top-header h1{{font-size:1.22rem;}}
            .section-intro h2{{font-size:1.55rem;}}
            .spotlight-prob{{font-size:1.25rem;}}
            .install-card{{padding:1rem;}}
            div[data-testid="stHorizontalBlock"]{{gap:.55rem;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
