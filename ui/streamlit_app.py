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
from utils.team_translations import canonical_team_name, t, translate_team_name, translate_text  # noqa: E402


LOGGER = logging.getLogger("sports_predictor")
NAV_ITEMS = [
    "Home",
    "World Cup Predictions",
    "Match History",
    "Settings",
]

COUNTRY_FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Brazil": "🇧🇷", "Cameroon": "🇨🇲", "Canada": "🇨🇦",
    "Cape Verde": "🇨🇻", "Chile": "🇨🇱", "China": "🇨🇳", "Colombia": "🇨🇴",
    "Costa Rica": "🇨🇷", "Croatia": "🇭🇷", "Curacao": "🇨🇼", "Czechia": "🇨🇿",
    "Czech Republic": "🇨🇿", "Denmark": "🇩🇰", "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
    "England": "🏴", "Finland": "🇫🇮", "France": "🇫🇷", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Greece": "🇬🇷", "Guatemala": "🇬🇹", "Honduras": "🇭🇳", "Iceland": "🇮🇸", "India": "🇮🇳", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Italy": "🇮🇹", "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Korea Republic": "🇰🇷", "Kosovo": "🇽🇰", "Mexico": "🇲🇽",
    "Mongolia": "🇲🇳", "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "New Zealand": "🇳🇿",
    "Nicaragua": "🇳🇮", "Nigeria": "🇳🇬", "North Macedonia": "🇲🇰", "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Peru": "🇵🇪", "Poland": "🇵🇱",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦", "Scotland": "🏴",
    "Senegal": "🇸🇳", "Serbia": "🇷🇸", "Singapore": "🇸🇬", "South Africa": "🇿🇦",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳",
    "Thailand": "🇹🇭", "Turkey": "🇹🇷", "Turkiye": "🇹🇷", "Ukraine": "🇺🇦", "United States": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿", "Venezuela": "🇻🇪", "Wales": "🏴",
    "Zimbabwe": "🇿🇼",
}

DISPLAY_ENGLISH_NAMES = {
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Turkiye": "Turkey",
    "Czechia": "Czechia",
}

COUNTRY_ALIASES = {
    "South Korea": "Korea Republic",
    "Republic of Korea": "Korea Republic",
    "USA": "United States",
    "United States of America": "United States",
    "Czech Republic": "Czechia",
    "Turkey": "Turkiye",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}

COUNTRY_ZH_OVERRIDES = {
    "Guatemala": "危地马拉",
    "Honduras": "洪都拉斯",
    "Peru": "秘鲁",
    "North Macedonia": "北马其顿",
}

WC_TEXT_ZH = {
    "Home": "首页",
    "World Cup Predictions": "世界杯预测",
    "Match History": "比赛历史",
    "Settings": "设置",
    "World Cup Predictor": "世界杯预测系统",
    "World Cup match predictions powered by Elo ratings, form, attack and defensive metrics": "基于 Elo、球队状态、进攻防守数据的世界杯比赛预测",
    "World Cup model status": "世界杯模型状态",
    "Prediction accuracy": "预测准确率",
    "Draw accuracy": "平局准确率",
    "Saved matches": "已保存比赛",
    "World Cup Predictions": "世界杯预测",
    "Auto Schedule Mode": "自动赛程模式",
    "Manual Prediction Mode": "手动预测模式",
    "Date": "日期",
    "Home Team": "主队",
    "Away Team": "客队",
    "Run Prediction": "开始预测",
    "No World Cup or international matches found for this date. Use Manual Prediction Mode to enter a matchup.": "该日期没有找到世界杯或国际比赛。你可以使用手动预测模式输入比赛。",
    "Upcoming World Cup / International Matches": "即将进行的世界杯 / 国际比赛",
    "Manual World Cup Prediction": "手动世界杯预测",
    "Match Time": "比赛时间",
    "Predicted Score": "预测比分",
    "Win Probability": "胜率",
    "Probability": "概率",
    "Reference Odds": "参考赔率",
    "Model reference odds for analysis only": "模型参考赔率，仅供分析",
    "Confidence": "信心指数",
    "High": "高",
    "Medium": "中",
    "Low": "低",
    "Home Win": "主胜",
    "Draw": "平局",
    "Away Win": "客胜",
    "Key Factors": "关键因素",
    "Risk Factors": "风险因素",
    "Recent World Cup Predictions": "最近世界杯预测",
    "Search country or match": "搜索国家或比赛",
    "All records": "全部记录",
    "Match": "比赛",
    "Prediction": "预测结果",
    "Score": "预测比分",
    "Result": "实际结果",
    "Correct": "是否命中",
    "Created": "生成时间",
    "World Cup data": "世界杯数据",
    "World Cup backtest": "世界杯回测",
    "Match Analysis": "赛前分析",
    "Sources": "数据来源",
    "FIFA / recent international fixtures / local model": "FIFA / 近期国际比赛数据 / 本地模型",
    "Some data is estimated by the local model": "部分数据来自本地模型估算",
    "Limited recent public data is available for this team, so the model uses historical international match data for estimation.": "该球队近期公开数据较少，模型已使用历史国际比赛数据进行估算。",
    "Data source": "数据来源",
    "Live APIs enabled": "实时 API 已启用",
    "Fallback mode ready": "备用模式已就绪",
    "Last updated": "最后更新",
    "Model status": "模型状态",
}

ZH_TEXT = {
    "Dashboard": "数据看板",
    "Live Predictions": "今日预测",
    "Content Studio": "内容工作室",
    "Copy-ready social content built from today's model board. Posts avoid guarantee language and stay framed as model predictions and watchlists.": "基于今日模型看板生成可直接复制的社媒内容，避免绝对化表达，保持为模型预测和观察名单。",
    "Install App": "安装到手机",
    "Football": "足球",
    "Team Analysis": "球队分析",
    "Results Tracker": "结果追踪",
    "Prediction History": "预测历史",
    "Backtest Reports": "回测报告",
    "Settings": "设置",
    "Display Mode": "显示模式",
    "Live model dashboard": "实时模型看板",
    "Live APIs enabled": "实时 API 已启用",
    "Fallback mode ready": "备用模式已就绪",
    "Last updated": "最后更新",
    "NBA and Football forecasts with Elo, momentum, fatigue and injury signals.": "结合 Elo、状态、疲劳和伤病信号的 NBA 与足球预测。",
    "Model status": "模型状态",
    "NBA accuracy": "NBA 准确率",
    "Football accuracy": "足球准确率",
    "Draw model": "平局模型",
    "Automation": "自动化",
    "NBA Accuracy": "NBA 准确率",
    "Football Accuracy": "足球准确率",
    "Football Draw Accuracy": "足球平局准确率",
    "Average Confidence": "平均信心",
    "Total Predictions": "预测总数",
    "Last backtest sample": "最近回测样本",
    "Draw recall": "平局召回率",
    "Draw Accuracy": "平局准确率",
    "Across backtests": "回测整体",
    "Saved history": "已保存历史",
    "Quick Actions": "快捷入口",
    "Today's Picks": "今日推荐",
    "Highest Confidence": "最高信心",
    "Open today's model board": "打开今日模型看板",
    "Review top confidence cards": "查看最高信心卡片",
    "Copy social posts": "复制社媒文案",
    "Check settled picks": "查看已结算预测",
    "Last Daily Run": "上次每日运行",
    "Last Result Update": "上次结果更新",
    "Automation Status": "自动化状态",
    "Prediction generation": "预测生成",
    "Actual result sync": "真实赛果同步",
    "GitHub Actions / local": "GitHub Actions / 本地",
    "Model Overview": "模型概览",
    "Latest Predictions": "最新预测",
    "Live Predictions": "今日预测",
    "Auto-refreshes every 5 minutes. Cards show model probability, score projection, confidence and strongest model signals.": "每 5 分钟自动刷新。卡片展示模型概率、预测比分、信心指数和主要信号。",
    "Today's NBA Predictions": "今日 NBA 预测",
    "Today's Football Predictions": "今日足球预测",
    "Generate Today's Content": "生成今日内容",
    "Daily Board": "今日看板",
    "Highest Confidence Pick": "最高信心预测",
    "Best Value Pick": "最佳价值预测",
    "Upset Alert": "冷门提醒",
    "Draw Alert": "平局提醒",
    "Injury Watch": "伤病观察",
    "No signal": "暂无信号",
    "Waiting for today's schedule.": "等待今日赛程。",
    "Win Probability": "胜率",
    "Home Win Probability": "主队胜率",
    "Away Win Probability": "客队胜率",
    "Draw Probability": "平局概率",
    "Predicted Score": "预测比分",
    "Confidence": "信心指数",
    "Key Factors": "关键因素",
    "Risk Factors": "风险因素",
    "Home": "主队",
    "Away": "客队",
    "Injury": "伤病",
    "Momentum": "状态",
    "Elo": "Elo",
    "Fatigue": "疲劳",
    "Daily Exports": "每日导出",
    "Download daily CSV": "下载每日 CSV",
    "Download daily TXT": "下载每日 TXT",
    "Shorts script": "短视频脚本",
    "Social posts": "社媒文案",
    "Content Studio": "内容工作室",
    "Copy-Ready Posts": "可复制内容",
    "Daily Short Script": "每日短视频脚本",
    "Twitter/X Post": "Twitter/X 帖文",
    "YouTube Shorts Title": "YouTube Shorts 标题",
    "Instagram Caption": "Instagram 文案",
    "Yesterday Result Recap": "昨日赛果复盘",
    "Regenerate today's content": "重新生成今日内容",
    "Generated": "生成时间",
    "Export all content as TXT": "导出全部内容 TXT",
    "Export social posts as CSV": "导出社媒内容 CSV",
    "Install App": "安装到手机",
    "Add AI Sports Predictor to your phone home screen for a standalone app-style experience.": "把 AI Sports Predictor 添加到手机桌面，像独立 App 一样使用。",
    "PWA Status": "PWA 状态",
    "Mobile Ready": "手机端就绪",
    "Open this site in Safari.": "用 Safari 打开网站。",
    "Tap the Share button.": "点击分享按钮。",
    "Choose Add to Home Screen.": "选择添加到主屏幕。",
    "Confirm the AI Sports Predictor icon.": "确认 AI Sports Predictor 图标。",
    "Open this site in Chrome.": "用 Chrome 打开网站。",
    "Tap the menu button.": "点击菜单按钮。",
    "Choose Add to Home Screen or Install app.": "选择添加到主屏幕或安装应用。",
    "Confirm the Sports AI shortcut.": "确认 Sports AI 快捷方式。",
    "Standalone display mode is enabled.": "已启用独立显示模式。",
    "Dark navy theme color is configured.": "已配置深蓝主题色。",
    "Home screen icons are included.": "已包含桌面图标。",
    "Offline cache support is registered when the browser allows it.": "浏览器允许时会注册离线缓存。",
    "Tip: if your browser does not show Install immediately, refresh once after the latest deployment finishes.": "提示：如果浏览器没有立即显示安装选项，请在最新部署完成后刷新一次。",
    "Accuracy": "准确率",
    "Avg Confidence": "平均信心",
    "Games Tested": "测试比赛数",
    "Backtest sample": "回测样本",
    "Calibration input": "校准输入",
    "Historical rows": "历史记录",
    "Football only": "仅足球",
    "Score Error": "比分误差",
    "Average points": "平均分差",
    "Run NBA Date Prediction": "运行 NBA 日期预测",
    "Run Football Match Prediction": "运行足球比赛预测",
    "Date": "日期",
    "Include injury impact": "包含伤病影响",
    "Run NBA Prediction": "运行 NBA 预测",
    "Home Team": "主队",
    "Away Team": "客队",
    "Mode": "模式",
    "Run Football Prediction": "运行足球预测",
    "Confidence Trend": "信心趋势",
    "Recent Form": "近期状态",
    "Historical Predictions": "历史预测",
    "Recent Wins / Losses": "近期命中 / 失误",
    "Performance Report": "表现报告",
    "Settled Predictions": "已结算预测",
    "API Mode": "API 模式",
    "Cache Status": "缓存状态",
    "Current Model Weights": "当前模型权重",
    "Model Version": "模型版本",
    "Calibration Status": "校准状态",
    "Accuracy Trend": "准确率趋势",
    "Confidence Distribution": "信心分布",
    "Draw Probability Calibration": "平局概率校准",
}


def tr(text: str) -> str:
    if is_zh():
        if text in WC_TEXT_ZH:
            return WC_TEXT_ZH[text]
        return decode_mojibake(ZH_TEXT.get(text, t(text, current_language())))
    return text


def current_language() -> str:
    return st.session_state.get("language_choice", "English")


def is_zh() -> bool:
    return current_language() in {"中文", "ä¸­æ–‡", "Ã¤Â¸Â­Ã¦â€“â€¡"}


def decode_mojibake(text: object) -> str:
    value = "" if text is None else str(text)
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def tx(text: object) -> str:
    return translate_text(text, current_language())


def team_display(name: object) -> str:
    return translate_team_name(name, current_language())


def country_flag(name: object) -> str:
    english = country_key(name)
    return COUNTRY_FLAGS.get(english, "🌐")


def country_key(name: object) -> str:
    english = canonical_team_name(name)
    return COUNTRY_ALIASES.get(english, english)


def country_english_name(name: object) -> str:
    english = country_key(name)
    return DISPLAY_ENGLISH_NAMES.get(english, english)


def country_chinese_name(name: object) -> str:
    key = country_key(name)
    return COUNTRY_ZH_OVERRIDES.get(key, decode_mojibake(translate_team_name(key, "中文")))


def country_display_dual(name: object) -> str:
    english = country_english_name(name)
    chinese = country_chinese_name(name)
    if chinese == english:
        return f"{country_flag(name)} {english}"
    return f"{country_flag(name)} {chinese} {english}"


def confidence_display(value: object) -> str:
    return t(str(value or "").strip(), current_language())


def main() -> None:
    ensure_project_dirs()
    load_environment()
    configure_logging(False)
    st.set_page_config(page_title="AI Sports Predictor", layout="wide", page_icon="AI")

    if st.session_state.get("language_choice") == "ä¸­æ–‡":
        st.session_state["language_choice"] = "中文"
    st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0, key="language_choice")
    theme_mode = st.sidebar.selectbox(tr("Display Mode"), ["Dark", "Light", "Auto"], index=0)
    st.session_state["theme_mode"] = theme_mode
    apply_theme(theme_mode)
    inject_pwa_assets()
    st.sidebar.markdown(
        f"""
        <div class="brand-block">
            <div class="brand-mark">AI</div>
            <div>
                <div class="brand-title">{html.escape(tr("World Cup Predictor"))}</div>
                <div class="brand-subtitle">{html.escape(tr("World Cup match predictions powered by Elo ratings, form, attack and defensive metrics"))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sync_quick_navigation()
    page = st.sidebar.radio("Navigation", NAV_ITEMS, index=1, key="nav_choice", format_func=tr, label_visibility="collapsed")
    render_sidebar_status()
    render_header(page)

    if page == "Home":
        render_world_cup_home()
    elif page == "World Cup Predictions":
        render_world_cup_predictions_page()
    elif page == "Match History":
        render_world_cup_history_page()
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
                    <h1>{html.escape(tr("World Cup Predictor"))}</h1>
                    <p>{html.escape(tr("World Cup match predictions powered by Elo ratings, form, attack and defensive metrics"))}</p>
                </div>
            </div>
            <div class="header-meta">
                <div class="status-pill live-dot">{html.escape(tr(live_status))}</div>
                <div class="updated-pill">{html.escape(tr("Last updated"))} {html.escape(updated)}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status() -> None:
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    history = world_cup_history_frame()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="panel-label">{html.escape(tr("World Cup model status"))}</div>
            <div class="sidebar-row"><span>{html.escape(tr("Prediction accuracy"))}</span><b>{percent(accuracy(football))}</b></div>
            <div class="sidebar-row"><span>{html.escape(tr("Draw accuracy"))}</span><b>{percent(football_draw_accuracy(football))}</b></div>
            <div class="sidebar-row"><span>{html.escape(tr("Saved matches"))}</span><b>{len(history):,}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    master = read_prediction_history()
    cols = st.columns(5)
    metric_card(cols[0], tr("NBA Accuracy"), percent(accuracy(nba)), tr("Last backtest sample"), "positive")
    metric_card(cols[1], tr("Football Accuracy"), percent(accuracy(football)), tr("Last backtest sample"), "positive")
    metric_card(cols[2], tr("Football Draw Accuracy"), percent(football_draw_accuracy(football)), tr("Draw recall"), "neutral")
    metric_card(cols[3], tr("Average Confidence"), percent(average_confidence([nba, football])), tr("Across backtests"), "neutral")
    metric_card(cols[4], tr("Total Predictions"), f"{len(master):,}", tr("Saved history"), "accent")
    render_quick_actions()
    render_automation_overview()

    st.markdown(f"### {tr('Model Overview')}")
    left, right = st.columns(2)
    with left:
        render_accuracy_trend(nba, football)
    with right:
        render_confidence_distribution(nba, football)
    st.markdown(f"### {tr('Latest Predictions')}")
    render_history_table(master.tail(12), compact=True)


def render_quick_actions() -> None:
    st.markdown(f"### {tr('Quick Actions')}")
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
                    <div class="quick-title">{html.escape(tr(label))}</div>
                    <div class="quick-caption">{html.escape(tr(caption))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(tr(label), key=f"quick_{target}_{label}"):
                go_to_page(target)


def render_automation_overview() -> None:
    automation = read_automation_status()
    cols = st.columns(3)
    metric_card(cols[0], tr("Last Daily Run"), short_datetime(automation.get("last_daily_run")), tr("Prediction generation"), "neutral")
    metric_card(cols[1], tr("Last Result Update"), short_datetime(automation.get("last_result_update")), tr("Actual result sync"), "neutral")
    metric_card(cols[2], tr("Automation Status"), tx(str(automation.get("automation_status") or automation.get("last_daily_status") or "ready")), tr("GitHub Actions / local"), "accent")


def render_world_cup_home() -> None:
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    history = world_cup_history_frame()
    st.markdown(
        f"""
        <div class="section-intro worldcup-hero">
            <h2>{html.escape(tr("World Cup Predictor"))}</h2>
            <p>{html.escape(tr("World Cup match predictions powered by Elo ratings, form, attack and defensive metrics"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    metric_card(cols[0], tr("Prediction accuracy"), percent(accuracy(football)), tr("World Cup model status"), "positive")
    metric_card(cols[1], tr("Draw accuracy"), percent(football_draw_accuracy(football)), tr("Draw"), "neutral")
    metric_card(cols[2], tr("Saved matches"), f"{len(history):,}", tr("Match History"), "accent")
    st.markdown("### " + tr("Recent World Cup Predictions"))
    recent = history.tail(4)
    if recent.empty:
        st.info(tr("No saved prediction history yet."))
    else:
        for _, row in recent.iterrows():
            result = prediction_result_from_row(row)
            if result:
                render_world_cup_match_card(result)


def render_world_cup_predictions_page() -> None:
    enable_auto_refresh()
    st.markdown(
        f"""
        <div class="section-intro">
            <h2>{html.escape(tr("World Cup Predictions"))}</h2>
            <p>{html.escape(tr("World Cup match predictions powered by Elo ratings, form, attack and defensive metrics"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    auto_tab, manual_tab = st.tabs([tr("Auto Schedule Mode"), tr("Manual Prediction Mode")])
    with auto_tab:
        date_value = st.text_input(tr("Date"), "today", key="wc_auto_date")
        football_results = safe_live_results("football", date_value)
        st.markdown("### " + tr("Upcoming World Cup / International Matches"))
        if football_results:
            render_live_cards(football_results, "World Cup")
        else:
            st.info(tr("No World Cup or international matches found for this date. Use Manual Prediction Mode to enter a matchup."))
    with manual_tab:
        st.markdown("### " + tr("Manual World Cup Prediction"))
        with st.form("world_cup_manual_prediction_form"):
            cols = st.columns(3)
            home = cols[0].text_input(tr("Home Team"), "Argentina")
            away = cols[1].text_input(tr("Away Team"), "France")
            manual_date = cols[2].text_input(tr("Date"), "tomorrow")
            submitted = st.form_submit_button(tr("Run Prediction"), type="primary")
        if submitted:
            results = run_prediction("football", manual_date, canonical_team_name(home), canonical_team_name(away), "WORLD_CUP", False)
            if results:
                for result in results:
                    render_prediction_card(result)
            else:
                st.warning(tr("No World Cup or international matches found for this date. Use Manual Prediction Mode to enter a matchup."))


def render_world_cup_history_page() -> None:
    st.markdown("### " + tr("Match History"))
    frame = world_cup_history_frame()
    if frame.empty:
        st.info(tr("No saved prediction history yet."))
        return
    search = st.text_input(tr("Search country or match"), "")
    filtered = frame.copy()
    if search:
        search_key = canonical_team_name(search)
        localized = localize_frame_for_display(filtered)
        filtered = filtered[
            filtered.astype(str).apply(lambda col: col.str.contains(search_key, case=False, na=False)).any(axis=1)
            | localized.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        ]
    st.caption(tx(f"{len(filtered):,} {tr('All records')}"))
    render_world_cup_history_table(filtered.tail(250))


def render_world_cup_history_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info(tr("No rows available."))
        return
    rows = []
    for _, row in frame.tail(250).iterrows():
        home = str(row.get("home_team") or "")
        away = str(row.get("away_team") or "")
        match_label = f"{country_display_dual(home)} VS {country_display_dual(away)}" if home and away else tx(row.get("match"))
        rows.append(
            {
                tr("Date"): str(row.get("date") or row.get("prediction_date") or "")[:10],
                tr("Match"): match_label,
                tr("Prediction"): tx(row.get("predicted_result") or row.get("predicted_winner") or ""),
                tr("Score"): tx(row.get("predicted_score") or ""),
                tr("Result"): tx(row.get("actual_result") or row.get("actual_winner") or ""),
                tr("Correct"): tx(row.get("prediction_correct") or row.get("correct") or ""),
                tr("Created"): str(row.get("created_at") or row.get("result_updated_at") or ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def world_cup_history_frame() -> pd.DataFrame:
    frame = read_prediction_history()
    if frame.empty or "sport" not in frame:
        return pd.DataFrame()
    return frame[frame["sport"].astype(str).str.lower().eq("football")].copy()


def prediction_result_from_row(row: pd.Series) -> PredictionResult | None:
    try:
        date_value = dt.date.fromisoformat(str(row.get("date") or row.get("prediction_date") or dt.date.today().isoformat())[:10])
    except ValueError:
        date_value = dt.date.today()
    home_prob = safe_float_value(row.get("win_probability_home") or row.get("home_win_probability"))
    away_prob = safe_float_value(row.get("win_probability_away") or row.get("away_win_probability"))
    draw_prob = safe_float_value(row.get("draw_probability"))
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    if not home or not away:
        return None
    return PredictionResult(
        sport="football",
        match=str(row.get("match") or f"{home} vs {away}"),
        prediction_date=date_value,
        home_team=home,
        away_team=away,
        predicted_winner=str(row.get("predicted_winner") or row.get("predicted_result") or ""),
        win_probability_home=home_prob,
        win_probability_away=away_prob,
        draw_probability=draw_prob,
        predicted_score=str(row.get("predicted_score") or ""),
        confidence=str(row.get("confidence") or "Low"),
        key_factors=split_pipe_text(row.get("key_factors")),
        risk_factors=split_pipe_text(row.get("risk_factors")),
        data_source=str(row.get("data_source") or "unknown"),
    )


def safe_float_value(value: object) -> float | None:
    try:
        if value is None or str(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def split_pipe_text(value: object) -> list[str]:
    text = str(value or "")
    return [item.strip() for item in text.split("|") if item.strip()]


def render_live_predictions_page() -> None:
    enable_auto_refresh()
    st.markdown(
        f"""
        <div class="section-intro">
            <h2>{html.escape(tr("Live Predictions"))}</h2>
            <p>{html.escape(tr("Auto-refreshes every 5 minutes. Cards show model probability, score projection, confidence and strongest model signals."))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    auto_tab, manual_tab = st.tabs(["Auto Schedule Mode", "Manual Prediction Mode"])
    with auto_tab:
        date_value = st.text_input(tr("Date"), "today", key="live_auto_date")
        nba_results = safe_live_results("nba", date_value)
        football_results = safe_live_results("football", date_value)
        package = build_daily_prediction_package(nba_results + football_results)
        render_daily_spotlights(package)
        st.markdown("### " + tr("Today's NBA Predictions"))
        render_live_cards(nba_results, "NBA")
        st.markdown("### " + tr("Today's Football Predictions"))
        render_live_cards(football_results, "Football")
        if not nba_results and not football_results:
            st.info("No scheduled games were found for this date. Use Manual Prediction Mode to enter a matchup.")
        if st.button(tr("Generate Today's Content"), type="primary"):
            refreshed = generate_daily_predictions()
            st.success(tx(f"{len(refreshed.predictions)} prediction(s) updated. {tr('Content Studio')}"))
        render_daily_exports(package)
    with manual_tab:
        with st.form("live_manual_prediction_form"):
            sport = st.selectbox("Sport", ["nba", "football"], format_func=lambda value: t(value.upper(), current_language()))
            cols = st.columns(3)
            home = cols[0].text_input(tr("Home Team"), "Lakers" if sport == "nba" else "Mexico")
            away = cols[1].text_input(tr("Away Team"), "Warriors" if sport == "nba" else "South Africa")
            manual_date = cols[2].text_input(tr("Date"), "tomorrow")
            mode = st.text_input(tr("Mode"), "WORLD_CUP") if sport == "football" else ""
            injuries = st.checkbox(tr("Include injury impact"), value=(sport == "nba"))
            submitted = st.form_submit_button("Run Manual Prediction", type="primary")
        if submitted:
            results = run_prediction(sport, manual_date, canonical_team_name(home), canonical_team_name(away), mode, injuries)
            if results:
                for result in results:
                    render_prediction_card(result)
            else:
                st.warning("Manual prediction could not be generated. Check team names and available historical data.")


def render_content_studio() -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <h2>{html.escape(tr("Content Studio"))}</h2>
            <p>{html.escape(tr("Copy-ready social content built from today's model board. Posts avoid guarantee language and stay framed as model predictions and watchlists."))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(tr("Regenerate today's content"), type="primary"):
        package = generate_daily_predictions()
        st.success(tx(f"Content regenerated for {len(package.predictions)} prediction(s)."))

    short_script = read_text_file(DAILY_SHORT_SCRIPT_TXT)
    social_posts = read_text_file(DAILY_SOCIAL_POSTS_TXT)
    result_recap = read_text_file(DAILY_RESULT_RECAP_TXT)
    social = parse_social_posts(social_posts)
    generated = latest_generated_time([DAILY_SHORT_SCRIPT_TXT, DAILY_SOCIAL_POSTS_TXT, DAILY_RESULT_RECAP_TXT])

    st.markdown("### " + tr("Copy-Ready Posts"))
    left, right = st.columns(2)
    with left:
        render_content_card(tr("Daily Short Script"), content_title(short_script, "TikTok / Shorts script"), content_body(short_script), content_hashtags(short_script), generated, "TikTok / Shorts")
        render_content_card(tr("Twitter/X Post"), "Daily model pick post", social.get("Twitter/X Post", ""), hashtags_from_text(social.get("Twitter/X Post", "")), generated, "Caption")
    with right:
        render_content_card(tr("YouTube Shorts Title"), social.get("YouTube Shorts Title", "Daily model pick to watch"), "", "#SportsAI #ModelPick #Shorts", generated, "Title")
        render_content_card(tr("Instagram Caption"), "Daily model board caption", social.get("Instagram Caption", ""), hashtags_from_text(social.get("Instagram Caption", "")), generated, "Caption")
    st.markdown("### " + tr("Yesterday Result Recap"))
    render_content_card(tr("Yesterday Result Recap"), "Yesterday's hits and misses", result_recap, "#SportsAI #PredictionRecap", generated, "Recap")

    all_content = tx(all_content_text(short_script, social_posts, result_recap))
    social_csv = tx(social_posts_csv(social, short_script, result_recap, generated))
    export_cols = st.columns(2)
    export_cols[0].download_button(tr("Export all content as TXT"), all_content, file_name="daily_content_pack.txt", mime="text/plain")
    export_cols[1].download_button(tr("Export social posts as CSV"), social_csv, file_name="daily_social_posts.csv", mime="text/csv")


def render_install_app_page() -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <h2>{html.escape(tr("Install App"))}</h2>
            <p>{html.escape(tr("Add AI Sports Predictor to your phone home screen for a standalone app-style experience."))}</p>
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
                    <div class="install-kicker">{html.escape(tr(subtitle))}</div>
                    <h3>{html.escape(tr(title))}</h3>
                    {html_list([tr(step) for step in steps])}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.info(tr("Tip: if your browser does not show Install immediately, refresh once after the latest deployment finishes."))


def render_nba_page() -> None:
    st.markdown("### NBA")
    render_sport_summary("nba")
    st.markdown("#### " + tr("Today's NBA Predictions"))
    render_live_cards(safe_live_results("nba"), "NBA")
    st.markdown("#### " + tr("Run NBA Date Prediction"))
    with st.form("nba_prediction_form"):
        date_value = st.text_input(tr("Date"), "tomorrow")
        show_injuries = st.checkbox(tr("Include injury impact"), value=True)
        submitted = st.form_submit_button(tr("Run NBA Prediction"), type="primary")
    if submitted:
        results = run_prediction("nba", date_value, "", "", "", show_injuries)
        if results:
            for result in results:
                render_prediction_card(result)
        else:
            st.warning("No NBA prediction was generated for that date. Try 2026-04-12 for a seeded historical test.")


def render_football_page() -> None:
    st.markdown("### " + tr("Football"))
    render_sport_summary("football")
    st.markdown("#### " + tr("Today's Football Predictions"))
    render_live_cards(safe_live_results("football"), "Football")
    st.markdown("#### " + tr("Run Football Match Prediction"))
    with st.form("football_prediction_form"):
        cols = st.columns(4)
        home = cols[0].text_input(tr("Home Team"), team_display("Mexico"))
        away = cols[1].text_input(tr("Away Team"), team_display("South Africa"))
        date_value = cols[2].text_input(tr("Date"), "today")
        mode = cols[3].text_input(tr("Mode"), "WORLD_CUP")
        submitted = st.form_submit_button(tr("Run Football Prediction"), type="primary")
    if submitted:
        for result in run_prediction("football", date_value, canonical_team_name(home), canonical_team_name(away), mode, False):
            render_prediction_card(result)


def render_sport_summary(sport: str) -> None:
    frame = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv") if sport == "nba" else read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    cols = st.columns(4)
    metric_card(cols[0], tr("Accuracy"), percent(accuracy(frame)), tr("Backtest sample"), "positive")
    metric_card(cols[1], tr("Avg Confidence"), percent(frame_average_confidence(frame)), tr("Calibration input"), "neutral")
    metric_card(cols[2], tr("Games Tested"), f"{len(frame):,}", tr("Historical rows"), "accent")
    if sport == "football":
        metric_card(cols[3], tr("Draw Accuracy"), percent(football_draw_accuracy(frame)), tr("Football only"), "neutral")
    else:
        metric_card(cols[3], tr("Score Error"), f"{safe_mean_column(frame, 'score_error'):.1f}", tr("Average points"), "neutral")


def render_live_cards(results: list[PredictionResult], sport_label: str) -> None:
    if not results:
        if is_zh():
            st.info(f"今日没有{t(sport_label, current_language())}比赛。仍可查看缓存报告和手动预测。")
        else:
            st.info(f"No {sport_label} games found today. Cached reports and manual predictions are still available.")
        return
    for row in chunked(results[:8], 2):
        cols = st.columns(len(row))
        for col, result in zip(cols, row):
            with col:
                render_match_card(result)


def render_daily_spotlights(package: DailyPredictionPackage) -> None:
    st.markdown("### " + tr("Daily Board"))
    cards = [
        ("Highest Confidence Pick", first_result(package.highest_confidence), "confidence-pick"),
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
                <div class="spotlight-label">{html.escape(tr(label))}</div>
                <div class="spotlight-main">{html.escape(tr("No signal"))}</div>
                <div class="spotlight-sub">{html.escape(tr("Waiting for today's schedule."))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    probability = result.draw_probability if tone == "draw" and result.draw_probability is not None else top_probability(result)
    st.markdown(
        f"""
        <div class="spotlight-card {tone}">
            <div class="spotlight-label">{html.escape(tr(label))}</div>
            <div class="spotlight-main">{html.escape(tx(result.predicted_winner))}</div>
            <div class="spotlight-sub">{html.escape(tx(result.match))}</div>
            <div class="spotlight-score">{html.escape(tx(result.predicted_score))}</div>
            <div class="spotlight-prob">{percent(probability)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_daily_exports(package: DailyPredictionPackage) -> None:
    st.markdown("### " + tr("Daily Exports"))
    cols = st.columns(4)
    export_file_button(cols[0], tr("Download daily CSV"), package.csv_path, "text/csv")
    export_file_button(cols[1], tr("Download daily TXT"), package.txt_path, "text/plain")
    export_file_button(cols[2], tr("Shorts script"), package.short_script_path, "text/plain")
    export_file_button(cols[3], tr("Social posts"), package.social_posts_path, "text/plain")
    st.caption(tx(f"Daily outputs saved with model version {package.model_version}."))


def render_content_card(platform: str, title: str, body: str, hashtags: str, generated: str, body_label: str) -> None:
    display_title = tx(title or platform)
    display_body = tx(body)
    display_hashtags = tx(hashtags or "#SportsAI #ModelPick")
    st.markdown(
        f"""
        <article class="content-card">
            <div class="content-meta"><span>{html.escape(platform)}</span><span>{html.escape(tr("Generated"))} {html.escape(generated)}</span></div>
            <h3>{html.escape(display_title)}</h3>
            <div class="content-chip">{html.escape(body_label)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    if body:
        st.text_area(f"{platform} {body_label}", display_body, height=180 if len(display_body) > 220 else 120, label_visibility="collapsed")
    else:
        st.text_input(f"{platform} title", display_title, label_visibility="collapsed")
    st.caption(display_hashtags)


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


def render_world_cup_match_card(result: PredictionResult) -> None:
    home_prob = result.win_probability_home or 0.0
    draw_prob = result.draw_probability or 0.0
    away_prob = result.win_probability_away or 0.0
    home_score, away_score = score_numbers(result.predicted_score)
    confidence_class = result.confidence.lower() if result.confidence else "low"
    source_text = str(getattr(result, "data_source", "") or "unknown")
    home_label = country_display_dual(result.home_team)
    away_label = country_display_dual(result.away_team)
    match_time = f"{result.prediction_date.isoformat()} · {source_text}"
    st.markdown(
        f"""
        <article class="wc-card">
            <div class="wc-card-top">
                <span>{html.escape(tr("Match Time"))}: {html.escape(match_time)}</span>
                <span class="confidence-badge {confidence_class}">{html.escape(tr("Confidence"))}: {html.escape(confidence_display(result.confidence))}</span>
            </div>
            <div class="wc-score-row">
                <div class="wc-team">
                    <div class="wc-team-flag">{html.escape(country_flag(result.home_team))}</div>
                    <div class="wc-team-name">{html.escape(country_chinese_name(result.home_team))}</div>
                    <div class="wc-team-en">{html.escape(country_english_name(result.home_team))}</div>
                </div>
                <div class="wc-center-score">
                    <div class="wc-vs">VS</div>
                    <div class="wc-score">{home_score} : {away_score}</div>
                    <div class="wc-score-label">{html.escape(tr("Predicted Score"))}</div>
                </div>
                <div class="wc-team">
                    <div class="wc-team-flag">{html.escape(country_flag(result.away_team))}</div>
                    <div class="wc-team-name">{html.escape(country_chinese_name(result.away_team))}</div>
                    <div class="wc-team-en">{html.escape(country_english_name(result.away_team))}</div>
                </div>
            </div>
            <div class="wc-prob-grid">
                {world_cup_probability_cell(home_label, country_win_label(result.home_team), home_prob)}
                {world_cup_probability_cell(tr("Draw"), tr("Draw"), draw_prob)}
                {world_cup_probability_cell(away_label, country_win_label(result.away_team), away_prob)}
            </div>
            <div class="wc-odds-note">{html.escape(tr("Model reference odds for analysis only"))}</div>
            <div class="factor-columns">
                <div><b>{html.escape(tr("Key Factors"))}</b>{html_list([tx(item) for item in factor_preview(result.key_factors, 3)])}</div>
                <div><b>{html.escape(tr("Risk Factors"))}</b>{html_list([tx(item) for item in factor_preview(result.risk_factors, 3)])}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def world_cup_probability_cell(team_label: str, label: str, probability: float) -> str:
    odds = reference_odds(probability)
    width = max(3, min(100, int(probability * 100)))
    return f"""
        <div class="wc-prob-card">
            <div class="wc-prob-team">{html.escape(team_label)}</div>
            <div class="wc-prob-label">{html.escape(label)}</div>
            <div class="wc-prob-value">{probability * 100:.0f}%</div>
            <div class="wc-prob-track"><span style="width:{width}%"></span></div>
            <div class="wc-odds">{html.escape(tr("Reference Odds"))}: {odds}</div>
        </div>
    """


def country_win_label(name: object) -> str:
    if is_zh():
        return f"{country_chinese_name(name)}胜"
    return f"{country_english_name(name)} win"


def reference_odds(probability: float) -> str:
    if probability <= 0:
        return "N/A"
    return f"{1 / probability:.2f}"


def score_numbers(score_text: str) -> tuple[int, int]:
    import re

    numbers = [int(item) for item in re.findall(r"\d+", str(score_text or ""))]
    if len(numbers) >= 2:
        return numbers[-2], numbers[-1]
    return 1, 1


def extract_recent_goal_stats(result: PredictionResult, team_name: str) -> tuple[float | None, float | None]:
    import re

    key = canonical_team_name(team_name)
    for item in result.key_factors:
        match = re.search(rf"{re.escape(key)} recent goals for ([0-9.]+), against ([0-9.]+)", str(item), re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def extract_elo_gap(result: PredictionResult) -> int | None:
    import re

    text = " | ".join(result.key_factors)
    match = re.search(r"elo_diff=([+-]?\d+)", text)
    return int(match.group(1)) if match else None


def world_cup_source_label(source_text: str) -> str:
    return tr("FIFA / recent international fixtures / local model")


def world_cup_estimate_note(result: PredictionResult) -> str:
    text = " | ".join(result.risk_factors + result.key_factors).lower()
    if "missing data" in text or "only 0 historical" in text or result.data_source not in {"live_api"}:
        return tr("Some data is estimated by the local model")
    return ""


def limited_data_note(result: PredictionResult) -> str:
    text = " | ".join(result.risk_factors + result.key_factors).lower()
    if "missing data" in text or "only 0 historical" in text:
        return tr("Limited recent public data is available for this team, so the model uses historical international match data for estimation.")
    return ""


def build_world_cup_analysis(result: PredictionResult, home_score: int, away_score: int) -> str:
    home = country_english_name(result.home_team)
    away = country_english_name(result.away_team)
    home_zh = country_chinese_name(result.home_team)
    away_zh = country_chinese_name(result.away_team)
    home_for, home_against = extract_recent_goal_stats(result, result.home_team)
    away_for, away_against = extract_recent_goal_stats(result, result.away_team)
    elo_gap = extract_elo_gap(result)
    home_prob = result.win_probability_home or 0.0
    draw_prob = result.draw_probability or 0.0
    away_prob = result.win_probability_away or 0.0
    favorite = home if home_prob >= away_prob else away
    favorite_zh = home_zh if home_prob >= away_prob else away_zh
    strength_text_en = "The two sides look closely matched" if elo_gap is None or abs(elo_gap) < 80 else f"The Elo gap points to a measurable strength edge for {favorite}"
    strength_text_zh = "双方整体实力较接近" if elo_gap is None or abs(elo_gap) < 80 else f"Elo 差值显示{favorite_zh}具备一定实力优势"
    home_attack = f"{home_for:.1f}" if home_for is not None else "historical"
    home_defense = f"{home_against:.1f}" if home_against is not None else "historical"
    away_attack = f"{away_for:.1f}" if away_for is not None else "historical"
    away_defense = f"{away_against:.1f}" if away_against is not None else "historical"
    score = f"{home_score}:{away_score}"
    odds_home = reference_odds(home_prob)
    odds_draw = reference_odds(draw_prob)
    odds_away = reference_odds(away_prob)
    note = limited_data_note(result)
    if is_zh():
        analysis = (
            f"{home_zh}近期进攻输出约为场均 {home_attack} 球，防守端场均失球约 {home_defense} 个；"
            f"{away_zh}近期进攻输出约为场均 {away_attack} 球，防守端场均失球约 {away_defense} 个。"
            f"{strength_text_zh}。结合世界杯和国际比赛历史表现，模型认为本场节奏不会过于开放，"
            f"更可能出现小比分胜负或接近平局的走势，因此预测比分为 {score}。"
            f"参考赔率由胜平负概率直接换算，当前约为主胜 {odds_home}、平局 {odds_draw}、客胜 {odds_away}。"
        )
        return f"{analysis}{note}"
    analysis = (
        f"{home} is averaging around {home_attack} goals scored and {home_defense} conceded in the recent model sample, "
        f"while {away} is around {away_attack} scored and {away_defense} conceded. "
        f"{strength_text_en}. Based on recent international form, defensive balance and World Cup profile, "
        f"the model expects a controlled match rather than an unusually high-scoring game, so the most likely score is {score}. "
        f"The reference odds are converted directly from the win-draw-loss probabilities: home {odds_home}, draw {odds_draw}, away {odds_away}."
    )
    return f"{analysis} {note}".strip()


def clean_country_win_label(name: object) -> str:
    return f"{country_chinese_name(name)}胜" if is_zh() else f"{country_english_name(name)} win"


def render_world_cup_match_card(result: PredictionResult) -> None:
    home_prob = result.win_probability_home or 0.0
    draw_prob = result.draw_probability or 0.0
    away_prob = result.win_probability_away or 0.0
    home_score, away_score = score_numbers(result.predicted_score)
    home_label = country_display_dual(result.home_team)
    away_label = country_display_dual(result.away_team)
    analysis = build_world_cup_analysis(result, home_score, away_score)
    estimate_note = world_cup_estimate_note(result)
    estimate_html = f'<p class="wc-estimate-note">{html.escape(estimate_note)}</p>' if estimate_note else ""
    st.markdown(
        f"""
        <article class="wc-card">
            <div class="wc-card-top">
                <span>{html.escape(tr("Match Time"))}: {html.escape(result.prediction_date.isoformat())}</span>
            </div>
            <div class="wc-score-row">
                <div class="wc-team">
                    <div class="wc-team-flag">{html.escape(country_flag(result.home_team))}</div>
                    <div class="wc-team-name">{html.escape(country_chinese_name(result.home_team))}</div>
                    <div class="wc-team-en">{html.escape(country_english_name(result.home_team))}</div>
                </div>
                <div class="wc-center-score">
                    <div class="wc-vs">VS</div>
                    <div class="wc-score">{home_score} : {away_score}</div>
                    <div class="wc-score-label">{html.escape(tr("Predicted Score"))}</div>
                </div>
                <div class="wc-team">
                    <div class="wc-team-flag">{html.escape(country_flag(result.away_team))}</div>
                    <div class="wc-team-name">{html.escape(country_chinese_name(result.away_team))}</div>
                    <div class="wc-team-en">{html.escape(country_english_name(result.away_team))}</div>
                </div>
            </div>
            <div class="wc-prob-grid">
                {world_cup_probability_cell(home_label, clean_country_win_label(result.home_team), home_prob)}
                {world_cup_probability_cell(tr("Draw"), tr("Draw"), draw_prob)}
                {world_cup_probability_cell(away_label, clean_country_win_label(result.away_team), away_prob)}
            </div>
            <div class="wc-odds-note">{html.escape(tr("Model reference odds for analysis only"))}</div>
            <section class="wc-analysis">
                <div class="wc-analysis-title">{html.escape(tr("Match Analysis"))}</div>
                <p>{html.escape(analysis)}</p>
                {estimate_html}
            </section>
            <div class="wc-sources">{html.escape(tr("Sources"))}: {html.escape(world_cup_source_label(result.data_source))}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_world_cup_probability_native(team_label: str, label: str, probability: float) -> None:
    st.caption(team_label)
    st.markdown(f"**{label}**")
    st.metric(tr("Probability"), f"{probability * 100:.0f}%")
    st.progress(max(0.0, min(1.0, probability)))
    st.caption(f"{tr('Reference Odds')}: {reference_odds(probability)}")


def render_world_cup_match_card(result: PredictionResult) -> None:
    home_prob = result.win_probability_home or 0.0
    draw_prob = result.draw_probability or 0.0
    away_prob = result.win_probability_away or 0.0
    home_score, away_score = score_numbers(result.predicted_score)
    analysis = build_world_cup_analysis(result, home_score, away_score)
    estimate_note = world_cup_estimate_note(result)

    with st.container(border=True):
        st.caption(f"{tr('Match Time')}: {result.prediction_date.isoformat()}")
        home_col, score_col, away_col = st.columns([1.2, 0.9, 1.2], vertical_alignment="center")
        with home_col:
            st.markdown(f"## {country_flag(result.home_team)}")
            st.markdown(f"### {country_chinese_name(result.home_team)}")
            st.caption(country_english_name(result.home_team))
        with score_col:
            st.markdown("### VS")
            st.markdown(f"# {home_score} : {away_score}")
            st.caption(tr("Predicted Score"))
        with away_col:
            st.markdown(f"## {country_flag(result.away_team)}")
            st.markdown(f"### {country_chinese_name(result.away_team)}")
            st.caption(country_english_name(result.away_team))

        prob_home, prob_draw, prob_away = st.columns(3)
        with prob_home:
            render_world_cup_probability_native(country_display_dual(result.home_team), clean_country_win_label(result.home_team), home_prob)
        with prob_draw:
            render_world_cup_probability_native(tr("Draw"), tr("Draw"), draw_prob)
        with prob_away:
            render_world_cup_probability_native(country_display_dual(result.away_team), clean_country_win_label(result.away_team), away_prob)

        st.caption(tr("Model reference odds for analysis only"))
        st.markdown(f"#### {tr('Match Analysis')}")
        st.write(analysis)
        if estimate_note:
            st.info(estimate_note)
        st.caption(f"{tr('Sources')}: {world_cup_source_label(result.data_source)}")


def render_match_card(result: PredictionResult) -> None:
    if result.sport == "football":
        render_world_cup_match_card(result)
        return
    home_prob = result.win_probability_home or 0.0
    away_prob = result.win_probability_away or 0.0
    home_width = max(4, min(96, int(home_prob * 100)))
    away_width = max(4, min(96, int(away_prob * 100)))
    confidence_class = result.confidence.lower() if result.confidence else "low"
    factors = split_factors(result)
    home_name = team_display(result.home_team)
    away_name = team_display(result.away_team)
    sport_name = t(result.sport.upper(), current_language())
    score_text = tx(result.predicted_score)
    confidence_text = confidence_display(result.confidence)
    confidence_separator = "：" if is_zh() else ": "
    source_text = str(getattr(result, "data_source", "") or "unknown")
    st.markdown(
        f"""
        <article class="match-card">
            <div class="match-topline">
                <span>{html.escape(sport_name)} · {html.escape(source_text)}</span>
                <span class="confidence-badge {confidence_class}">{html.escape(tr("Confidence"))}{confidence_separator}{html.escape(confidence_text)}</span>
            </div>
            <div class="match-main">
                <div class="team-panel home">
                    <div class="team-logo">{team_initials(home_name)}</div>
                    <div class="team-copy">
                        <div class="team-role">{html.escape(tr("Home"))}</div>
                        <div class="team-name">{html.escape(home_name)}</div>
                        <div class="team-prob green">{percent(home_prob)}</div>
                    </div>
                </div>
                <div class="score-panel">
                    <div class="score-label">{html.escape(tr("Predicted Score"))}</div>
                    <div class="score-box">{html.escape(score_text)}</div>
                </div>
                <div class="team-panel away">
                    <div class="team-logo">{team_initials(away_name)}</div>
                    <div class="team-copy">
                        <div class="team-role">{html.escape(tr("Away"))}</div>
                        <div class="team-name">{html.escape(away_name)}</div>
                        <div class="team-prob blue">{percent(away_prob)}</div>
                    </div>
                </div>
            </div>
            <div class="probability-grid">
                <div><div class="prob-label">{html.escape(tr("Home Win Probability"))}</div><div class="prob-track"><span style="width:{home_width}%"></span></div></div>
                <div><div class="prob-label">{html.escape(tr("Draw Probability"))}</div><div class="draw-value">{percent(result.draw_probability)}</div></div>
                <div><div class="prob-label">{html.escape(tr("Away Win Probability"))}</div><div class="prob-track away"><span style="width:{away_width}%"></span></div></div>
            </div>
            <div class="signal-grid">
                <div><b>{html.escape(tr("Injury"))}</b><span>{html.escape(tx(short_signal(factors["injury"])))}</span></div>
                <div><b>{html.escape(tr("Momentum"))}</b><span>{html.escape(tx(short_signal(factors["momentum"])))}</span></div>
                <div><b>{html.escape(tr("Elo"))}</b><span>{html.escape(tx(short_signal(factors["elo"])))}</span></div>
                <div><b>{html.escape(tr("Fatigue"))}</b><span>{html.escape(tx(short_signal(factors["fatigue"])))}</span></div>
            </div>
            <div class="factor-columns">
                <div><b>{html.escape(tr("Key Factors"))}</b>{html_list([tx(item) for item in factor_preview(result.key_factors, 3)])}</div>
                <div><b>{html.escape(tr("Risk Factors"))}</b>{html_list([tx(item) for item in factor_preview(result.risk_factors, 3)])}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(result: PredictionResult) -> None:
    render_match_card(result)
    csv_path, txt_path = export_prediction(result)
    export_cols = st.columns(3)
    export_cols[0].download_button(tr("Export prediction as CSV"), csv_path.read_text(encoding="utf-8"), file_name=csv_path.name, mime="text/csv")
    export_cols[1].download_button(tr("Export prediction as TXT"), txt_path.read_text(encoding="utf-8"), file_name=txt_path.name, mime="text/plain")
    export_cols[2].success(tx(f"Saved to outputs/{csv_path.name} and outputs/{txt_path.name}"))


def render_backtest_report() -> None:
    st.markdown("### " + tr("Backtest Reports"))
    report = BACKTEST_REPORT_TXT.read_text(encoding="utf-8") if BACKTEST_REPORT_TXT.exists() else ""
    if not report:
        st.info(tx("No backtest report is available yet. Run the backtest commands from the terminal first."))
        return
    st.download_button(tr("Export backtest report"), tx(report), file_name="backtest_report.txt", mime="text/plain")
    with st.expander(tr("Historical performance report"), expanded=True):
        st.text_area(tr("Report text"), tx(report), height=320, label_visibility="collapsed")
    nba = read_csv(NBA_DATA_DIR / "nba_backtest_results.csv")
    football = read_csv(FOOTBALL_DATA_DIR / "football_backtest_results.csv")
    col1, col2 = st.columns(2)
    with col1:
        render_accuracy_trend(nba, football)
    with col2:
        render_draw_calibration()


def render_team_analysis() -> None:
    st.markdown("### " + tr("Team Analysis"))
    query = st.text_input(tr("Search team"), team_display("Mexico"))
    query_key = canonical_team_name(query)
    history = all_history_frames()
    if history.empty:
        st.info(tr("No prediction or backtest history is available yet."))
        return
    filtered = filter_team_history(history, query_key)
    if filtered.empty:
        st.info(tr("No matching team records found."))
        return
    elo = read_csv(DATA_DIR / "elo_ratings.csv")
    team_elo = team_elo_value(elo, query_key)
    cols = st.columns(5)
    metric_card(cols[0], tr("Matched Records"), f"{len(filtered):,}", tr("History rows"), "accent")
    metric_card(cols[1], tr("Win Rate"), percent(accuracy(filtered)), tr("Rows with result"), "positive")
    metric_card(cols[2], tr("Avg Confidence"), percent(frame_average_confidence(filtered)), tr("Model confidence"), "neutral")
    metric_card(cols[3], tr("Current Elo"), f"{team_elo:.0f}" if team_elo else "N/A", tr("Latest rating"), "accent")
    metric_card(cols[4], tr("Injury Impact"), tx(injury_summary(query_key)), tr("NBA cache"), "neutral")
    left, right = st.columns(2)
    with left:
        st.markdown("#### " + tr("Confidence Trend"))
        render_confidence_trend(filtered)
    with right:
        st.markdown("#### " + tr("Recent Form"))
        render_recent_form_panel(filtered)
    st.markdown("#### " + tr("Historical Predictions"))
    render_history_table(filtered.tail(100), compact=False)


def render_prediction_history_page() -> None:
    st.markdown("### " + tr("Prediction History"))
    frame = read_prediction_history()
    if frame.empty:
        st.info(tr("No saved prediction history yet."))
        return
    controls = st.columns([1.4, 1.0, 1.0, 1.0])
    search = controls[0].text_input(tr("Search team or match"), "")
    sport_options = ["All"] + sorted(frame["sport"].dropna().astype(str).unique().tolist()) if "sport" in frame else ["All"]
    sport = controls[1].selectbox(tr("Sport"), sport_options, format_func=lambda value: t(value, current_language()))
    confidence_options = ["All"] + sorted(frame["confidence"].dropna().astype(str).unique().tolist()) if "confidence" in frame else ["All"]
    confidence = controls[2].selectbox(tr("Confidence"), confidence_options, format_func=lambda value: t(value, current_language()))
    date_filter = controls[3].text_input(tr("Date contains"), "")
    filtered = frame.copy()
    if search:
        search_key = canonical_team_name(search)
        localized = localize_frame_for_display(filtered)
        filtered = filtered[
            filtered.astype(str).apply(lambda col: col.str.contains(search_key, case=False, na=False)).any(axis=1)
            | localized.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        ]
    if sport != "All" and "sport" in filtered:
        filtered = filtered[filtered["sport"].astype(str) == sport]
    if confidence != "All" and "confidence" in filtered:
        filtered = filtered[filtered["confidence"].astype(str) == confidence]
    if date_filter:
        filtered = filtered[filtered.astype(str).apply(lambda col: col.str.contains(date_filter, case=False, na=False)).any(axis=1)]
    st.caption(tx(f"{len(filtered):,} matching predictions"))
    render_history_table(filtered.tail(250), compact=False)


def render_results_tracker() -> None:
    st.markdown("### " + tr("Results Tracker"))
    render_automation_overview()
    frame = read_prediction_history()
    if frame.empty:
        st.info(tr("No saved predictions are available yet."))
        return
    frame = normalize_history_columns(frame)
    settled = settled_predictions(frame)
    pending = frame[frame.get("actual_result", "").astype(str) == ""] if "actual_result" in frame else frame
    today = dt.date.today()
    cols = st.columns(5)
    metric_card(cols[0], tr("Pending Predictions"), f"{len(pending):,}", tr("Waiting for result"), "neutral")
    metric_card(cols[1], tr("Settled Predictions"), f"{len(settled):,}", tr("Actual result found"), "accent")
    metric_card(cols[2], tr("Accuracy Today"), percent(period_accuracy(settled, today, today)), tr("Settled today"), "positive")
    metric_card(cols[3], tr("Accuracy This Week"), percent(period_accuracy(settled, today - dt.timedelta(days=7), today)), tr("Last 7 days"), "positive")
    metric_card(cols[4], tr("Accuracy This Month"), percent(period_accuracy(settled, today.replace(day=1), today)), tr("Current month"), "positive")
    if st.button(tr("Update actual results now"), type="primary"):
        summary = update_results()
        st.success(tx(f"Results updated. Settled: {summary.get('settled', 0)} · Pending: {summary.get('pending', 0)}"))
    st.markdown("#### " + tr("Recent Wins / Losses"))
    render_recent_result_form(settled)
    st.markdown("#### " + tr("Performance Report"))
    report = PERFORMANCE_REPORT_TXT.read_text(encoding="utf-8") if PERFORMANCE_REPORT_TXT.exists() else "Run update-results to generate a performance report."
    st.text_area(tr("Performance report"), tx(report), height=260, label_visibility="collapsed")
    st.markdown("#### " + tr("Settled Predictions"))
    render_history_table(settled.tail(100), compact=False)


def render_model_settings() -> None:
    st.markdown("### " + tr("Settings"))
    tuning_path = OUTPUTS_DIR / "model_weight_tuning.json"
    tuning = tuning_path.read_text(encoding="utf-8") if tuning_path.exists() else "{}"
    model_version = MODEL_VERSION_JSON.read_text(encoding="utf-8") if MODEL_VERSION_JSON.exists() else "{}"
    automation = read_automation_status()
    api_rows = [
        {"Service": "News API", "Mode": "Live" if os.getenv("NEWS_API_KEY") else "Fallback"},
        {"Service": "World Cup data", "Mode": "Live" if os.getenv("FOOTBALL_DATA_KEY") else "Fallback"},
    ]
    cache_rows = [
        {"File": "World Cup backtest", "Path": project_relative(FOOTBALL_DATA_DIR / "football_backtest_results.csv"), "Available": (FOOTBALL_DATA_DIR / "football_backtest_results.csv").exists()},
        {"File": "Elo ratings", "Path": project_relative(DATA_DIR / "elo_ratings.csv"), "Available": (DATA_DIR / "elo_ratings.csv").exists()},
        {"File": "Cache directory", "Path": project_relative(CACHE_DIR), "Available": CACHE_DIR.exists()},
        {"File": "Backtest report", "Path": project_relative(BACKTEST_REPORT_TXT), "Available": BACKTEST_REPORT_TXT.exists()},
    ]
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### " + tr("API Mode"))
        st.dataframe(localize_frame_for_display(pd.DataFrame(api_rows)), use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("#### " + tr("Cache Status"))
        st.dataframe(localize_frame_for_display(pd.DataFrame(cache_rows)), use_container_width=True, hide_index=True)
    st.markdown("#### " + tr("Current Model Weights"))
    try:
        st.code(tx(json.dumps(json.loads(tuning), indent=2)), language="json")
    except Exception:
        st.code(tx(tuning), language="json")
    st.markdown("#### " + tr("Model Version"))
    try:
        st.code(tx(json.dumps(json.loads(model_version), indent=2)), language="json")
    except Exception:
        st.code(tx(model_version), language="json")
    st.markdown("#### " + tr("Automation Status"))
    st.code(tx(json.dumps(automation, indent=2, ensure_ascii=False)), language="json")
    st.markdown("#### " + tr("Calibration Status"))
    render_draw_calibration()


def run_prediction(sport: str, date_value: str, home: str, away: str, mode: str, show_injuries: bool) -> list[PredictionResult]:
    args = Namespace(sport=sport, date=date_value, home=home, away=away, mode=mode, backtest=False, evaluate=False, injuries=show_injuries, season="2025-26", limit=100, verbose=False)
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_prediction_request sport=%s date=%s", sport, date_value)
    return predictor.predict(args)


def run_live_prediction_for_ui(sport: str, date_value: str = "today") -> list[PredictionResult]:
    args = Namespace(sport=sport, date=date_value, home="", away="", mode="WORLD_CUP", backtest=False, evaluate=False, injuries=False, season="2025-26", limit=100, verbose=False)
    predictor = NBAPredictor() if sport == "nba" else FootballPredictor()
    LOGGER.info("streamlit_live_prediction_request sport=%s", sport)
    return predictor.predict_live(args) if sport == "football" else predictor.predict(args)


def safe_live_results(sport: str, date_value: str = "today") -> list[PredictionResult]:
    try:
        return run_live_prediction_for_ui(sport, date_value)
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
    st.markdown("#### " + tr("Accuracy Trend"))
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
        st.info(tx("Run backtests to populate the accuracy trend."))
        return
    trend = pd.concat(rows, ignore_index=True)
    if px:
        fig = px.line(trend, x="date", y="rolling_accuracy", color="sport", template=plotly_template())
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(trend, x="date", y="rolling_accuracy", color="sport")


def render_confidence_distribution(nba: pd.DataFrame, football: pd.DataFrame) -> None:
    st.markdown("#### " + tr("Confidence Distribution"))
    rows = []
    for sport, frame in (("NBA", nba), ("Football", football)):
        if frame.empty or "confidence" not in frame:
            continue
        counts = frame["confidence"].value_counts().rename_axis("confidence").reset_index(name="games")
        counts["sport"] = sport
        rows.append(counts)
    if not rows:
        st.info(tx("No confidence data available."))
        return
    data = pd.concat(rows, ignore_index=True)
    if px:
        fig = px.bar(data, x="confidence", y="games", color="sport", barmode="group", template=plotly_template())
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data, x="confidence", y="games", color="sport")


def render_draw_calibration() -> None:
    st.markdown("#### " + tr("Draw Probability Calibration"))
    frame = read_csv(FOOTBALL_DATA_DIR / "calibration_report.csv")
    if frame.empty:
        st.info(tx("Football calibration data is not available."))
        return
    if px:
        fig = px.line(frame, x="bucket", y=["avg_predicted_probability", "actual_win_rate"], template=plotly_template())
        apply_chart_style(fig)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(frame, x="bucket", y=["avg_predicted_probability", "actual_win_rate"])


def render_confidence_trend(frame: pd.DataFrame) -> None:
    if frame.empty or "date" not in frame:
        st.info(tx("No confidence trend available."))
        return
    local = frame.copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    value_col = "confidence_value" if "confidence_value" in local else "predicted_probability"
    if value_col not in local:
        st.info(tx("No confidence values available."))
        return
    local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
    local = local.dropna(subset=["date", value_col]).sort_values("date")
    if local.empty:
        st.info(tx("No confidence trend available."))
        return
    st.line_chart(local.tail(50), x="date", y=value_col)


def render_recent_form_panel(frame: pd.DataFrame) -> None:
    if "correct" not in frame or frame.empty:
        st.info(tx("Recent form is unavailable."))
        return
    recent = frame.tail(10)
    form = "".join("W" if str(value).lower() in ("true", "1") else "L" for value in recent["correct"].tolist())
    st.markdown(f"<div class='form-strip'>{' '.join(form)}</div>", unsafe_allow_html=True)
    render_history_table(recent, compact=True)


def render_history_table(frame: pd.DataFrame, compact: bool) -> None:
    if frame.empty:
        st.info(tr("No rows available."))
        return
    preferred = ["date", "prediction_date", "sport", "match", "home_team", "away_team", "predicted_winner", "predicted_result", "predicted_score", "confidence", "actual_score", "actual_winner", "actual_result", "prediction_correct", "correct", "result_updated_at"]
    cols = [col for col in preferred if col in frame.columns]
    display = localize_frame_for_display((frame[cols] if cols else frame).tail(100 if compact else 250))
    st.dataframe(display, use_container_width=True, hide_index=True)


def localize_frame_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not is_zh():
        return frame
    display = frame.copy()
    text_columns = [
        "sport",
        "match",
        "home_team",
        "away_team",
        "predicted_winner",
        "predicted_result",
        "predicted_score",
        "confidence",
        "actual_score",
        "actual_winner",
        "actual_result",
        "key_factors",
        "risk_factors",
        "Service",
        "Mode",
        "File",
        "Available",
    ]
    for column in text_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: tr(tx(value)))
    display = display.rename(columns={column: t(column, current_language()) for column in display.columns})
    return display


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
        return tx("Content has not been generated yet. Use Generate Today's Content first.")
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return tx("Content could not be loaded.")


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
        return tx("Not generated")
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


def plotly_template() -> str:
    return "plotly_white" if st.session_state.get("theme_mode") == "Light" else "plotly_dark"


def apply_chart_style(fig) -> None:
    is_light = st.session_state.get("theme_mode") == "Light"
    grid_color = "rgba(100,116,139,.20)" if is_light else "rgba(148,163,184,.13)"
    text_color = "#0F172A" if is_light else "#F8FAFC"
    muted_color = "#64748B" if is_light else "#94A3B8"
    fig.update_layout(
        colorway=["#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#7C3AED"],
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color, size=12),
        legend=dict(font=dict(color=muted_color), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#0F1B2E" if not is_light else "#FFFFFF", bordercolor="#22324A" if not is_light else "#E2E8F0", font=dict(color=text_color)),
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color, tickfont=dict(color=muted_color))
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color, tickfont=dict(color=muted_color))


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
    light_scope = "" if theme_mode == "Dark" else ":root {--bg:#F8FAFC;--panel:#FFFFFF;--panel-2:#F8FAFC;--border:#E2E8F0;--text:#0F172A;--muted:#64748B;--primary:#2563EB;--success:#16A34A;--warning:#D97706;--danger:#DC2626;--shadow:0 14px 34px rgba(15,23,42,.09);--card-gradient:linear-gradient(180deg,#FFFFFF,#F8FAFC);--sidebar-bg:#FFFFFF;--sidebar-text:#0F172A;--sidebar-muted:#64748B;}"
    auto_scope = "" if theme_mode != "Auto" else "@media (prefers-color-scheme: light) {:root {--bg:#F8FAFC;--panel:#FFFFFF;--panel-2:#F8FAFC;--border:#E2E8F0;--text:#0F172A;--muted:#64748B;--primary:#2563EB;--success:#16A34A;--warning:#D97706;--danger:#DC2626;--shadow:0 14px 34px rgba(15,23,42,.09);--card-gradient:linear-gradient(180deg,#FFFFFF,#F8FAFC);--sidebar-bg:#FFFFFF;--sidebar-text:#0F172A;--sidebar-muted:#64748B;}}"
    st.markdown(
        f"""
        <style>
        :root {{--bg:#07111F;--panel:#0F1B2E;--panel-2:#14243A;--border:#22324A;--text:#F8FAFC;--muted:#94A3B8;--primary:#3B82F6;--success:#22C55E;--warning:#F59E0B;--danger:#EF4444;--shadow:0 18px 42px rgba(0,0,0,.28);--card-gradient:linear-gradient(180deg,#102039,#0F1B2E);--sidebar-bg:#081423;--sidebar-text:#E2E8F0;--sidebar-muted:#94A3B8;--radius:16px;}}
        {light_scope}{auto_scope}
        .stApp {{background:var(--bg);color:var(--text);font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}}
        header[data-testid="stHeader"] {{background:transparent;}}
        .block-container {{padding-top:1.1rem;padding-bottom:2.25rem;max-width:1440px;}}
        section[data-testid="stSidebar"] {{background:var(--sidebar-bg);border-right:1px solid var(--border);}}
        section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span {{color:var(--sidebar-text);}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{border:1px solid transparent;border-radius:12px;margin:.12rem 0;padding:.42rem .5rem;transition:background .16s ease,border-color .16s ease,transform .16s ease;}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{background:rgba(59,130,246,.10);border-color:color-mix(in srgb,var(--primary) 32%,transparent);transform:translateX(2px);}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {{font-weight:750;color:var(--sidebar-text) !important;}}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{background:linear-gradient(90deg,color-mix(in srgb,var(--primary) 22%,transparent),transparent);border-color:color-mix(in srgb,var(--primary) 46%,transparent);}}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{border-radius:12px;border-color:var(--border);background:var(--panel);}}
        .brand-block,.header-left,.match-topline,.sidebar-row,.match-main,.team-panel {{display:flex;align-items:center;}}
        .brand-block {{gap:.8rem;padding:.85rem .3rem 1rem;}}
        .brand-mark,.app-logo {{width:42px;height:42px;display:grid;place-items:center;border-radius:12px;color:white;font-weight:850;background:linear-gradient(135deg,var(--primary),var(--success));box-shadow:0 12px 28px color-mix(in srgb,var(--primary) 28%,transparent);}}
        .brand-title {{color:var(--sidebar-text);font-weight:850;line-height:1.05;}}
        .brand-subtitle,.panel-label {{color:var(--sidebar-muted);font-size:.76rem;margin-top:.25rem;}}
        .sidebar-panel {{margin-top:1rem;padding:.85rem;border:1px solid var(--border);background:var(--panel);border-radius:var(--radius);box-shadow:var(--shadow);}}
        .sidebar-row {{justify-content:space-between;color:var(--muted);padding-top:.52rem;font-size:.84rem;}}
        .sidebar-row b {{color:var(--success);font-weight:850;}}
        .top-header {{display:flex;justify-content:space-between;gap:1rem;align-items:center;margin-bottom:1.35rem;padding:1.05rem 1.15rem;border:1px solid var(--border);border-radius:var(--radius);background:var(--card-gradient);box-shadow:var(--shadow);}}
        .header-left {{gap:1rem;}}
        .top-header h1 {{color:var(--text);font-size:1.62rem;margin:0;letter-spacing:0;font-weight:850;}}
        .top-header p,.section-intro p {{margin:.25rem 0 0;color:var(--muted);}}
        .header-meta {{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap;justify-content:flex-end;}}
        .status-pill,.updated-pill {{border-radius:999px;padding:.46rem .72rem;background:var(--panel-2);border:1px solid var(--border);color:var(--text);font-size:.83rem;}}
        .live-dot::before {{content:"";display:inline-block;width:8px;height:8px;margin-right:7px;background:var(--success);border-radius:50%;box-shadow:0 0 16px var(--success);}}
        h2,h3,h4 {{color:var(--text);letter-spacing:0;}}
        .section-intro {{margin:.5rem 0 1.2rem;}}
        .section-intro h2 {{margin:0;color:var(--text);font-size:1.8rem;font-weight:850;}}
        .metric-card,.match-card,.mini-card,.spotlight-card,.content-card,.quick-card,.install-card {{background:var(--card-gradient);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);}}
        .metric-card {{padding:1rem;min-height:112px;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease;}}
        .metric-card:hover,.match-card:hover,.spotlight-card:hover,.content-card:hover,.quick-card:hover,.install-card:hover {{transform:translateY(-2px);border-color:color-mix(in srgb,var(--primary) 55%,var(--border));box-shadow:0 20px 48px rgba(0,0,0,.24);}}
        .metric-label {{color:var(--muted);font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;}}
        .metric-value {{color:var(--text);font-size:1.95rem;font-weight:900;margin-top:.38rem;overflow-wrap:anywhere;font-variant-numeric:tabular-nums;}}
        .metric-card.positive .metric-value {{color:var(--success);}} .metric-card.accent .metric-value {{color:var(--primary);}}
        .metric-caption {{color:var(--muted);font-size:.84rem;margin-top:.35rem;}}
        .spotlight-card {{box-sizing:border-box;width:100%;height:176px;padding:1rem;margin-bottom:1rem;position:relative;overflow:hidden;border-radius:var(--radius) !important;transition:transform .16s ease,border-color .16s ease;}}
        .spotlight-card::before {{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:linear-gradient(90deg,var(--primary),var(--success));}}
        .spotlight-card.confidence-pick::before {{background:linear-gradient(90deg,var(--primary),var(--success));}}
        .spotlight-card.value::before {{background:linear-gradient(90deg,var(--warning),var(--success));}}
        .spotlight-card.upset::before {{background:linear-gradient(90deg,var(--danger),var(--warning));}}
        .spotlight-card.draw::before {{background:linear-gradient(90deg,#7C3AED,var(--primary));}}
        .spotlight-card.injury::before {{background:linear-gradient(90deg,var(--danger),var(--warning));}}
        .spotlight-card.empty {{opacity:.72;}}
        .spotlight-label {{color:var(--muted);font-size:.7rem;text-transform:uppercase;font-weight:850;letter-spacing:.06em;}}
        .spotlight-main {{color:var(--text);font-size:1.08rem;font-weight:900;margin-top:.55rem;line-height:1.16;}}
        .spotlight-sub {{color:var(--muted);font-size:.78rem;margin-top:.35rem;line-height:1.25;}}
        .spotlight-score {{color:var(--text);font-size:.83rem;margin-top:.7rem;line-height:1.25;}}
        .spotlight-prob {{color:var(--success);font-size:1.55rem;font-weight:950;margin-top:.5rem;font-variant-numeric:tabular-nums;}}
        .content-card {{padding:1rem;margin:.85rem 0 .45rem;transition:transform .16s ease,border-color .16s ease;}}
        .content-card h3 {{margin:.5rem 0;color:var(--text);font-size:1.08rem;line-height:1.2;}}
        .content-meta {{display:flex;justify-content:space-between;gap:.7rem;color:var(--muted);font-size:.74rem;text-transform:uppercase;font-weight:800;letter-spacing:.04em;}}
        .content-chip {{display:inline-flex;margin-top:.2rem;padding:.25rem .5rem;border-radius:999px;background:color-mix(in srgb,var(--primary) 14%,transparent);color:var(--primary);font-size:.76rem;font-weight:850;}}
        .quick-card {{padding:.9rem;margin:.7rem 0 .45rem;min-height:92px;transition:transform .16s ease,border-color .16s ease;}}
        .quick-title {{font-size:1rem;font-weight:900;color:var(--text);line-height:1.15;}}
        .quick-caption {{font-size:.82rem;color:var(--muted);margin-top:.45rem;line-height:1.28;}}
        .install-card {{padding:1.15rem;min-height:290px;transition:transform .16s ease,border-color .16s ease;}}
        .install-icon {{width:54px;height:54px;display:grid;place-items:center;border-radius:16px;color:white;font-weight:900;background:linear-gradient(135deg,var(--primary),var(--success));box-shadow:0 10px 30px color-mix(in srgb,var(--primary) 30%,transparent);margin-bottom:1rem;}}
        .install-kicker {{color:var(--success);font-size:.76rem;text-transform:uppercase;font-weight:900;letter-spacing:.06em;}}
        .install-card h3 {{margin:.45rem 0 .7rem;color:var(--text);}}
        .install-card ul {{margin:.3rem 0 0;padding-left:1.1rem;color:var(--muted);line-height:1.45;font-size:.9rem;}}
        .worldcup-hero {{background:linear-gradient(135deg,color-mix(in srgb,var(--primary) 14%,transparent),color-mix(in srgb,var(--panel) 84%,transparent));border:1px solid var(--border);border-radius:18px;padding:1.35rem;box-shadow:var(--shadow);}}
        .wc-card {{border:1px solid var(--border);border-radius:18px;background:var(--card-gradient);box-shadow:var(--shadow);padding:1.1rem;margin-bottom:1rem;transition:transform .18s ease,border-color .18s ease;}}
        .wc-card:hover {{transform:translateY(-2px);border-color:color-mix(in srgb,var(--primary) 42%,var(--border));}}
        .wc-card-top {{display:flex;justify-content:space-between;gap:.75rem;align-items:center;color:var(--muted);font-size:.8rem;font-weight:800;text-transform:uppercase;letter-spacing:.04em;margin-bottom:1rem;}}
        .wc-score-row {{display:grid;grid-template-columns:1fr minmax(150px,.6fr) 1fr;gap:1rem;align-items:center;}}
        .wc-team {{border:1px solid var(--border);border-radius:16px;background:var(--panel-2);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:1rem .8rem;}}
        .wc-team-flag {{font-size:3.2rem;line-height:1;margin-bottom:.55rem;}}
        .wc-team-name {{font-size:1.35rem;font-weight:900;color:var(--text);line-height:1.15;}}
        .wc-team-en {{font-size:1rem;font-weight:750;color:var(--muted);margin-top:.2rem;}}
        .wc-center-score {{text-align:center;}}
        .wc-vs {{color:var(--muted);font-weight:900;letter-spacing:.18em;font-size:.8rem;}}
        .wc-score {{font-size:4.2rem;line-height:1;font-weight:950;color:var(--text);letter-spacing:0;margin:.2rem 0;text-shadow:0 12px 36px color-mix(in srgb,var(--primary) 30%,transparent);}}
        .wc-score-label {{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:850;}}
        .wc-prob-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-top:1rem;}}
        .wc-prob-card {{border:1px solid var(--border);background:var(--panel-2);border-radius:14px;padding:.85rem;min-width:0;}}
        .wc-prob-team {{font-size:.85rem;color:var(--muted);font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
        .wc-prob-label {{font-size:.86rem;color:var(--text);font-weight:850;margin-top:.2rem;}}
        .wc-prob-value {{font-size:1.65rem;font-weight:950;color:var(--success);margin:.25rem 0;}}
        .wc-prob-track {{height:7px;border-radius:999px;background:color-mix(in srgb,var(--muted) 16%,transparent);overflow:hidden;}}
        .wc-prob-track span {{display:block;height:100%;background:linear-gradient(90deg,var(--primary),var(--success));border-radius:999px;}}
        .wc-odds {{margin-top:.45rem;color:var(--muted);font-size:.82rem;font-weight:750;}}
        .wc-odds-note {{margin-top:.7rem;color:var(--muted);font-size:.84rem;text-align:center;}}
        .wc-analysis {{margin-top:1rem;border:1px solid var(--border);border-radius:14px;background:var(--panel-2);padding:1rem;}}
        .wc-analysis-title {{color:var(--text);font-size:.92rem;font-weight:900;margin-bottom:.45rem;}}
        .wc-analysis p {{margin:.25rem 0 0;color:var(--muted);font-size:.94rem;line-height:1.58;}}
        .wc-estimate-note {{color:var(--warning) !important;font-size:.84rem !important;}}
        .wc-sources {{margin-top:.75rem;color:var(--muted);font-size:.82rem;text-align:center;}}
        .match-card {{padding:1.05rem;margin-bottom:1rem;}}
        .match-topline {{justify-content:space-between;color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem;font-weight:850;}}
        .confidence-badge {{border-radius:999px;padding:.34rem .62rem;background:color-mix(in srgb,var(--primary) 14%,transparent);color:var(--primary);border:1px solid color-mix(in srgb,var(--primary) 28%,transparent);font-weight:850;}}
        .confidence-badge.high {{background:color-mix(in srgb,var(--success) 14%,transparent);color:var(--success);border-color:color-mix(in srgb,var(--success) 30%,transparent);}} .confidence-badge.medium {{background:color-mix(in srgb,var(--warning) 14%,transparent);color:var(--warning);border-color:color-mix(in srgb,var(--warning) 30%,transparent);}} .confidence-badge.low {{background:color-mix(in srgb,var(--muted) 12%,transparent);color:var(--muted);border-color:var(--border);}}
        .match-main {{display:grid;grid-template-columns:1fr minmax(150px,.8fr) 1fr;gap:.9rem;align-items:stretch;}}
        .team-panel {{gap:.75rem;min-width:0;padding:.85rem;border:1px solid var(--border);border-radius:14px;background:var(--panel-2);}}
        .team-panel.away {{justify-content:flex-end;text-align:right;}}
        .team-panel.away .team-logo {{order:2;}}
        .team-logo {{flex:0 0 auto;width:52px;height:52px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(135deg,color-mix(in srgb,var(--primary) 32%,transparent),color-mix(in srgb,var(--success) 24%,transparent));border:1px solid var(--border);color:var(--text);font-weight:900;}}
        .team-copy {{min-width:0;}}
        .team-name {{color:var(--text);font-size:1.02rem;font-weight:850;line-height:1.15;overflow-wrap:anywhere;}} .team-role {{color:var(--muted);font-size:.72rem;margin-bottom:.18rem;text-transform:uppercase;font-weight:850;letter-spacing:.05em;}}
        .team-prob {{font-size:1.55rem;font-weight:950;margin-top:.38rem;font-variant-numeric:tabular-nums;}}
        .team-prob.green {{color:var(--success);}} .team-prob.blue {{color:var(--primary);}}
        .score-panel {{display:grid;place-items:center;text-align:center;border:1px solid var(--border);border-radius:14px;background:color-mix(in srgb,var(--panel-2) 82%,transparent);padding:.85rem;}}
        .score-label {{color:var(--muted);font-size:.72rem;text-transform:uppercase;font-weight:850;letter-spacing:.05em;margin-bottom:.34rem;}}
        .score-box {{text-align:center;color:var(--text);font-size:1.05rem;line-height:1.25;font-weight:900;}}
        .probability-grid {{display:grid;grid-template-columns:1fr .62fr 1fr;gap:.75rem;margin:1rem 0;align-items:end;}}
        .prob-label {{color:var(--muted);font-size:.73rem;font-weight:750;}} .draw-value {{color:var(--warning);font-weight:900;font-size:1.18rem;text-align:center;font-variant-numeric:tabular-nums;}}
        .prob-track {{height:8px;border-radius:999px;background:color-mix(in srgb,var(--muted) 20%,transparent);overflow:hidden;margin-top:.45rem;}} .prob-track span {{display:block;height:100%;background:linear-gradient(90deg,var(--success),#86efac);}} .prob-track.away span {{background:linear-gradient(90deg,var(--primary),#93c5fd);}}
        .signal-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-bottom:.9rem;}}
        .signal-grid div {{padding:.66rem;border:1px solid var(--border);border-radius:12px;background:var(--panel-2);min-height:88px;}}
        .signal-grid b,.factor-columns b {{display:block;color:var(--text);font-size:.78rem;margin-bottom:.35rem;}} .signal-grid span {{color:var(--muted);font-size:.78rem;line-height:1.25;}}
        .factor-columns {{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}} .factor-columns ul {{margin:.2rem 0 0;padding-left:1rem;color:var(--muted);font-size:.8rem;line-height:1.35;}}
        .form-strip {{font-size:1.4rem;letter-spacing:.35rem;color:var(--success);background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:1rem;margin-bottom:1rem;}}
        div.stButton>button,div[data-testid="stDownloadButton"]>button {{width:100%;border-radius:10px;border:1px solid color-mix(in srgb,var(--primary) 36%,var(--border));background:var(--panel-2);color:var(--text);font-weight:800;}}
        div[data-testid="stDataFrame"] {{border-radius:14px;overflow:hidden;overflow-x:auto;}} div[data-testid="stAlert"] {{border-radius:14px;}}
        .app-footer {{display:flex;justify-content:space-between;gap:1rem;margin:2rem 0 .5rem;padding:1rem 0;color:var(--muted);font-size:.82rem;border-top:1px solid var(--border);}}
        @media (max-width:768px) {{
            .block-container{{padding:.65rem .55rem 1.25rem;max-width:100%;}}
            section[data-testid="stSidebar"] div[role="radiogroup"] label{{padding:.5rem .55rem;margin:.08rem 0;}}
            .brand-block{{padding:.6rem .25rem .8rem;}}
            .top-header,.factor-columns,.app-footer{{flex-direction:column;display:flex;align-items:flex-start;}}
            .top-header{{padding:.9rem;border-radius:16px;margin-bottom:1rem;}}
            .header-left{{gap:.75rem;}}
            .header-meta{{justify-content:flex-start;gap:.45rem;}}
            .status-pill,.updated-pill{{font-size:.76rem;padding:.4rem .55rem;}}
            .match-main,.probability-grid,.signal-grid,.wc-score-row,.wc-prob-grid{{grid-template-columns:1fr;gap:.55rem;margin:.7rem 0;}}
            .wc-card{{padding:.85rem;margin-bottom:.75rem;border-radius:14px;}}
            .wc-card-top{{align-items:flex-start;flex-direction:column;}}
            .wc-team{{min-height:118px;padding:.85rem;}}
            .wc-team-flag{{font-size:2.7rem;}}
            .wc-team-name{{font-size:1.18rem;}}
            .wc-score{{font-size:3.3rem;}}
            .score-panel{{width:100%;padding:.75rem;}}
            .team-logo{{width:42px;height:42px;border-radius:12px;}}
            .team-name{{font-size:.94rem;}}
            .team-panel.away{{justify-content:flex-start;text-align:left;}}
            .team-panel.away .team-logo{{order:0;}}
            .team-prob{{font-size:1.35rem;}}
            .metric-card,.content-card,.install-card,.quick-card{{min-height:auto;margin-bottom:.65rem;border-radius:14px;box-shadow:0 10px 26px rgba(0,0,0,.18);}}
            .spotlight-card{{height:156px;margin-bottom:.65rem;border-radius:14px !important;box-shadow:0 10px 26px rgba(0,0,0,.18);}}
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
