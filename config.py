from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
NBA_DATA_DIR = DATA_DIR / "nba"
FOOTBALL_DATA_DIR = DATA_DIR / "football"
LOG_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONTENT_OUTPUTS_DIR = OUTPUTS_DIR / "content"
UI_DIR = PROJECT_ROOT / "ui"
UTILS_DIR = PROJECT_ROOT / "utils"

MASTER_PREDICTIONS_CSV = DATA_DIR / "predictions_master.csv"
OUTPUT_PREDICTIONS_CSV = OUTPUTS_DIR / "predictions.csv"
DAILY_PREDICTIONS_CSV = OUTPUTS_DIR / "daily_predictions.csv"
DAILY_PREDICTIONS_TXT = OUTPUTS_DIR / "daily_predictions.txt"
DAILY_SHORT_SCRIPT_TXT = CONTENT_OUTPUTS_DIR / "daily_short_script.txt"
DAILY_SOCIAL_POSTS_TXT = CONTENT_OUTPUTS_DIR / "daily_social_posts.txt"
DAILY_RESULT_RECAP_TXT = CONTENT_OUTPUTS_DIR / "daily_result_recap.txt"
PERFORMANCE_REPORT_TXT = OUTPUTS_DIR / "performance_report.txt"
AUTOMATION_STATUS_JSON = OUTPUTS_DIR / "automation_status.json"
BACKTEST_REPORT_TXT = OUTPUTS_DIR / "backtest_report.txt"
WEIGHT_TUNING_JSON = OUTPUTS_DIR / "model_weight_tuning.json"
MODEL_VERSION_JSON = PROJECT_ROOT / "model_version.json"
ELO_RATINGS_CSV = DATA_DIR / "elo_ratings.csv"
PIPELINE_SQLITE = DATA_DIR / "worldcup_pipeline.sqlite3"
PIPELINE_LOG = LOG_DIR / "pipeline.log"
NBA_PREDICTIONS_CSV = NBA_DATA_DIR / "nba_predictions.csv"
FOOTBALL_PREDICTIONS_CSV = FOOTBALL_DATA_DIR / "football_predictions.csv"
APP_LOG = LOG_DIR / "app.log"
ENV_FILE = PROJECT_ROOT / ".env"
DATABASE_URL = os.getenv("DATABASE_URL", "")
ESPN_BASE_URL = os.getenv("ESPN_BASE_URL", "https://site.web.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY") or os.getenv("FOOTBALL_DATA_KEY", "")


def project_relative(path: Path) -> str:
    """Return a repository-relative path for UI/report display."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return path.name


def load_environment() -> None:
    """Load optional .env values without making local development depend on shell setup."""

    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
        return
    except Exception:
        pass
    if not ENV_FILE.exists():
        return
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            import os

            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception:
        return


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        NBA_DATA_DIR,
        FOOTBALL_DATA_DIR,
        LOG_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        MODELS_DIR / "nba",
        MODELS_DIR / "football",
        CACHE_DIR,
        OUTPUTS_DIR,
        CONTENT_OUTPUTS_DIR,
        UI_DIR,
        UTILS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
