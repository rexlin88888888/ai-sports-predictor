from __future__ import annotations

import datetime as dt
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import requests

try:
    from config import PIPELINE_LOG, ensure_project_dirs
except ImportError:  # pragma: no cover
    from ..config import PIPELINE_LOG, ensure_project_dirs


LOGGER = logging.getLogger("sports_predictor.pipeline")
TIMEOUT_SECONDS = 20


def configure_pipeline_logging() -> None:
    ensure_project_dirs()
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(PIPELINE_LOG, when="D", interval=1, backupCount=7, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.addHandler(logging.StreamHandler())


def request_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, retries: int = 3) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            LOGGER.warning("request failed attempt=%s url=%s error=%s", attempt, url, exc)
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def date_range(days_back: int, days_forward: int) -> list[dt.date]:
    today = dt.date.today()
    return [today + dt.timedelta(days=offset) for offset in range(-days_back, days_forward + 1)]


def yyyymmdd(value: dt.date) -> str:
    return value.strftime("%Y%m%d")
