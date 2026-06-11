from __future__ import annotations

import datetime as dt
from io import StringIO
import re
from typing import Any

import pandas as pd
import requests

try:
    from config import ELO_RATINGS_CSV
except ImportError:  # pragma: no cover
    from ..config import ELO_RATINGS_CSV

from .common import LOGGER, TIMEOUT_SECONDS, configure_pipeline_logging
from .db import upsert_team_elo
try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name


ELO_URL = "https://www.eloratings.net/"


def fetch_elo_ratings() -> list[dict[str, Any]]:
    configure_pipeline_logging()
    rows = fetch_elo_from_web()
    if not rows:
        rows = fetch_elo_from_local_csv()
    for row in rows:
        upsert_team_elo(row["team"], float(row["elo_rating"]), row["last_updated"], row["source"])
    LOGGER.info("Elo update complete count=%s", len(rows))
    return rows


def fetch_elo_from_web() -> list[dict[str, Any]]:
    try:
        html = requests.get(ELO_URL, timeout=TIMEOUT_SECONDS).text
    except Exception as exc:
        LOGGER.warning("Elo web fetch failed: %s", exc)
        return []
    rows = parse_elo_html(html)
    if not rows:
        rows = parse_elo_tables(html)
    if not rows:
        LOGGER.warning("Elo parser found no rows from eloratings.net")
    return rows


def parse_elo_tables(html: str) -> list[dict[str, Any]]:
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return []
    today = dt.date.today()
    rows: list[dict[str, Any]] = []
    for table in tables:
        columns = [str(column).lower() for column in table.columns]
        team_idx = next((idx for idx, column in enumerate(columns) if "team" in column or "country" in column), None)
        rating_idx = next((idx for idx, column in enumerate(columns) if "elo" in column or "rating" in column), None)
        if team_idx is None or rating_idx is None:
            continue
        for _, row in table.iterrows():
            try:
                rating = float(row.iloc[rating_idx])
            except (TypeError, ValueError):
                continue
            team = normalize_team_name(clean_team_name(str(row.iloc[team_idx])))
            if team:
                rows.append({"team": team, "elo_rating": rating, "last_updated": today, "source": "eloratings.net"})
        if rows:
            break
    return rows


def parse_elo_html(html: str) -> list[dict[str, Any]]:
    today = dt.date.today()
    patterns = [
        re.compile(r"\['(?P<rank>\d+)'\s*,\s*'(?P<team>[^']+)'\s*,\s*'(?P<rating>\d+(?:\.\d+)?)'", re.I),
        re.compile(r'"rank"\s*:\s*"?\d+"?.{0,80}?"country"\s*:\s*"(?P<team>[^"]+)".{0,80}?"rating"\s*:\s*"?(?P<rating>\d+(?:\.\d+)?)"?', re.I | re.S),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(html):
            team = normalize_team_name(clean_team_name(match.group("team")))
            if not team or team in seen:
                continue
            seen.add(team)
            rows.append({"team": team, "elo_rating": float(match.group("rating")), "last_updated": today, "source": "eloratings.net"})
        if len(rows) >= 20:
            break
    return rows


def clean_team_name(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("&amp;", "&").strip()


def fetch_elo_from_local_csv() -> list[dict[str, Any]]:
    if not ELO_RATINGS_CSV.exists():
        return []
    try:
        frame = pd.read_csv(ELO_RATINGS_CSV)
    except Exception as exc:
        LOGGER.warning("Local Elo CSV read failed: %s", exc)
        return []
    team_col = "team" if "team" in frame.columns else "Team" if "Team" in frame.columns else None
    elo_col = "elo" if "elo" in frame.columns else "elo_rating" if "elo_rating" in frame.columns else "rating" if "rating" in frame.columns else None
    if not team_col or not elo_col:
        return []
    today = dt.date.today()
    rows = []
    for _, row in frame.iterrows():
        try:
            rating = float(row[elo_col])
        except (TypeError, ValueError):
            continue
        rows.append({"team": normalize_team_name(row[team_col]), "elo_rating": rating, "last_updated": today, "source": "local_csv"})
    return rows
