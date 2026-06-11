from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
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
from .db import delete_team_elo_source_except, normalize_team_elo_source_labels, upsert_team_elo
try:
    from core.team_names import normalize_team_name
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name


ELO_BASE_URL = "https://www.eloratings.net/"
ELO_URL = ELO_BASE_URL
ELO_WORLD_TSV_URL = ELO_BASE_URL + "World.tsv"
ELO_TEAM_NAMES_TSV_URL = ELO_BASE_URL + "en.teams.tsv"
REQUEST_HEADERS = {"User-Agent": "AI-Sports-Predictor/1.0"}


def fetch_elo_ratings() -> list[dict[str, Any]]:
    configure_pipeline_logging()
    rows = fetch_elo_from_web()
    if not rows:
        rows = fetch_elo_from_local_csv()
    for row in rows:
        upsert_team_elo(row["team"], float(row["elo_rating"]), row["last_updated"], row["source"])
    if rows and any(row["source"] == "ELO" for row in rows):
        removed = delete_team_elo_source_except("ELO", [row["team"] for row in rows if row["source"] == "ELO"])
        if removed:
            LOGGER.info("Removed stale ELO rows count=%s", removed)
    normalize_team_elo_source_labels()
    LOGGER.info("Elo update complete count=%s", len(rows))
    return rows


def fetch_elo_from_web() -> list[dict[str, Any]]:
    try:
        world_response = requests.get(ELO_WORLD_TSV_URL, timeout=TIMEOUT_SECONDS, headers=REQUEST_HEADERS)
        world_response.raise_for_status()
        team_response = requests.get(ELO_TEAM_NAMES_TSV_URL, timeout=TIMEOUT_SECONDS, headers=REQUEST_HEADERS)
        team_response.raise_for_status()
    except Exception as exc:
        LOGGER.warning("Elo web fetch failed: %s", exc)
        return []

    teams = parse_elo_team_dictionary(response_text(team_response))
    rows = parse_elo_world_tsv(response_text(world_response), teams, last_modified_date(world_response))
    if not rows:
        html = requests.get(ELO_URL, timeout=TIMEOUT_SECONDS, headers=REQUEST_HEADERS).text
        rows = parse_elo_html(html)
    if not rows:
        html = requests.get(ELO_URL, timeout=TIMEOUT_SECONDS, headers=REQUEST_HEADERS).text
        rows = parse_elo_tables(html)
    if not rows:
        LOGGER.warning("Elo parser found no rows from eloratings.net")
    else:
        LOGGER.info("Elo web source loaded count=%s source=ELO", len(rows))
    return rows


def parse_elo_world_tsv(text: str, team_names: dict[str, str], last_updated: dt.date | None = None) -> list[dict[str, Any]]:
    """Parse eloratings.net World.tsv current rankings.

    World.tsv uses compact team codes. ratings.js maps field 2 to the team
    code and field 3 to the current Elo rating.
    """

    updated = last_updated or dt.date.today()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        team_code = fields[2].strip()
        team = normalize_team_name(team_names.get(team_code, team_code))
        try:
            rating = float(fields[3])
        except (TypeError, ValueError):
            continue
        if not team or team in seen or not (500 <= rating <= 2600):
            continue
        seen.add(team)
        rows.append({"team": team, "elo_rating": rating, "last_updated": updated, "source": "ELO"})
    return rows


def parse_elo_team_dictionary(text: str) -> dict[str, str]:
    teams: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        code = fields[0].strip()
        name = clean_team_name(fields[1])
        if code and name:
            teams[code] = normalize_team_name(name)
    return teams


def last_modified_date(response: requests.Response) -> dt.date:
    header = response.headers.get("Last-Modified")
    if not header:
        return dt.date.today()
    try:
        return parsedate_to_datetime(header).date()
    except (TypeError, ValueError, IndexError, OverflowError):
        return dt.date.today()


def response_text(response: requests.Response) -> str:
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


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
                rows.append({"team": team, "elo_rating": rating, "last_updated": today, "source": "ELO"})
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
            rows.append({"team": team, "elo_rating": float(match.group("rating")), "last_updated": today, "source": "ELO"})
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
        if "sport" in frame.columns and str(row.get("sport") or "").strip().lower() not in {"football", "soccer", "world_cup", "international"}:
            continue
        try:
            rating = float(row[elo_col])
        except (TypeError, ValueError):
            continue
        updated = parse_local_updated_at(row.get("updated_at") if "updated_at" in frame.columns else None) or today
        rows.append({"team": normalize_team_name(row[team_col]), "elo_rating": rating, "last_updated": updated, "source": "Estimated"})
    return rows


def parse_local_updated_at(value: object) -> dt.date | None:
    if value is None:
        return None
    try:
        return dt.datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None
