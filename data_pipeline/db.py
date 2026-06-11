from __future__ import annotations

import datetime as dt
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from config import PIPELINE_SQLITE
except ImportError:  # pragma: no cover
    from ..config import PIPELINE_SQLITE

try:
    from core.team_names import normalize_team_name, normalized_team_key
except ImportError:  # pragma: no cover
    from ..core.team_names import normalize_team_name, normalized_team_key


POSTGRES_SCHEMAS = {
    "matches": """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            match_time_utc TIMESTAMP,
            status TEXT,
            home_score INT,
            away_score INT,
            venue TEXT,
            stage TEXT,
            group_name TEXT,
            data_source TEXT,
            data_timestamp TIMESTAMP,
            elo_diff_home_advantage INT
        )
    """,
    "team_elo": """
        CREATE TABLE IF NOT EXISTS team_elo (
            team TEXT PRIMARY KEY,
            elo_rating FLOAT,
            last_updated DATE,
            source TEXT
        )
    """,
    "team_stats": """
        CREATE TABLE IF NOT EXISTS team_stats (
            team TEXT,
            match_date DATE,
            opponent TEXT,
            goals_scored INT,
            goals_conceded INT,
            result TEXT,
            source TEXT,
            PRIMARY KEY (team, match_date, opponent)
        )
    """,
}


SQLITE_SCHEMAS = {name: ddl.replace("TIMESTAMP", "TEXT").replace("DATE", "TEXT").replace("FLOAT", "REAL") for name, ddl in POSTGRES_SCHEMAS.items()}


def using_postgres() -> bool:
    return bool(os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://")))


@contextmanager
def connect() -> Iterator[Any]:
    if using_postgres():
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DATABASE_URL is PostgreSQL but psycopg2-binary is not installed") from exc
        conn = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)
    else:
        PIPELINE_SQLITE.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(PIPELINE_SQLITE)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database() -> None:
    schemas = POSTGRES_SCHEMAS if using_postgres() else SQLITE_SCHEMAS
    with connect() as conn:
        cur = conn.cursor()
        for ddl in schemas.values():
            cur.execute(ddl)


def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    with connect() as conn:
        conn.cursor().execute(sql_for_driver(sql), params)


def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(sql_for_driver(sql), params)
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def upsert_match(row: dict[str, Any]) -> None:
    initialize_database()
    row = dict(row)
    row["home_team"] = normalize_team_name(row.get("home_team"))
    row["away_team"] = normalize_team_name(row.get("away_team"))
    existing = fetch_all("SELECT data_source FROM matches WHERE match_id=?", (str(row.get("match_id") or ""),))
    if existing and source_priority(existing[0].get("data_source")) > source_priority(row.get("data_source")):
        return
    fields = [
        "match_id", "home_team", "away_team", "match_time_utc", "status", "home_score", "away_score",
        "venue", "stage", "group_name", "data_source", "data_timestamp", "elo_diff_home_advantage",
    ]
    values = tuple(row.get(field) for field in fields)
    if using_postgres():
        placeholders = ",".join(["%s"] * len(fields))
        updates = ",".join(f"{field}=EXCLUDED.{field}" for field in fields[1:])
        sql = f"INSERT INTO matches ({','.join(fields)}) VALUES ({placeholders}) ON CONFLICT (match_id) DO UPDATE SET {updates}"
    else:
        placeholders = ",".join(["?"] * len(fields))
        updates = ",".join(f"{field}=excluded.{field}" for field in fields[1:])
        sql = f"INSERT INTO matches ({','.join(fields)}) VALUES ({placeholders}) ON CONFLICT(match_id) DO UPDATE SET {updates}"
    execute(sql, values)


def upsert_team_elo(team: str, elo_rating: float, last_updated: dt.date, source: str) -> None:
    initialize_database()
    team = normalize_team_name(team)
    if using_postgres():
        sql = "INSERT INTO team_elo (team, elo_rating, last_updated, source) VALUES (%s,%s,%s,%s) ON CONFLICT (team) DO UPDATE SET elo_rating=EXCLUDED.elo_rating,last_updated=EXCLUDED.last_updated,source=EXCLUDED.source"
    else:
        sql = "INSERT INTO team_elo (team, elo_rating, last_updated, source) VALUES (?,?,?,?) ON CONFLICT(team) DO UPDATE SET elo_rating=excluded.elo_rating,last_updated=excluded.last_updated,source=excluded.source"
    execute(sql, (team, elo_rating, last_updated.isoformat(), source))


def upsert_team_stat(team: str, match_date: dt.date, opponent: str, goals_scored: int, goals_conceded: int, result: str, source: str) -> None:
    initialize_database()
    team = normalize_team_name(team)
    opponent = normalize_team_name(opponent)
    if using_postgres():
        sql = "INSERT INTO team_stats (team, match_date, opponent, goals_scored, goals_conceded, result, source) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (team, match_date, opponent) DO UPDATE SET goals_scored=EXCLUDED.goals_scored,goals_conceded=EXCLUDED.goals_conceded,result=EXCLUDED.result,source=EXCLUDED.source"
    else:
        sql = "INSERT INTO team_stats (team, match_date, opponent, goals_scored, goals_conceded, result, source) VALUES (?,?,?,?,?,?,?) ON CONFLICT(team, match_date, opponent) DO UPDATE SET goals_scored=excluded.goals_scored,goals_conceded=excluded.goals_conceded,result=excluded.result,source=excluded.source"
    execute(sql, (team, match_date.isoformat(), opponent, goals_scored, goals_conceded, result, source))


def duplicate_match_groups() -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT match_id, home_team, away_team, match_time_utc, status, data_source FROM matches ORDER BY match_time_utc, data_source"
    )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        match_date = str(row.get("match_time_utc") or "")[:10]
        key = (match_date, normalized_team_key(row.get("home_team")), normalized_team_key(row.get("away_team")))
        if not all(key):
            continue
        groups.setdefault(key, []).append(row)
    return [
        {"date": key[0], "home_team": key[1], "away_team": key[2], "rows": value}
        for key, value in groups.items()
        if len(value) > 1
    ]


def merge_duplicate_matches() -> dict[str, Any]:
    initialize_database()
    before = duplicate_match_groups()
    removed_ids: list[str] = []
    for group in before:
        rows = group["rows"]
        keep = choose_match_row_to_keep(rows)
        for row in rows:
            if row["match_id"] == keep["match_id"]:
                continue
            removed_ids.append(str(row["match_id"]))
    if removed_ids:
        placeholders = ",".join(["?"] * len(removed_ids))
        execute(f"DELETE FROM matches WHERE match_id IN ({placeholders})", tuple(removed_ids))
    after = duplicate_match_groups()
    return {
        "duplicate_groups_before": len(before),
        "duplicate_rows_removed": len(removed_ids),
        "duplicate_groups_after": len(after),
        "removed_match_ids": removed_ids,
    }


def choose_match_row_to_keep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, int, int, str]:
        source = str(row.get("data_source") or "")
        priority = 3 if "ESPN" in source else 2 if "openfootball" in source else 1
        canonical_score = int(str(row.get("home_team") or "") == normalize_team_name(row.get("home_team")))
        canonical_score += int(str(row.get("away_team") or "") == normalize_team_name(row.get("away_team")))
        canonical_id_score = int(str(row.get("match_id") or "") == canonical_match_id_for_row(row))
        return (priority, canonical_score, canonical_id_score, str(row.get("match_id") or ""))

    return sorted(rows, key=score, reverse=True)[0]


def source_priority(source: object) -> int:
    value = str(source or "")
    if "ESPN" in value:
        return 30
    if "openfootball" in value:
        return 20
    if value:
        return 10
    return 0


def canonical_match_id_for_row(row: dict[str, Any]) -> str:
    import re

    match_date = str(row.get("match_time_utc") or "")[:10]
    home = re.sub(r"[^A-Za-z0-9]+", "_", normalize_team_name(row.get("home_team"))).strip("_")
    away = re.sub(r"[^A-Za-z0-9]+", "_", normalize_team_name(row.get("away_team"))).strip("_")
    return f"wc_{match_date}_{home}_vs_{away}"


def sql_for_driver(sql: str) -> str:
    if using_postgres():
        return sql.replace("?", "%s")
    return sql
