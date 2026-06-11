from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

try:
    from config import OUTPUTS_DIR
except ImportError:  # pragma: no cover
    from ..config import OUTPUTS_DIR


OPENAPI_MATCH_EXPLAIN: dict[str, Any] = {
    "paths": {
        "/match/{id}/explain": {
            "get": {
                "summary": "Return top model factors and data provenance for one match",
                "operationId": "explainMatch",
                "tags": ["match"],
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Prediction match id, usually home_vs_away or the saved match field with spaces replaced by underscores.",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Top explanation factors and data sources",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["top_factors", "data_source"],
                                    "properties": {
                                        "top_factors": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "required": ["name", "value"],
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "value": {"type": "number"},
                                                },
                                            },
                                        },
                                        "data_source": {
                                            "type": "object",
                                            "properties": {
                                                "elo": {"type": "string", "example": "ELO"},
                                                "form": {"type": "string", "example": "ESPN"},
                                            },
                                        },
                                    },
                                    "example": {
                                        "top_factors": [
                                            {"name": "elo_diff", "value": 62},
                                            {"name": "recent_attack", "value": 23},
                                            {"name": "defence_form", "value": 18},
                                        ],
                                        "data_source": {"elo": "ELO", "form": "ESPN"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    }
}


def explain_match_json(match_id: str, predictions_path: Path | None = None) -> dict[str, object]:
    """Swagger/OpenAPI: GET /match/{id}/explain.

    Returns:
    {
      "top_factors": [{"name": "elo_diff", "value": 62}],
      "data_source": {"elo": "ELO", "form": "ESPN"}
    }
    """

    row = find_prediction_row(match_id, predictions_path or OUTPUTS_DIR / "predictions.csv")
    if not row:
        return {"top_factors": [], "data_source": {"elo": "Estimated", "form": "Estimated"}}
    key_factors = row.get("key_factors", "")
    return {
        "top_factors": parse_top_factors(key_factors),
        "data_source": parse_data_sources(key_factors, row.get("data_source", "")),
    }


def find_prediction_row(match_id: str, predictions_path: Path) -> dict[str, str] | None:
    if not predictions_path.exists():
        return None
    requested = normalize_match_id(match_id)
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = {
                row.get("match", ""),
                f"{row.get('home_team', '')}_vs_{row.get('away_team', '')}",
                f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
            }
            if requested in {normalize_match_id(candidate) for candidate in candidates}:
                return row
    return None


def normalize_match_id(value: object) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("_vs_", " vs ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_top_factors(key_factors: str) -> list[dict[str, float | str]]:
    explicit = parse_explicit_top_factors(key_factors)
    if explicit:
        return explicit
    inferred = infer_top_factors(key_factors)
    return inferred[:3]


def parse_explicit_top_factors(key_factors: str) -> list[dict[str, float | str]]:
    marker = "top_factors="
    if marker not in key_factors:
        return []
    segment = key_factors.split(marker, 1)[1].split("|", 1)[0]
    factors: list[dict[str, float | str]] = []
    for item in segment.split(","):
        if ":" not in item:
            continue
        name, value = item.split(":", 1)
        try:
            factors.append({"name": name.strip(), "value": round(float(value), 2)})
        except ValueError:
            continue
    return sorted(factors, key=lambda factor: abs(float(factor["value"])), reverse=True)[:3]


def infer_top_factors(key_factors: str) -> list[dict[str, float | str]]:
    factors: list[dict[str, float | str]] = []
    elo_match = re.search(r"elo_diff=([+-]?\d+(?:\.\d+)?)", key_factors)
    if elo_match:
        factors.append({"name": "elo_diff", "value": round(float(elo_match.group(1)), 2)})

    recent_lines = re.findall(
        r"recent goals for\s+([0-9.]+),\s+against\s+([0-9.]+)",
        key_factors,
        flags=re.IGNORECASE,
    )
    if len(recent_lines) >= 2:
        home_for, home_against = map(float, recent_lines[0])
        away_for, away_against = map(float, recent_lines[1])
        factors.append({"name": "recent_attack", "value": round((home_for - away_for) * 100.0, 2)})
        factors.append({"name": "defence_form", "value": round((away_against - home_against) * 100.0, 2)})

    rank_match = re.search(r"FIFA rank edge feature\s+([+-]?\d+(?:\.\d+)?)", key_factors)
    if rank_match:
        factors.append({"name": "fifa_rank_edge", "value": round(float(rank_match.group(1)) * 100.0, 2)})
    return sorted(factors, key=lambda factor: abs(float(factor["value"])), reverse=True)


def parse_data_sources(key_factors: str, row_source: str = "") -> dict[str, str]:
    elo_source = first_source_value(key_factors, ("data_source_home_elo=", "data_source_away_elo="))
    form_source = first_source_value(key_factors, ("data_source_home_recent=", "data_source_away_recent="))
    if not elo_source:
        elo_source = row_source
    if not form_source:
        form_source = row_source
    return {"elo": source_label(elo_source), "form": source_label(form_source)}


def first_source_value(text: str, prefixes: tuple[str, ...]) -> str:
    for part in str(text or "").split("|"):
        stripped = part.strip()
        for prefix in prefixes:
            if stripped.startswith(prefix):
                return stripped.split("=", 1)[1].strip()
    return ""


def source_label(source: object) -> str:
    value = str(source or "").strip()
    upper = value.upper()
    if "ESPN" in upper:
        return "ESPN"
    if "ELO" in upper:
        return "ELO"
    return "Estimated"


def create_match_explain_router() -> Any:
    """Return a FastAPI router for GET /match/{id}/explain when FastAPI is installed."""

    try:
        from fastapi import APIRouter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install fastapi to mount /match/{id}/explain") from exc

    router = APIRouter()

    @router.get(
        "/match/{id}/explain",
        summary="Return top model factors and data provenance for one match",
        tags=["match"],
        openapi_extra=OPENAPI_MATCH_EXPLAIN["paths"]["/match/{id}/explain"]["get"],
    )
    def explain_match(id: str) -> dict[str, object]:
        return explain_match_json(id)

    return router
