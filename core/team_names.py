from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM_ALIASES_PATH = PROJECT_ROOT / "team_aliases.json"


def _clean_key(name: object) -> str:
    value = "" if name is None else str(name)
    value = value.replace("\u00a0", " ").replace("&amp;", "&")
    value = re.sub(r"\s+", " ", value.strip())
    return value


def _lookup_key(name: object) -> str:
    return _clean_key(name).casefold()


@lru_cache(maxsize=1)
def load_team_aliases() -> dict[str, str]:
    if not TEAM_ALIASES_PATH.exists():
        return {}
    try:
        payload = json.loads(TEAM_ALIASES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    aliases: dict[str, str] = {}
    for raw_name, canonical in payload.items():
        aliases[_lookup_key(raw_name)] = _clean_key(canonical)
        aliases[_lookup_key(canonical)] = _clean_key(canonical)
    return aliases


def normalize_team_name(name: object) -> str:
    cleaned = _clean_key(name)
    if not cleaned:
        return ""
    aliases = load_team_aliases()
    return aliases.get(_lookup_key(cleaned), cleaned)


def normalized_team_key(name: object) -> str:
    normalized = normalize_team_name(name)
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.casefold())
    return " ".join(normalized.split())


def team_names_match(left: object, right: object) -> bool:
    left_key = normalized_team_key(left)
    right_key = normalized_team_key(right)
    return bool(left_key and right_key and left_key == right_key)
