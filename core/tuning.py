from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..config import WEIGHT_TUNING_JSON
    from .utils import safe_float
except ImportError:
    from config import WEIGHT_TUNING_JSON
    from core.utils import safe_float


FEATURE_COLUMNS = {
    "elo_weight": "elo_difference",
    "fatigue_weight": "fatigue_edge",
    "injury_weight": "injury_edge",
    "momentum_weight": "recent_form_edge",
    "home_advantage_weight": "home_advantage_edge",
}


def tune_model_weights(frame: pd.DataFrame, sport: str) -> dict[str, Any]:
    """Small deterministic grid search over available backtest feature columns."""

    if frame.empty or "actual_result" not in frame:
        return default_weights(sport, "missing_backtest_data")
    available = {name: column for name, column in FEATURE_COLUMNS.items() if column in frame.columns}
    if not available:
        return default_weights(sport, "missing_feature_columns")

    baseline = [safe_float(value, 0.5) for value in frame.get("predicted_probability", [])]
    actual = [int(safe_float(value, 0.0)) for value in frame["actual_result"].tolist()]
    if not baseline or len(baseline) != len(actual):
        return default_weights(sport, "invalid_backtest_rows")

    feature_values = {
        name: normalize_series([safe_float(value, 0.0) for value in frame[column].tolist()])
        for name, column in available.items()
    }
    candidates = [0.0, 0.15, 0.3, 0.45]
    best_score = -1.0
    best_weights = {name: 0.0 for name in FEATURE_COLUMNS}
    for combo in itertools.product(candidates, repeat=len(feature_values)):
        weights = dict(zip(feature_values.keys(), combo))
        correct = 0
        for idx, prob in enumerate(baseline):
            adjusted = logit(prob)
            for name, weight in weights.items():
                adjusted += weight * feature_values[name][idx]
            predicted = sigmoid(adjusted) >= 0.5
            correct += int(predicted == bool(actual[idx]))
        score = correct / len(actual)
        if score > best_score:
            best_score = score
            best_weights = {**best_weights, **weights}

    result = {
        "sport": sport,
        "accuracy": round(best_score, 4),
        "weights": {key: round(value, 3) for key, value in best_weights.items()},
        "available_features": list(available.values()),
        "status": "ok",
    }
    save_tuning_result(result)
    return result


def default_weights(sport: str, reason: str) -> dict[str, Any]:
    result = {
        "sport": sport,
        "accuracy": 0.0,
        "weights": {
            "elo_weight": 0.22,
            "fatigue_weight": 0.12,
            "injury_weight": 0.16,
            "momentum_weight": 0.14,
            "home_advantage_weight": 0.08,
        },
        "available_features": [],
        "status": reason,
    }
    save_tuning_result(result)
    return result


def save_tuning_result(result: dict[str, Any], path: Path = WEIGHT_TUNING_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[result["sport"]] = result
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    std = math.sqrt(variance) or 1.0
    return [(value - avg) / std for value in values]


def logit(probability: float) -> float:
    probability = max(0.01, min(0.99, probability))
    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))

