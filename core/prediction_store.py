from __future__ import annotations

try:
    from ..config import MASTER_PREDICTIONS_CSV, OUTPUT_PREDICTIONS_CSV
    from .prediction_result import PREDICTION_FIELDNAMES, PredictionResult
    from .utils import append_csv_row
except ImportError:
    from config import MASTER_PREDICTIONS_CSV, OUTPUT_PREDICTIONS_CSV
    from core.prediction_result import PREDICTION_FIELDNAMES, PredictionResult
    from core.utils import append_csv_row


def save_prediction_outputs(result: PredictionResult) -> None:
    """Save every generated prediction to the unified output files."""

    row = result.to_row()
    fieldnames = list(PREDICTION_FIELDNAMES)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    append_csv_row(OUTPUT_PREDICTIONS_CSV, row, fieldnames)
    append_csv_row(MASTER_PREDICTIONS_CSV, row, fieldnames)
