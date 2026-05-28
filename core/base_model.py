from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace

from .prediction_result import PredictionResult


class SportPredictor(ABC):
    @abstractmethod
    def predict(self, args: Namespace) -> list[PredictionResult]:
        raise NotImplementedError

    @abstractmethod
    def backtest(self, args: Namespace) -> dict[str, object]:
        raise NotImplementedError

