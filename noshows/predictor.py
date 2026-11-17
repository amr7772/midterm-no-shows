"""Turn a booking into a chance of a no-show and an action."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "no_show.joblib"
INFO_PATH = MODELS_DIR / "model_info.json"


class NoShowPredictor:
    """Loads the saved pipeline and model_info.json once, when created."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        info_path: str | Path = INFO_PATH,
    ) -> None:
        self.pipeline = joblib.load(model_path)
        self.info = json.loads(Path(info_path).read_text())
        self.threshold = float(self.info["threshold"])

    def __repr__(self) -> str:
        return f"NoShowPredictor(threshold={self.threshold:.2f})"