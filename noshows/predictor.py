"""Turn a booking into a chance of a no-show and an action."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from noshows.appointment import Appointment

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "no_show.joblib"
INFO_PATH = MODELS_DIR / "model_info.json"

CALL_ACTION = "Call the patient the day before to confirm"
NO_CALL_ACTION = "No call needed"


@dataclass(frozen=True)
class Prediction:
    """The chance of a no-show and the action that goes with it."""

    probability: float
    call: bool
    action: str


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
        self.feature_columns = (
            self.info["numeric_features"] + self.info["categorical_features"]
        )

    def __repr__(self) -> str:
        return f"NoShowPredictor(threshold={self.threshold:.2f})"

    def probability(self, appointment: Appointment) -> float:
        """Return the chance, from 0 to 1, that the patient does not come."""
        frame = appointment.to_frame()[self.feature_columns]
        return float(self.pipeline.predict_proba(frame)[0, 1])

    def decide(self, probability: float) -> Prediction:
        """Flag for a call when the chance reaches the threshold."""
        call = probability >= self.threshold
        action = CALL_ACTION if call else NO_CALL_ACTION
        return Prediction(probability=probability, call=call, action=action)

    def predict(self, appointment: Appointment) -> Prediction:
        """Return the probability and the action together."""
        return self.decide(self.probability(appointment))
