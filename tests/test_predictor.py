"""Tests for NoShowPredictor, using a small model built on the fly."""

import json

import joblib
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshows import predictor as predictor_module
from noshows.appointment import Appointment
from noshows.data import make_appointments
from noshows.predictor import NoShowPredictor, Prediction

NUMERIC = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]
CATEGORICAL = ["weekday", "booking", "insurance", "treatment"]
THRESHOLD = 0.3

SAFE = {
    "lead_days": 1,
    "age": 60,
    "prior_no_shows": 0,
    "hour": 12,
    "weekday": "wed",
    "booking": "front_desk",
    "insurance": "provincial",
    "treatment": "acupuncture",
    "first_visit": 0,
    "distance_km": 2.0,
}
RISKY = {
    "lead_days": 50,
    "age": 20,
    "prior_no_shows": 4,
    "hour": 8,
    "weekday": "fri",
    "booking": "online",
    "insurance": "self_pay",
    "treatment": "physio",
    "first_visit": 1,
    "distance_km": 30.0,
}


def make(**fields):
    """Build an Appointment from a dictionary of fields."""
    return Appointment(**fields)


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    """Fit a small model and save it with a model_info.json."""
    folder = tmp_path_factory.mktemp("models")
    data = make_appointments()
    numbers = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numbers, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    pipeline = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
    pipeline.fit(data[NUMERIC + CATEGORICAL], data["no_show"])
    info = {
        "threshold": THRESHOLD,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "excluded_columns": ["appointment_id", "fee_charged"],
    }
    model_path = folder / "no_show.joblib"
    info_path = folder / "model_info.json"
    joblib.dump(pipeline, model_path)
    info_path.write_text(json.dumps(info))
    return model_path, info_path


@pytest.fixture
def predictor(model_files):
    return NoShowPredictor(*model_files)


def test_threshold_comes_from_model_info(predictor):
    assert predictor.threshold == THRESHOLD


def test_repr_is_readable(predictor):
    assert repr(predictor) == "NoShowPredictor(threshold=0.30)"


def test_probability_is_between_0_and_1(predictor):
    for fields in (SAFE, RISKY):
        assert 0.0 <= predictor.probability(make(**fields)) <= 1.0


def test_riskier_booking_scores_higher(predictor):
    safe = predictor.probability(make(**SAFE))
    risky = predictor.probability(make(**RISKY))
    assert risky > safe


@pytest.mark.parametrize(
    ("probability", "expected_call"),
    [
        (THRESHOLD, True),
        (THRESHOLD - 0.0001, False),
        (THRESHOLD + 0.0001, True),
        (0.0, False),
        (1.0, True),
    ],
)
def test_decision_at_just_below_and_just_above_threshold(
    predictor, probability, expected_call
):
    assert predictor.decide(probability).call is expected_call


def test_predict_returns_probability_and_action_together(predictor):
    prediction = predictor.predict(make(**RISKY))
    assert isinstance(prediction, Prediction)
    assert prediction.call == (prediction.probability >= predictor.threshold)
    assert prediction.action


def test_missing_distance_can_be_scored(predictor):
    appointment = make(**{**SAFE, "distance_km": None})
    assert 0.0 <= predictor.probability(appointment) <= 1.0


def test_model_file_is_read_once(model_files, monkeypatch):
    real_load = joblib.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(predictor_module.joblib, "load", counting_load)
    predictor = NoShowPredictor(*model_files)
    predictor.predict(make(**RISKY))
    predictor.predict(make(**SAFE))
    assert len(calls) == 1
