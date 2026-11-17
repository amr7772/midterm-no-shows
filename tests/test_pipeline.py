"""Tests for the split and the pipeline."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
    split_data,
)


def test_excluded_columns_not_in_features():
    assert "appointment_id" not in FEATURES
    assert "fee_charged" not in FEATURES
    assert TARGET not in FEATURES
    assert len(FEATURES) == 10
    assert sorted(FEATURES) == sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_split_size_and_stratified():
    frame = make_appointments()
    x_train, x_test, y_train, y_test = split_data(frame)
    assert len(x_train) == 3000
    assert len(x_test) == 1000
    assert len(y_train) == 3000
    assert len(y_test) == 1000
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_pipeline_handles_missing_distance():
    frame = make_appointments()
    x_train, x_test, y_train, _ = split_data(frame)
    assert x_train["distance_km"].isna().any()
    pipe = build_pipeline(DummyClassifier())
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    assert len(preds) == 1000
    assert set(preds) <= {0, 1}


def test_pipeline_with_logistic_regression():
    frame = make_appointments()
    x_train, x_test, y_train, _ = split_data(frame)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]
    assert len(probs) == 1000
    assert np.all((probs >= 0) & (probs <= 1))
