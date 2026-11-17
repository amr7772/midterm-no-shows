"""Features, split, and pipeline for the no-show check."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "weekday",
    "booking",
    "insurance",
    "treatment",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED = ["appointment_id", "fee_charged"]

TARGET = "no_show"


def split_data(frame):
    """Split into train and test, keeping the no-show share the same."""
    x = frame[FEATURES]
    y = frame[TARGET]
    return train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


def build_pipeline(model, numeric=None, categorical=None):
    """Put the cleaning steps in front of any model."""
    # Use the usual lists when no lists are given.
    if numeric is None:
        numeric = NUMERIC_FEATURES
    if categorical is None:
        categorical = CATEGORICAL_FEATURES
    numbers = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categories = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    clean = ColumnTransformer(
        [
            ("num", numbers, numeric),
            ("cat", categories, categorical),
        ]
    )
    return Pipeline([("clean", clean), ("model", model)])
