"""Compare models by cross-validation on the training data only."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from noshows.data import make_appointments
from noshows.pipeline import (
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_pipeline,
    split_data,
)


def compare_models(x_train, y_train):
    """Run 5-fold cross-validation for each model and return the table."""
    models = {
        "dummy": DummyClassifier(),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "hist_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in models.items():
        pipe = build_pipeline(model)
        results = cross_validate(pipe, x_train, y_train, cv=folds, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            scores = results["test_" + metric]
            row[metric + "_mean"] = scores.mean()
            row[metric + "_std"] = scores.std()
        rows.append(row)
    return pd.DataFrame(rows)


def tune(x_train, y_train):
    """Tune the chosen logistic model, scoring by ROC-AUC."""
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": ["balanced", None],
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=folds)
    search.fit(x_train, y_train)
    return search


def leakage_check(frame):
    """Run CV with fee_charged added, to show the leak in plain numbers."""
    x = frame[FEATURES + ["fee_charged"]]
    y = frame[TARGET]
    # Same split settings as split_data, so the same rows are used.
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(model, numeric=NUMERIC_FEATURES + ["fee_charged"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(pipe, x_train, y_train, cv=folds, scoring=["roc_auc"])
    return results["test_roc_auc"].mean(), results["test_roc_auc"].std()


if __name__ == "__main__":
    frame = make_appointments()
    x_train, _, y_train, _ = split_data(frame)
    table = compare_models(x_train, y_train)
    print(table.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/cv_results.csv", index=False)
    logistic = table[table["model"] == "logistic_balanced"].iloc[0]
    print(
        "Untuned logistic ROC-AUC:",
        round(float(logistic["roc_auc_mean"]), 4),
        "+/-",
        round(float(logistic["roc_auc_std"]), 4),
    )
    search = tune(x_train, y_train)
    print("Best settings:", search.best_params_)
    print("Best ROC-AUC:", round(float(search.best_score_), 4))
    leak_mean, leak_std = leakage_check(frame)
    print(
        "Leakage ROC-AUC:",
        round(float(leak_mean), 4),
        "+/-",
        round(float(leak_std), 4),
    )
