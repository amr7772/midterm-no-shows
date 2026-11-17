"""Compare, tune, choose the threshold, test once and save: one command.

Run it from the top of the repository with: python -m noshows.train
"""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from noshows.compare import compare_models, tune
from noshows.data import make_appointments
from noshows.evaluation import choose_threshold, cost_by_threshold, summarize
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent  # the top of the repository
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LEAKY_COLUMN = "fee_charged"  # recorded after the visit, so it gives the answer away


def make_folds() -> StratifiedKFold:
    """Five stratified folds, shuffled with the fixed seed."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def out_of_fold_probabilities(pipeline, X_train, y_train):
    """Each training row is scored by a model that did not see it during fitting."""
    return cross_val_predict(
        pipeline, X_train, y_train, cv=make_folds(), method="predict_proba"
    )[:, 1]


def leakage_check(frame, X_train, y_train, model) -> dict:
    """Cross-validated ROC-AUC without and with the after-the-visit column."""
    honest = cross_val_score(
        build_pipeline(clone(model)),
        X_train,
        y_train,
        cv=make_folds(),
        scoring="roc_auc",
    )
    # the same training rows, taken back from the full table with the leaky column
    X_leaky = frame.loc[X_train.index, FEATURES + [LEAKY_COLUMN]]
    leaky = cross_val_score(
        build_pipeline(clone(model), numeric=NUMERIC_FEATURES + [LEAKY_COLUMN]),
        X_leaky,
        y_train,
        cv=make_folds(),
        scoring="roc_auc",
    )
    return {
        "column": LEAKY_COLUMN,
        "honest_cv_roc_auc": round(float(honest.mean()), 3),
        "with_leaky_column_cv_roc_auc": round(float(leaky.mean()), 3),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)

    # 1. Compare the candidates on the training data only (Part 2's functions).
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    print(comparison.to_string(index=False))

    # 2. Tune the chosen model. GridSearchCV refits the best one on all training rows.
    search = tune(X_train, y_train)
    final = search.best_estimator_
    print("Best settings:", search.best_params_)

    # 3. Choose the threshold from out-of-fold training probabilities only.
    oof = out_of_fold_probabilities(final, X_train, y_train)
    threshold = choose_threshold(y_train, oof)
    cv_costs = cost_by_threshold(y_train, oof)
    print("Threshold chosen on the training data:", threshold)

    # 4. See leakage for yourself: the same model with the after-the-visit column.
    leakage = leakage_check(frame, X_train, y_train, final.named_steps["model"])
    print("Leakage check:", leakage)

    # 5. The test set, used once, at the very end.
    test_probabilities = final.predict_proba(X_test)[:, 1]
    test_results = summarize(y_test, test_probabilities, threshold)
    print("Test set:", test_results)

    # 6. Save the whole fitted pipeline and everything needed to trust it.
    joblib.dump(final, MODELS_DIR / "no_show.joblib")
    info = {
        "scikit_learn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": EXCLUDED,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_score": round(float(search.best_score_), 3),
        "test_results": test_results,
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    metrics = {
        "test": test_results,
        "leakage_check": leakage,
        "cv_cost_by_threshold": {str(t): c for t, c in cv_costs.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/no_show.joblib, models/model_info.json, reports/metrics.json")


if __name__ == "__main__":
    main()
