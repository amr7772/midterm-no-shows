"""The cost rule, the threshold choice and the test-set summary."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CALL_COST = 8  # a reminder call costs $8 of staff time, and the patient then comes
MISSED_COST = 35  # a no-show that nobody phoned about costs the clinic $35
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]  # 0.05 to 0.95


def total_cost(
    y_true, flagged, call_cost: float = CALL_COST, missed_cost: float = MISSED_COST
) -> float:
    """$8 for every patient flagged, plus $35 for every no-show not flagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    calls = int(flagged.sum())
    missed = int(((y_true == 1) & ~flagged).sum())
    return call_cost * calls + missed_cost * missed


def cost_by_threshold(
    y_true,
    probabilities,
    thresholds=THRESHOLDS,
    call_cost: float = CALL_COST,
    missed_cost: float = MISSED_COST,
) -> dict:
    """Return {threshold: total cost} when every booking at or above it is called."""
    probabilities = np.asarray(probabilities)
    return {
        t: total_cost(y_true, probabilities >= t, call_cost, missed_cost)
        for t in thresholds
    }


def choose_threshold(
    y_true,
    probabilities,
    thresholds=THRESHOLDS,
    call_cost: float = CALL_COST,
    missed_cost: float = MISSED_COST,
) -> float:
    """Return the threshold with the lowest total cost (the highest one on a tie)."""
    costs = cost_by_threshold(y_true, probabilities, thresholds, call_cost, missed_cost)
    # on a tie, prefer the higher threshold: the same cost with fewer $8 calls
    return min(costs, key=lambda t: (costs[t], -t))


def policy_costs(y_true, probabilities, threshold: float) -> dict:
    """Cost of calling nobody, calling everybody, and calling whom the model flags."""
    y_true = np.asarray(y_true)
    nobody = np.zeros(len(y_true), dtype=bool)
    everybody = np.ones(len(y_true), dtype=bool)
    model = np.asarray(probabilities) >= threshold
    return {
        "call_nobody": total_cost(y_true, nobody),
        "call_everybody": total_cost(y_true, everybody),
        "model": total_cost(y_true, model),
    }


def summarize(y_true, probabilities, threshold: float) -> dict:
    """Confusion matrix, metrics and the three costs at the chosen threshold."""
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "appointments": int(len(predicted)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": round(float(precision_score(y_true, predicted)), 3),
        "recall": round(float(recall_score(y_true, predicted)), 3),
        "f1": round(float(f1_score(y_true, predicted)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 3),
        "costs": policy_costs(y_true, probabilities, threshold),
    }
