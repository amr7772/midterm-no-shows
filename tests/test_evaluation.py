"""Tests for the cost rule and the threshold choice, on examples worked out by hand."""

import pytest

from noshows.evaluation import (
    THRESHOLDS,
    choose_threshold,
    cost_by_threshold,
    policy_costs,
    summarize,
    total_cost,
)


def test_cost_of_a_small_example_worked_out_by_hand():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two calls (2 x 8 = 16) and one no-show nobody phoned (1 x 35 = 35)
    assert total_cost(y_true, flagged) == 2 * 8 + 1 * 35 == 51


def test_calling_nobody_is_cheapest_when_nobody_misses():
    y_true = [0, 0, 0, 0]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    costs = policy_costs(y_true, probabilities, threshold=0.5)
    # nobody misses, so every call is wasted: calling nobody costs $0
    assert costs["call_nobody"] == 0
    assert costs["call_nobody"] < costs["call_everybody"] == 4 * 8


def test_calling_everybody_is_cheapest_when_everybody_misses():
    y_true = [1, 1, 1, 1]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    costs = policy_costs(y_true, probabilities, threshold=0.5)
    # four calls (4 x 8 = 32) are far cheaper than four misses (4 x 35 = 140)
    assert costs["call_everybody"] == 32
    assert costs["call_everybody"] < costs["call_nobody"] == 140


def test_threshold_on_a_hand_worked_example():
    y_true = [1, 0, 0, 1]
    probabilities = [0.9, 0.6, 0.2, 0.4]
    # 0.05 to 0.20: call all four                    -> 4 x 8          = 32
    # 0.25 to 0.40: call 0.9, 0.6, 0.4; miss nobody  -> 3 x 8          = 24 (lowest)
    # 0.45 to 0.60: call 0.9, 0.6; miss the 0.4      -> 2 x 8 + 35     = 51
    # 0.65 to 0.90: call 0.9; miss the 0.4           -> 8 + 35         = 43
    # 0.95:         call nobody; miss both           -> 2 x 35         = 70
    # on a tie the higher threshold wins: the same $24 with the fewest calls
    assert choose_threshold(y_true, probabilities) == 0.40
    assert cost_by_threshold(y_true, probabilities)[0.40] == 24


@pytest.mark.parametrize(
    ("call_cost", "missed_cost", "expected"),
    [
        (100, 35, 0.95),  # a call costs more than a miss: call nobody
        (8, 10_000, 0.05),  # a miss is very costly: call everybody
    ],
)
def test_threshold_at_both_extremes(call_cost, missed_cost, expected):
    # the lowest score is a no-show, and every score is below 0.95
    y_true = [1, 0, 1, 0, 0]
    probabilities = [0.07, 0.2, 0.4, 0.6, 0.92]
    chosen = choose_threshold(
        y_true, probabilities, call_cost=call_cost, missed_cost=missed_cost
    )
    assert chosen == expected


def test_thresholds_run_from_005_to_095():
    assert THRESHOLDS[0] == 0.05 and THRESHOLDS[-1] == 0.95
    assert len(THRESHOLDS) == 19


def test_summary_holds_the_three_costs_and_the_confusion_matrix():
    y_true = [1, 0, 0, 1]
    probabilities = [0.9, 0.6, 0.2, 0.4]
    result = summarize(y_true, probabilities, threshold=0.5)
    assert result["costs"] == {"call_nobody": 70, "call_everybody": 32, "model": 51}
    matrix = result["confusion_matrix"]
    assert sum(matrix.values()) == 4
    assert matrix["true_positive"] == 1 and matrix["false_negative"] == 1
