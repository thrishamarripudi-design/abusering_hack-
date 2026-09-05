import numpy as np

from abusering.evaluation.metrics import (
    DEFAULT_COST_CONFIG,
    compute_metrics,
    expected_loss,
    optimize_threshold,
)


def test_expected_loss_zero_for_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    amounts = np.array([10.0, 20.0, 30.0, 40.0])
    loss = expected_loss(y_true, y_pred, amounts, DEFAULT_COST_CONFIG)
    # 2 true positives incur investigation cost only, no FP/FN cost
    expected = 2 * DEFAULT_COST_CONFIG["investigation_cost"]
    assert loss == expected


def test_expected_loss_penalizes_false_negatives_by_amount():
    y_true = np.array([1])
    y_pred = np.array([0])
    amounts = np.array([1000.0])
    loss = expected_loss(y_true, y_pred, amounts, DEFAULT_COST_CONFIG)
    assert loss == 1000.0 * DEFAULT_COST_CONFIG["loss_rate_on_missed_abuse"]


def test_expected_loss_penalizes_false_positives_fixed_cost():
    y_true = np.array([0])
    y_pred = np.array([1])
    amounts = np.array([99999.0])  # amount should NOT affect FP cost
    loss = expected_loss(y_true, y_pred, amounts, DEFAULT_COST_CONFIG)
    assert loss == DEFAULT_COST_CONFIG["false_positive_cost"] + DEFAULT_COST_CONFIG["investigation_cost"]


def test_optimize_threshold_never_touches_test_set_by_construction():
    """optimize_threshold only accepts one (y_true, y_score) pair — the
    caller is responsible for passing validation data, and this test just
    confirms the function is pure / deterministic given its inputs."""
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_score = rng.random(500)
    amounts = rng.uniform(10, 500, size=500)
    r1 = optimize_threshold(y_true, y_score, amounts, DEFAULT_COST_CONFIG, n_candidates=50)
    r2 = optimize_threshold(y_true, y_score, amounts, DEFAULT_COST_CONFIG, n_candidates=50)
    assert r1["selected_threshold"] == r2["selected_threshold"]


def test_compute_metrics_confusion_matrix_consistency():
    y_true = np.array([0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    y_pred = (y_score >= 0.5).astype(int)
    amounts = np.array([10, 20, 30, 40, 50], dtype=float)
    m = compute_metrics(y_true, y_score, y_pred, amounts, DEFAULT_COST_CONFIG)
    assert m["true_positives"] + m["false_negatives"] == int(y_true.sum())
    assert m["true_negatives"] + m["false_positives"] == int((y_true == 0).sum())
