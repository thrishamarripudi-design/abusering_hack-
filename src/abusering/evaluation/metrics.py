"""Metrics, financial cost model, and threshold optimization.

Cost config is externalized (Sec 16) — not hardcoded into model code.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_COST_CONFIG = {
    "false_positive_cost": 500.0,       # cost of wrongly blocking/reviewing a legit txn
    "investigation_cost": 100.0,        # cost of manually investigating a flagged txn
    "loss_rate_on_missed_abuse": 0.85,  # fraction of txn amount lost if a true abuse txn is missed (FN)
}


def load_cost_config(path: str | pathlib.Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_COST_CONFIG)
    p = pathlib.Path(path)
    if not p.exists():
        return dict(DEFAULT_COST_CONFIG)
    with open(p) as f:
        cfg = json.load(f)
    merged = dict(DEFAULT_COST_CONFIG)
    merged.update(cfg)
    return merged


def expected_loss(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray, cost_cfg: dict) -> float:
    """total_expected_loss = FP cost + FN cost + investigation cost, per Sec 16."""
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)

    fp_cost = fp_mask.sum() * cost_cfg["false_positive_cost"]
    fn_cost = float(np.sum(amounts[fn_mask]) * cost_cfg["loss_rate_on_missed_abuse"])
    investigation_cost = (fp_mask.sum() + tp_mask.sum()) * cost_cfg["investigation_cost"]
    return float(fp_cost + fn_cost + investigation_cost)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray, cost_cfg: dict) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "positive_prediction_rate": float((y_pred == 1).mean()),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "expected_financial_loss": expected_loss(y_true, y_pred, amounts, cost_cfg),
    }
    return metrics


def precision_at_recall(y_true, y_score, recall_target: float) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    mask = recalls >= recall_target
    return float(precisions[mask].max()) if mask.any() else 0.0


def recall_at_precision(y_true, y_score, precision_target: float) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    mask = precisions >= precision_target
    return float(recalls[mask].max()) if mask.any() else 0.0


def optimize_threshold(y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray, cost_cfg: dict,
                        n_candidates: int = 199) -> dict:
    """Grid search thresholds, pick the one minimizing expected financial
    loss on the given (validation) set. Sec 17: never touch the test set here."""
    candidates = np.linspace(0.01, 0.99, n_candidates)
    best = None
    curve = []
    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        loss = expected_loss(y_true, y_pred, amounts, cost_cfg)
        row = {
            "threshold": float(thr),
            "expected_loss": loss,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
        }
        curve.append(row)
        if best is None or loss < best["expected_loss"]:
            best = row
    return {"selected_threshold": best["threshold"], "selected_row": best, "curve": curve}
