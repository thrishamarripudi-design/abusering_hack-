"""Baselines (Sec 11). The rules baseline represents a plausible
non-ML operational approach and must be beaten to justify ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_baseline_scores(n: int, seed: int, positive_rate: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=n)


def majority_baseline_scores(n: int) -> np.ndarray:
    # majority class is "legitimate" -> always score 0
    return np.zeros(n)


def rule_based_scores(df: pd.DataFrame, thresholds: dict | None = None) -> np.ndarray:
    """Configurable rule detector using coordination signals: shared-device
    count, shared-IP count, transaction velocity, payment-instrument reuse.
    Produces a 0-1 "rule score" = fraction of rules tripped, so it can be
    thresholded like a model score for a fair comparison."""
    t = thresholds or {
        "shared_device_count": 4,
        "shared_ip_count": 6,
        "ip_transaction_velocity_1h": 5,
        "shared_instrument_count": 3,
        "transactions_in_time_window_15m": 4,
    }
    hits = np.zeros(len(df))
    n_rules = len(t)
    for col, thr in t.items():
        hits += (df[col].to_numpy() >= thr).astype(float)
    return hits / n_rules
