"""Automated leakage checks. Sec 26 + 37: the build must fail loudly if
critical leakage is detected.
"""

from __future__ import annotations

from abusering.features.build import FEATURE_COLUMNS

FORBIDDEN_FEATURE_COLUMNS = {
    "ring_id", "abuse_type", "generator_type", "ground_truth_label", "abuse_label",
    "scenario_id", "customer_id", "transaction_id",
}


def check_no_label_leakage(feature_columns: list[str]) -> list[str]:
    violations = [c for c in feature_columns if c in FORBIDDEN_FEATURE_COLUMNS]
    return violations


def check_feature_columns_match_declared(feature_columns: list[str]) -> list[str]:
    """Every column actually fed to the model must be in the declared,
    audited FEATURE_COLUMNS list — prevents an accidental extra column
    (e.g. a raw id) from silently entering the model."""
    return [c for c in feature_columns if c not in FEATURE_COLUMNS]


def check_temporal_ordering(features_df, transactions_df) -> list[str]:
    """Sanity check: prediction_time in the feature table must equal the
    transaction's own timestamp (features are computed AT the transaction's
    own time using only strictly-prior events — verified structurally by
    the single forward pass in features/build.py, and here we just confirm
    the join key / time alignment is intact)."""
    merged = features_df.merge(
        transactions_df[["transaction_id", "timestamp"]], on="transaction_id", how="left"
    )
    mismatches = merged[merged["prediction_time"] != merged["timestamp"]]
    return mismatches["transaction_id"].tolist()


def run_all_checks(feature_columns: list[str], features_df=None, transactions_df=None) -> dict:
    label_violations = check_no_label_leakage(feature_columns)
    undeclared = check_feature_columns_match_declared(feature_columns)
    time_violations = []
    if features_df is not None and transactions_df is not None:
        time_violations = check_temporal_ordering(features_df, transactions_df)

    passed = not label_violations and not undeclared and not time_violations
    return {
        "passed": passed,
        "label_leakage_violations": label_violations,
        "undeclared_columns": undeclared,
        "temporal_ordering_violations_sample": time_violations[:20],
        "temporal_ordering_violation_count": len(time_violations),
    }
