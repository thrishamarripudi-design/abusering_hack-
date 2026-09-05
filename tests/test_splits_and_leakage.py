import pandas as pd

from abusering.evaluation.leakage import check_feature_columns_match_declared, check_no_label_leakage
from abusering.features.build import FEATURE_COLUMNS
from abusering.splits.ring_split import make_splits, verify_no_ring_overlap


def _fake_ground_truth():
    rows = []
    for i in range(10):
        ring_id = f"RING_{i:03d}"
        for j in range(5):
            rows.append(
                {
                    "transaction_id": f"A_{i}_{j}",
                    "ring_id": ring_id,
                    "abuse_type": "shared_device",
                    "abuse_label": 1,
                }
            )
    for k in range(200):
        rows.append({"transaction_id": f"L_{k}", "ring_id": None, "abuse_type": None, "abuse_label": 0})
    return pd.DataFrame(rows)


def test_no_ring_overlap():
    gt = _fake_ground_truth()
    manifest = make_splits(gt, seed=1)
    verify_no_ring_overlap(manifest)  # should not raise

    tr, va, te = (
        set(manifest["train_ring_ids"]),
        set(manifest["validation_ring_ids"]),
        set(manifest["test_ring_ids"]),
    )
    assert tr and va and te
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_split_covers_all_transactions():
    gt = _fake_ground_truth()
    manifest = make_splits(gt, seed=1)
    all_split_txns = (
        set(manifest["train_transaction_ids"])
        | set(manifest["validation_transaction_ids"])
        | set(manifest["test_transaction_ids"])
    )
    assert all_split_txns == set(gt["transaction_id"])


def test_split_deterministic_given_seed():
    gt = _fake_ground_truth()
    m1 = make_splits(gt, seed=99)
    m2 = make_splits(gt, seed=99)
    assert m1["train_ring_ids"] == m2["train_ring_ids"]
    assert m1["test_ring_ids"] == m2["test_ring_ids"]


def test_no_label_leakage_columns():
    assert check_no_label_leakage(FEATURE_COLUMNS) == []


def test_undeclared_columns_detected():
    bad_cols = FEATURE_COLUMNS + ["ring_id"]
    violations = check_no_label_leakage(bad_cols)
    assert "ring_id" in violations


def test_declared_columns_match():
    assert check_feature_columns_match_declared(FEATURE_COLUMNS) == []
    assert "not_a_real_feature" in check_feature_columns_match_declared(
        FEATURE_COLUMNS + ["not_a_real_feature"]
    )
