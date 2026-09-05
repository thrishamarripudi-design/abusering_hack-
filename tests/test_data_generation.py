import pandas as pd
import pytest

from abusering.data.config import get_config
from abusering.data.generator import generate_dataset


@pytest.fixture(scope="module")
def small_cfg():
    cfg = get_config("demo", seed=7)
    # shrink for fast tests
    from dataclasses import replace

    return replace(cfg, n_customers=600, n_merchants=100, n_transactions=2500, n_rings=6)


@pytest.fixture(scope="module")
def dataset(small_cfg):
    return generate_dataset(small_cfg)


def test_deterministic_generation(small_cfg):
    ds1 = generate_dataset(small_cfg)
    ds2 = generate_dataset(small_cfg)
    pd.testing.assert_frame_equal(
        ds1.transactions.reset_index(drop=True), ds2.transactions.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        ds1.ground_truth.reset_index(drop=True), ds2.ground_truth.reset_index(drop=True)
    )


def test_ground_truth_valid(dataset):
    gt = dataset.ground_truth
    assert set(gt["abuse_label"].unique()) <= {0, 1}
    assert gt["transaction_id"].is_unique
    # every transaction has exactly one ground truth row
    assert set(gt["transaction_id"]) == set(dataset.transactions["transaction_id"])
    # abuse rows have a ring_id, legit rows don't
    assert gt.loc[gt.abuse_label == 1, "ring_id"].notna().all()
    assert gt.loc[gt.abuse_label == 0, "ring_id"].isna().all()


def test_legitimate_lookalikes_exist(dataset):
    seg_counts = dataset.customers["customer_segment"].value_counts()
    for seg in ["family_shared_device", "office_shared_ip", "hostel_shared_network"]:
        assert seg_counts.get(seg, 0) > 0, f"missing legitimate lookalike segment {seg}"


def test_abuse_rings_exist(dataset):
    assert len(dataset.ring_stats) > 0
    assert dataset.ring_stats["ring_size_accounts"].min() >= 2


def test_no_negative_account_age_at_ring_level(dataset):
    """Every ring-synthetic customer's account_created_at must precede
    (or equal) every transaction timestamp attributed to them."""
    merged = dataset.transactions.merge(dataset.customers, on="customer_id")

    # Only check synthetic abuse-ring customers
    merged = merged[merged["customer_segment"] == "ring_synthetic"]

    bad = merged[merged["timestamp"] < merged["account_created_at"]]

    assert len(bad) == 0, (
        f"{len(bad)} ring-synthetic transactions occur before their customer's account creation"
    )


def test_amount_positive(dataset):
    assert (dataset.transactions["amount"] > 0).all()


def test_ring_scale_keeps_abuse_rate_near_target():
    """Regression test for a real bug: fixed absolute ring sizes diluted
    the abuse rate to ~0.9% at the `full` profile's scale (100 rings vs.
    a 12.5x larger transaction count) against a 6% target. Ring size must
    scale with the target abuse-transaction volume instead."""
    from dataclasses import replace

    small_demo_cfg = replace(
        get_config("demo", seed=3), n_customers=600, n_merchants=100, n_transactions=2500, n_rings=6
    )
    bigger_cfg = replace(
        small_demo_cfg, n_transactions=2500 * 8, n_rings=8
    )  # rings barely scale, txns scale 8x

    ds_small = generate_dataset(small_demo_cfg)
    ds_big = generate_dataset(bigger_cfg)

    rate_small = ds_small.ground_truth["abuse_label"].mean()
    rate_big = ds_big.ground_truth["abuse_label"].mean()

    target = small_demo_cfg.target_abuse_rate
    # both should land within a reasonable band of the target rate, not
    # collapse toward zero as transaction count grows faster than ring count
    assert abs(rate_small - target) < 0.06
    assert abs(rate_big - target) < 0.06
