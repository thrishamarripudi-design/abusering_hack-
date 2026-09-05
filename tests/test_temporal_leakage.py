import pandas as pd

from abusering.features.build import FEATURE_COLUMNS, build_features


def _tiny_transactions():
    base = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "customer_id": ["C1", "C1", "C1"],
            "merchant_id": ["M1", "M1", "M1"],
            "timestamp": [base, base + pd.Timedelta(hours=2), base + pd.Timedelta(days=5)],
            "amount": [100.0, 120.0, 90.0],
            "payment_instrument_id": ["PI1", "PI1", "PI1"],
            "device_id": ["D1", "D1", "D1"],
            "ip_id": ["IP1", "IP1", "IP1"],
            "status": ["success", "success", "success"],
        }
    )


def _customers():
    return pd.DataFrame(
        {
            "customer_id": ["C1"],
            "account_created_at": [pd.Timestamp("2024-01-01")],
            "customer_segment": ["normal"],
            "country": ["IN"],
        }
    )


def _merchants():
    return pd.DataFrame(
        {
            "merchant_id": ["M1"],
            "category": ["grocery"],
            "risk_profile": ["low"],
            "created_at": [pd.Timestamp("2023-01-01")],
        }
    )


def test_future_chargeback_does_not_leak_into_past_transaction():
    txns = _tiny_transactions()
    customers = _customers()
    merchants = _merchants()

    # a chargeback on T1, filed 10 days AFTER T1 (i.e. also after T2, but
    # arguably before T3's prediction time... make it clearly after all three)
    late_ts = txns["timestamp"].max() + pd.Timedelta(days=10)
    chargebacks = pd.DataFrame(
        {
            "chargeback_id": ["CB1"],
            "transaction_id": ["T1"],
            "timestamp": [late_ts],
            "amount": [100.0],
            "reason": ["unauthorized"],
        }
    )
    refunds = pd.DataFrame(columns=["refund_id", "transaction_id", "timestamp", "amount", "reason"])

    feats = build_features(txns, customers, merchants, refunds, chargebacks)
    # None of T1/T2/T3's chargeback_rate_30d should reflect CB1, since CB1's
    # own timestamp is after every transaction's prediction_time.
    assert (feats["chargeback_rate_30d"] == 0).all(), (
        "a chargeback filed after all transactions leaked into chargeback_rate_30d"
    )


def test_future_refund_does_not_leak_into_earlier_transaction():
    txns = _tiny_transactions()
    customers = _customers()
    merchants = _merchants()

    # refund on T2, timestamped AFTER T3 (should not affect T1 or T2's own features,
    # since T2's features are computed at T2's own prediction_time, before the refund exists)
    late_ts = txns["timestamp"].max() + pd.Timedelta(days=1)
    refunds = pd.DataFrame(
        {
            "refund_id": ["R1"],
            "transaction_id": ["T2"],
            "timestamp": [late_ts],
            "amount": [120.0],
            "reason": ["not_as_described"],
        }
    )
    chargebacks = pd.DataFrame(columns=["chargeback_id", "transaction_id", "timestamp", "amount", "reason"])

    feats = build_features(txns, customers, merchants, refunds, chargebacks)
    t1_row = feats[feats.transaction_id == "T1"].iloc[0]
    t2_row = feats[feats.transaction_id == "T2"].iloc[0]
    assert t1_row["refund_rate_30d"] == 0
    assert t2_row["refund_rate_30d"] == 0  # the refund postdates T2 itself


def test_future_device_relationship_does_not_affect_earlier_transaction():
    """A device shared by a second customer AFTER T1's timestamp must not
    inflate T1's shared_device_count."""
    txns = _tiny_transactions()
    future_customer_txn = pd.DataFrame(
        {
            "transaction_id": ["T4"],
            "customer_id": ["C2"],
            "merchant_id": ["M1"],
            "timestamp": [txns["timestamp"].max() + pd.Timedelta(days=1)],  # after all of C1's txns
            "amount": [50.0],
            "payment_instrument_id": ["PI2"],
            "device_id": ["D1"],  # shares C1's device, but only in the future
            "ip_id": ["IP2"],
            "status": ["success"],
        }
    )
    all_txns = pd.concat([txns, future_customer_txn], ignore_index=True)
    customers = pd.concat(
        [
            _customers(),
            pd.DataFrame(
                {
                    "customer_id": ["C2"],
                    "account_created_at": [pd.Timestamp("2024-06-01")],
                    "customer_segment": ["normal"],
                    "country": ["IN"],
                }
            ),
        ],
        ignore_index=True,
    )
    merchants = _merchants()
    refunds = pd.DataFrame(columns=["refund_id", "transaction_id", "timestamp", "amount", "reason"])
    chargebacks = pd.DataFrame(columns=["chargeback_id", "transaction_id", "timestamp", "amount", "reason"])

    feats = build_features(all_txns, customers, merchants, refunds, chargebacks)
    t1 = feats[feats.transaction_id == "T1"].iloc[0]
    t3 = feats[feats.transaction_id == "T3"].iloc[0]  # also before C2's txn
    # shared_device_count = distinct customers who've used this device so far.
    # T1 is C1's first-ever transaction: no one (not even C1) has used D1 yet.
    assert t1["shared_device_count"] == 0
    # T3 is C1's own third transaction: only C1 itself has used D1 so far
    # (from T1/T2) — C2's future use of D1 must NOT be counted here.
    assert t3["shared_device_count"] == 1


def test_feature_columns_present():
    txns = _tiny_transactions()
    feats = build_features(
        txns,
        _customers(),
        _merchants(),
        pd.DataFrame(columns=["refund_id", "transaction_id", "timestamp", "amount", "reason"]),
        pd.DataFrame(columns=["chargeback_id", "transaction_id", "timestamp", "amount", "reason"]),
    )
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing declared feature column {col}"
