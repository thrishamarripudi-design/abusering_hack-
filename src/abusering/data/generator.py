"""Synthetic transactional data generator.

Implements Sec. 5 (legitimate behavior, incl. "looks suspicious but isn't"
populations) and Sec. 6 (six coordinated abuse-ring mechanisms) of the spec.

CRITICAL: the abuse-ring generator internals (ring_id, abuse_type,
generator_type) are written ONLY to `ground_truth.parquet`. They are never
merged into `transactions.parquet` or any feature table. See
`abusering.reports.leakage` for the automated check that enforces this.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from abusering.data.config import GenerationConfig

SEGMENTS = [
    "normal",
    "frequent",
    "high_value",
    "family_shared_device",
    "office_shared_ip",
    "hostel_shared_network",
    "small_business",
]
SEGMENT_WEIGHTS = [0.42, 0.16, 0.08, 0.12, 0.10, 0.07, 0.05]

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "food_delivery", "fashion",
    "gaming", "utilities", "subscriptions", "marketplace", "financial_services",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _date_range_seconds(cfg: GenerationConfig) -> tuple[int, int]:
    start = int(dt.datetime.fromisoformat(cfg.start_date).timestamp())
    end = int(dt.datetime.fromisoformat(cfg.end_date).timestamp())
    return start, end


@dataclass
class GeneratedDataset:
    customers: pd.DataFrame
    merchants: pd.DataFrame
    transactions: pd.DataFrame
    refunds: pd.DataFrame
    chargebacks: pd.DataFrame
    relationships: pd.DataFrame
    ground_truth: pd.DataFrame
    ring_stats: pd.DataFrame


def generate_customers(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    start_ts, end_ts = _date_range_seconds(cfg)
    # accounts can pre-date the observation window by up to ~2 years
    created = rng.integers(start_ts - 730 * 86400, end_ts, size=cfg.n_customers)
    segments = rng.choice(SEGMENTS, size=cfg.n_customers, p=SEGMENT_WEIGHTS)
    countries = rng.choice(
        ["IN", "US", "GB", "AE", "SG"], size=cfg.n_customers, p=[0.55, 0.2, 0.1, 0.1, 0.05]
    )
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:07d}" for i in range(cfg.n_customers)],
            "account_created_at": pd.to_datetime(created, unit="s"),
            "customer_segment": segments,
            "country": countries,
        }
    )
    return df


def generate_merchants(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    start_ts, _ = _date_range_seconds(cfg)
    created = rng.integers(start_ts - 1500 * 86400, start_ts, size=cfg.n_merchants)
    categories = rng.choice(MERCHANT_CATEGORIES, size=cfg.n_merchants)
    risk = rng.choice(["low", "medium", "high"], size=cfg.n_merchants, p=[0.7, 0.24, 0.06])
    return pd.DataFrame(
        {
            "merchant_id": [f"MERCH_{i:06d}" for i in range(cfg.n_merchants)],
            "category": categories,
            "risk_profile": risk,
            "created_at": pd.to_datetime(created, unit="s"),
        }
    )


def _assign_resource_pools(cfg: GenerationConfig, customers: pd.DataFrame, rng: np.random.Generator):
    """Give every customer a personal device/ip/instrument, and additionally
    group some customers into legitimate shared-resource clusters (family,
    office, hostel) so that "shared device/IP" alone is NOT a valid abuse
    signal in this dataset (Sec. 5)."""
    n = len(customers)
    personal_device = np.array([f"DEV_{i:07d}" for i in range(n)])
    personal_ip = np.array([f"IP_{i:07d}" for i in range(n)])
    personal_instrument = np.array([f"PI_{i:07d}" for i in range(n)])

    device_pool = personal_device.copy()
    ip_pool = personal_ip.copy()
    instrument_pool = personal_instrument.copy()

    seg = customers["customer_segment"].to_numpy()

    # Family shared device: groups of 2-5 share ONE device_id (not ip/instrument)
    fam_idx = np.where(seg == "family_shared_device")[0]
    rng.shuffle(fam_idx)
    ptr = 0
    fam_dev_counter = 0
    while ptr < len(fam_idx):
        gsize = int(rng.integers(2, 6))
        group = fam_idx[ptr: ptr + gsize]
        shared_dev = f"DEV_FAM_{fam_dev_counter:05d}"
        device_pool[group] = shared_dev
        fam_dev_counter += 1
        ptr += gsize

    # Office shared IP: groups of 5-30 share ONE ip_id (not device/instrument)
    off_idx = np.where(seg == "office_shared_ip")[0]
    rng.shuffle(off_idx)
    ptr = 0
    off_ip_counter = 0
    while ptr < len(off_idx):
        gsize = int(rng.integers(5, 31))
        group = off_idx[ptr: ptr + gsize]
        shared_ip = f"IP_OFFICE_{off_ip_counter:05d}"
        ip_pool[group] = shared_ip
        off_ip_counter += 1
        ptr += gsize

    # Hostel shared network: groups of 10-50 share ONE ip_id (broader, noisier)
    hostel_idx = np.where(seg == "hostel_shared_network")[0]
    rng.shuffle(hostel_idx)
    ptr = 0
    hostel_ip_counter = 0
    while ptr < len(hostel_idx):
        gsize = int(rng.integers(10, 51))
        group = hostel_idx[ptr: ptr + gsize]
        shared_ip = f"IP_HOSTEL_{hostel_ip_counter:05d}"
        ip_pool[group] = shared_ip
        hostel_ip_counter += 1
        ptr += gsize

    # Some customers legitimately have multiple devices / instruments
    multi_dev_mask = rng.random(n) < 0.12
    multi_pi_mask = rng.random(n) < 0.18

    return {
        "primary_device": device_pool,
        "primary_ip": ip_pool,
        "primary_instrument": instrument_pool,
        "multi_device": multi_dev_mask,
        "multi_instrument": multi_pi_mask,
    }


def _sample_amount(rng: np.random.Generator, segment: str, size: int) -> np.ndarray:
    base = {
        "normal": (6.5, 0.7),
        "frequent": (6.2, 0.6),
        "high_value": (8.0, 0.6),
        "family_shared_device": (6.3, 0.7),
        "office_shared_ip": (6.4, 0.6),
        "hostel_shared_network": (5.8, 0.8),
        "small_business": (7.5, 0.9),
    }[segment]
    mu, sigma = base
    return np.round(np.exp(rng.normal(mu, sigma, size=size)), 2)


def generate_legitimate_transactions(
    cfg: GenerationConfig,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    pools: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    start_ts, end_ts = _date_range_seconds(cfg)
    n_target = int(cfg.n_transactions * (1 - cfg.target_abuse_rate))

    segment_txn_rate = {
        "normal": 1.0,
        "frequent": 2.6,
        "high_value": 1.4,
        "family_shared_device": 1.1,
        "office_shared_ip": 1.0,
        "hostel_shared_network": 0.9,
        "small_business": 3.2,
    }
    weights = customers["customer_segment"].map(segment_txn_rate).to_numpy()
    weights = weights / weights.sum()
    cust_idx = rng.choice(len(customers), size=n_target, p=weights)

    merchant_ids = merchants["merchant_id"].to_numpy()
    txn_merchant = rng.choice(merchant_ids, size=n_target)

    # a transaction can never occur before its own customer's account was
    # created — sample timestamps per-customer with that floor enforced
    cust_created_ts = (customers["account_created_at"].astype("int64") // 10**9).to_numpy()
    low = np.maximum(start_ts, cust_created_ts[cust_idx] + 60)
    low = np.minimum(low, end_ts - 1)  # guard against a customer created after end_ts
    ts = rng.integers(low, end_ts, size=n_target)

    seg_arr = customers["customer_segment"].to_numpy()[cust_idx]
    amounts = np.empty(n_target)
    for seg in SEGMENTS:
        mask = seg_arr == seg
        if mask.any():
            amounts[mask] = _sample_amount(rng, seg, mask.sum())

    customer_ids = customers["customer_id"].to_numpy()[cust_idx]
    device_ids = pools["primary_device"][cust_idx]
    ip_ids = pools["primary_ip"][cust_idx]
    instrument_ids = pools["primary_instrument"][cust_idx]

    # legit multi-device / multi-instrument noise: ~ occasionally use an alt one
    multi_dev = pools["multi_device"][cust_idx]
    alt_device = rng.integers(0, len(customers), size=n_target)
    use_alt_dev = multi_dev & (rng.random(n_target) < 0.3)
    device_ids = np.where(use_alt_dev, pools["primary_device"][alt_device], device_ids)

    multi_pi = pools["multi_instrument"][cust_idx]
    alt_pi = rng.integers(0, len(customers), size=n_target)
    use_alt_pi = multi_pi & (rng.random(n_target) < 0.25)
    instrument_ids = np.where(use_alt_pi, pools["primary_instrument"][alt_pi], instrument_ids)

    status = rng.choice(["success", "failed"], size=n_target, p=[0.96, 0.04])

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:08d}" for i in range(n_target)],
            "customer_id": customer_ids,
            "merchant_id": txn_merchant,
            "timestamp": pd.to_datetime(ts, unit="s"),
            "amount": amounts,
            "payment_instrument_id": instrument_ids,
            "device_id": device_ids,
            "ip_id": ip_ids,
            "status": status,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Abuse ring generators (Sec. 6 A-F). Each returns (transactions_df, meta)
# where meta carries ring_id / abuse_type ONLY for the ground_truth table.
# ---------------------------------------------------------------------------

def _new_ring_customers(cfg, k, rng, offset, burst_window_s=None, start_ts=None, registry: dict | None = None):
    """Synthesize k new customer_ids for a ring, optionally created in a
    tight burst window (Sec 6.E). Creation always precedes start_ts so
    account_age_days is never negative for ring-synthetic accounts. If a
    registry dict is provided, (customer_id -> created_at) is recorded there
    so downstream code reuses this exact timestamp instead of resampling."""
    ids = [f"CUST_RING_{offset}_{j:03d}" for j in range(k)]
    if burst_window_s is not None:
        created = rng.integers(start_ts, start_ts + burst_window_s, size=k)
    else:
        window_start, _ = _date_range_seconds(cfg)
        earliest = window_start - 400 * 86400
        latest = start_ts - 3600 if start_ts is not None else window_start
        latest = max(latest, earliest + 1)
        created = rng.integers(earliest, latest, size=k)
    created_ts = pd.to_datetime(created, unit="s")
    if registry is not None:
        for cid, cts in zip(ids, created_ts):
            registry[cid] = cts
    return ids, created_ts


def _ring_scale(cfg: GenerationConfig) -> float:
    """Ring size (and therefore per-ring transaction volume) scales so the
    realized abuse rate stays close to cfg.target_abuse_rate across
    profiles, instead of using fixed absolute ring sizes that dilute to
    near-zero abuse rate at large `n_transactions` (this was a real bug:
    the `full` profile originally produced 0.94% abuse rate against a 6%
    target because n_rings grew 2.5x while n_transactions grew 12.5x).
    BASELINE_AVG_TXN_PER_RING is calibrated from an unscaled run of the
    `demo` profile (2,162 abuse transactions / 40 rings ≈ 54)."""
    BASELINE_AVG_TXN_PER_RING = 54.05
    target_abuse_txns = cfg.n_transactions * cfg.target_abuse_rate
    baseline_abuse_txns = cfg.n_rings * BASELINE_AVG_TXN_PER_RING
    return max(1.0, target_abuse_txns / baseline_abuse_txns)


def _scaled_range(lo: int, hi: int, scale: float) -> tuple[int, int]:
    return max(lo, int(round(lo * scale))), max(lo + 1, int(round(hi * scale)))


def _gen_shared_device_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    scale = _ring_scale(cfg)
    k_lo, k_hi = _scaled_range(6, 25, scale)
    k = int(rng.integers(k_lo, k_hi))
    cust_ids, created = _new_ring_customers(cfg, k, rng, ring_id, start_ts=start_ts, registry=registry)
    device = f"DEV_RING_{ring_id}"
    n_txn = int(rng.integers(k * 2, k * 6))
    cust_pick = rng.integers(0, k, size=n_txn)
    merch = rng.choice(merchants["merchant_id"].to_numpy(), size=n_txn)
    span = int(rng.integers(3, 30)) * 86400

    # Ensure transactions occur after customer creation
    min_txn_time = int(created.max().timestamp()) + 60
    ts = np.sort(rng.integers(min_txn_time, min_txn_time + span, size=n_txn))

    amt = np.round(np.exp(rng.normal(6.0, 0.4, size=n_txn)), 2)
    ip = rng.choice([f"IP_RING_{ring_id}_A", f"IP_RING_{ring_id}_B"], size=n_txn)
    instrument = np.array([f"PI_{cust_ids[c]}" for c in cust_pick])
    txns = pd.DataFrame({
        "transaction_id": [f"TXNR_{txn_ctr + i:08d}" for i in range(n_txn)],
        "customer_id": [cust_ids[c] for c in cust_pick],
        "merchant_id": merch,
        "timestamp": pd.to_datetime(ts, unit="s"),
        "amount": amt,
        "payment_instrument_id": instrument,
        "device_id": device,
        "ip_id": ip,
        "status": "success",
    })
    return txns, "shared_device"

def _gen_shared_instrument_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    scale = _ring_scale(cfg)
    k_lo, k_hi = _scaled_range(5, 20, scale)
    k = int(rng.integers(k_lo, k_hi))
    cust_ids, created = _new_ring_customers(cfg, k, rng, ring_id, start_ts=start_ts, registry=registry)
    instrument = f"PI_RING_{ring_id}"
    n_txn = int(rng.integers(k * 2, k * 5))
    cust_pick = rng.integers(0, k, size=n_txn)
    merch = rng.choice(merchants["merchant_id"].to_numpy(), size=n_txn)
    span = int(rng.integers(5, 45)) * 86400

    # Ensure transactions occur after customer creation
    min_txn_time = int(created.max().timestamp()) + 60
    ts = np.sort(rng.integers(min_txn_time, min_txn_time + span, size=n_txn))

    amt = np.round(np.exp(rng.normal(5.8, 0.5, size=n_txn)), 2)
    device = np.array([f"DEV_{cust_ids[c]}" for c in cust_pick])
    ip = np.array([f"IP_{cust_ids[c]}" for c in cust_pick])
    txns = pd.DataFrame({
        "transaction_id": [f"TXNR_{txn_ctr + i:08d}" for i in range(n_txn)],
        "customer_id": [cust_ids[c] for c in cust_pick],
        "merchant_id": merch,
        "timestamp": pd.to_datetime(ts, unit="s"),
        "amount": amt,
        "payment_instrument_id": instrument,
        "device_id": device,
        "ip_id": ip,
        "status": "success",
    })
    return txns, "shared_instrument"

def _gen_velocity_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    scale = _ring_scale(cfg)
    k_lo, k_hi = _scaled_range(10, 40, scale)
    k = int(rng.integers(k_lo, k_hi))
    cust_ids, created = _new_ring_customers(cfg, k, rng, ring_id, start_ts=start_ts, registry=registry)
    n_txn = k  # ~1 coordinated txn per account, tight burst

    # Ensure transactions occur after customer creation
    min_txn_time = int(created.max().timestamp()) + 60
    burst_start = min_txn_time + int(rng.integers(0, 60 * 86400))
    window_s = int(rng.integers(60, 900))  # coordinated within minutes
    ts = np.sort(rng.integers(burst_start, burst_start + window_s, size=n_txn))

    merch_choice = rng.choice(merchants["merchant_id"].to_numpy())
    amt = np.round(np.exp(rng.normal(6.2, 0.25, size=n_txn)), 2)  # similar amounts
    device = np.array([f"DEV_{c}" for c in cust_ids])
    ip = np.array([f"IP_{c}" for c in cust_ids])
    instrument = np.array([f"PI_{c}" for c in cust_ids])
    txns = pd.DataFrame({
        "transaction_id": [f"TXNR_{txn_ctr + i:08d}" for i in range(n_txn)],
        "customer_id": cust_ids,
        "merchant_id": merch_choice,
        "timestamp": pd.to_datetime(ts, unit="s"),
        "amount": amt,
        "payment_instrument_id": instrument,
        "device_id": device,
        "ip_id": ip,
        "status": "success",
    })
    return txns, "coordinated_velocity"

def _gen_dispute_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    scale = _ring_scale(cfg)
    k_lo, k_hi = _scaled_range(4, 15, scale)
    k = int(rng.integers(k_lo, k_hi))
    cust_ids, created = _new_ring_customers(cfg, k, rng, ring_id, start_ts=start_ts, registry=registry)
    n_txn = int(rng.integers(k * 3, k * 8))
    cust_pick = rng.integers(0, k, size=n_txn)
    merch = rng.choice(merchants["merchant_id"].to_numpy(), size=n_txn)
    span = int(rng.integers(20, 90)) * 86400

    # Ensure transactions occur after customer creation
    min_txn_time = int(created.max().timestamp()) + 60
    ts = np.sort(rng.integers(min_txn_time, min_txn_time + span, size=n_txn))

    amt = np.round(np.exp(rng.normal(6.8, 0.5, size=n_txn)), 2)
    device = np.array([f"DEV_{cust_ids[c]}" for c in cust_pick])
    ip = np.array([f"IP_{cust_ids[c]}" for c in cust_pick])
    instrument = np.array([f"PI_{cust_ids[c]}" for c in cust_pick])
    txns = pd.DataFrame({
        "transaction_id": [f"TXNR_{txn_ctr + i:08d}" for i in range(n_txn)],
        "customer_id": [cust_ids[c] for c in cust_pick],
        "merchant_id": merch,
        "timestamp": pd.to_datetime(ts, unit="s"),
        "amount": amt,
        "payment_instrument_id": instrument,
        "device_id": device,
        "ip_id": ip,
        "status": "success",
    })
    return txns, "dispute_abuse"

def _gen_creation_burst_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    scale = _ring_scale(cfg)
    k_lo, k_hi = _scaled_range(8, 30, scale)
    k = int(rng.integers(k_lo, k_hi))
    burst_window = int(rng.integers(600, 3600))  # accounts created within ~1hr
    cust_ids, created = _new_ring_customers(cfg, k, rng, ring_id, burst_window_s=burst_window, start_ts=start_ts, registry=registry)
    n_txn = int(rng.integers(k, k * 3))
    cust_pick = rng.integers(0, k, size=n_txn)
    merch = rng.choice(merchants["merchant_id"].to_numpy(), size=n_txn)
    # transact soon after creation (coordinated onboarding-to-cashout pattern)
    delay = rng.integers(3600, 5 * 86400, size=n_txn)
    ts = created.astype("int64").to_numpy()[cust_pick] // 10**9 + delay
    amt = np.round(np.exp(rng.normal(6.5, 0.4, size=n_txn)), 2)
    device = rng.choice([f"DEV_RING_{ring_id}_A", f"DEV_RING_{ring_id}_B", f"DEV_RING_{ring_id}_C"], size=n_txn)
    ip = np.array([f"IP_{cust_ids[c]}" for c in cust_pick])
    instrument = np.array([f"PI_{cust_ids[c]}" for c in cust_pick])
    txns = pd.DataFrame({
        "transaction_id": [f"TXNR_{txn_ctr + i:08d}" for i in range(n_txn)],
        "customer_id": [cust_ids[c] for c in cust_pick],
        "merchant_id": merch,
        "timestamp": pd.to_datetime(ts, unit="s"),
        "amount": amt,
        "payment_instrument_id": instrument,
        "device_id": device,
        "ip_id": ip,
        "status": "success",
    })
    return txns, "coordinated_creation"


def _gen_hybrid_ring(cfg, ring_id, merchants, rng, txn_ctr, start_ts, registry=None):
    parts = []
    generators = [_gen_shared_device_ring, _gen_shared_instrument_ring, _gen_velocity_ring]
    chosen = rng.choice(len(generators), size=2, replace=False)
    ctr = txn_ctr
    for sub_i, gi in enumerate(chosen):
        # distinct offset per sub-mechanism avoids customer_id collisions
        # between the two mechanisms combined into this hybrid ring
        sub_offset = f"{ring_id}h{sub_i}"
        sub_txns, _ = generators[gi](cfg, sub_offset, merchants, rng, ctr, start_ts, registry=registry)
        ctr += len(sub_txns) + 1
        parts.append(sub_txns)
    txns = pd.concat(parts, ignore_index=True)
    txns["transaction_id"] = [f"TXNR_{txn_ctr + i:08d}" for i in range(len(txns))]
    return txns, "hybrid"


RING_GENERATORS = {
    "shared_device": _gen_shared_device_ring,
    "shared_instrument": _gen_shared_instrument_ring,
    "coordinated_velocity": _gen_velocity_ring,
    "dispute_abuse": _gen_dispute_ring,
    "coordinated_creation": _gen_creation_burst_ring,
    "hybrid": _gen_hybrid_ring,
}


def generate_abuse_rings(
    cfg: GenerationConfig, merchants: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    start_ts, end_ts = _date_range_seconds(cfg)
    ring_types = list(cfg.ring_type_weights.keys())
    ring_probs = list(cfg.ring_type_weights.values())
    chosen_types = rng.choice(ring_types, size=cfg.n_rings, p=ring_probs)

    registry: dict = {}
    all_txns = []
    ring_rows = []
    txn_ctr = 0
    for i, ring_type in enumerate(chosen_types):
        ring_id = f"RING_{i:03d}"
        ring_start = int(rng.integers(start_ts, end_ts - 5 * 86400))
        gen_fn = RING_GENERATORS[ring_type]
        txns, abuse_type = gen_fn(cfg, i, merchants, rng, txn_ctr, ring_start, registry=registry)
        txns["ring_id"] = ring_id
        txns["abuse_type"] = abuse_type
        all_txns.append(txns)
        txn_ctr += len(txns) + 1
        ring_rows.append({
            "ring_id": ring_id,
            "abuse_type": abuse_type,
            "ring_size_accounts": txns["customer_id"].nunique(),
            "n_transactions": len(txns),
            "amount_at_risk": round(float(txns["amount"].sum()), 2),
            "period_start": txns["timestamp"].min(),
            "period_end": txns["timestamp"].max(),
        })

    ring_txns = pd.concat(all_txns, ignore_index=True) if all_txns else pd.DataFrame()
    ring_stats = pd.DataFrame(ring_rows)
    return ring_txns, ring_stats, registry


def _generate_refunds_chargebacks(
    transactions: pd.DataFrame, ground_truth: pd.DataFrame, rng: np.random.Generator
):
    merged = transactions.merge(ground_truth[["transaction_id", "abuse_label"]], on="transaction_id")

    # legitimate refund/chargeback rates
    legit = merged[merged.abuse_label == 0]
    refund_mask_legit = rng.random(len(legit)) < 0.03
    cb_mask_legit = rng.random(len(legit)) < 0.006

    # dispute-abuse ring txns get elevated refund/chargeback rates (Sec 6.D)
    abuse = merged[merged.abuse_label == 1]
    refund_mask_abuse = rng.random(len(abuse)) < 0.35
    cb_mask_abuse = rng.random(len(abuse)) < 0.22

    def _build(df, mask, prefix, reasons):
        sub = df[mask]
        if len(sub) == 0:
            return pd.DataFrame(columns=[f"{prefix}_id", "transaction_id", "timestamp", "amount", "reason"])
        delay = rng.integers(3600, 20 * 86400, size=len(sub))
        ts = sub["timestamp"].astype("int64").to_numpy() // 10**9 + delay
        reason = rng.choice(reasons, size=len(sub))
        amt = sub["amount"].to_numpy() * rng.uniform(0.5, 1.0, size=len(sub))
        return pd.DataFrame({
            f"{prefix}_id": [f"{prefix.upper()}_{i:08d}" for i in range(len(sub))],
            "transaction_id": sub["transaction_id"].to_numpy(),
            "timestamp": pd.to_datetime(ts, unit="s"),
            "amount": np.round(amt, 2),
            "reason": reason,
        })

    refunds = pd.concat([
        _build(legit, refund_mask_legit, "refund", ["not_as_described", "changed_mind", "duplicate", "damaged"]),
        _build(abuse, refund_mask_abuse, "refund", ["not_received", "unauthorized", "not_as_described"]),
    ], ignore_index=True)
    chargebacks = pd.concat([
        _build(legit, cb_mask_legit, "chargeback", ["unauthorized", "service_not_provided"]),
        _build(abuse, cb_mask_abuse, "chargeback", ["unauthorized", "friendly_fraud", "service_not_provided"]),
    ], ignore_index=True)

    refunds["refund_id"] = [f"REFUND_{i:08d}" for i in range(len(refunds))]
    chargebacks["chargeback_id"] = [f"CHARGEBACK_{i:08d}" for i in range(len(chargebacks))]
    return refunds, chargebacks


def _build_relationships(transactions: pd.DataFrame) -> pd.DataFrame:
    edges = []
    for target_col, target_type in [
        ("device_id", "device"), ("ip_id", "ip"),
        ("payment_instrument_id", "instrument"), ("merchant_id", "merchant"),
    ]:
        g = transactions.groupby(["customer_id", target_col])["timestamp"].min().reset_index()
        g = g.rename(columns={target_col: "target_id", "timestamp": "created_at"})
        g["source_id"] = g["customer_id"]
        g["source_type"] = "customer"
        g["target_type"] = target_type
        g["relationship_type"] = f"customer_{target_type}"
        edges.append(g[["source_id", "source_type", "target_id", "target_type", "relationship_type", "created_at"]])
    return pd.concat(edges, ignore_index=True)


def generate_dataset(cfg: GenerationConfig) -> GeneratedDataset:
    rng = _rng(cfg.seed)

    customers = generate_customers(cfg, rng)
    merchants = generate_merchants(cfg, rng)
    pools = _assign_resource_pools(cfg, customers, rng)
    legit_txns = generate_legitimate_transactions(cfg, customers, merchants, pools, rng)

    ring_txns, ring_stats, ring_cust_registry = generate_abuse_rings(cfg, merchants, rng)

    # ring-only customers must also appear in the customers table, using the
    # EXACT created_at timestamps the ring generators already computed
    # (never re-sampled independently, or account_age could go negative)
    ring_cust_ids = set(ring_txns["customer_id"].unique()) if len(ring_txns) else set()
    existing = set(customers["customer_id"])
    new_ring_custs = ring_cust_ids - existing
    if new_ring_custs:
        ids = list(new_ring_custs)
        created = [ring_cust_registry[c] for c in ids]
        extra = pd.DataFrame({
            "customer_id": ids,
            "account_created_at": pd.to_datetime(created),
            "customer_segment": "ring_synthetic",
            "country": rng.choice(["IN", "US", "GB", "AE", "SG"], size=len(ids)),
        })
        customers = pd.concat([customers, extra], ignore_index=True)

    ground_truth_abuse = ring_txns[["transaction_id", "ring_id", "abuse_type"]].copy()
    ground_truth_abuse["abuse_label"] = 1

    ring_txns_clean = ring_txns.drop(columns=["ring_id", "abuse_type"])
    transactions = pd.concat([legit_txns, ring_txns_clean], ignore_index=True)
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    ground_truth_legit = pd.DataFrame({
        "transaction_id": legit_txns["transaction_id"],
        "ring_id": None,
        "abuse_type": None,
        "abuse_label": 0,
    })
    ground_truth = pd.concat([ground_truth_legit, ground_truth_abuse], ignore_index=True)
    # dedupe safety: keep the abuse label if a txn_id collided (shouldn't happen given prefixes)
    ground_truth = ground_truth.drop_duplicates(subset="transaction_id", keep="last")

    refunds, chargebacks = _generate_refunds_chargebacks(transactions, ground_truth, rng)
    relationships = _build_relationships(transactions)

    return GeneratedDataset(
        customers=customers,
        merchants=merchants,
        transactions=transactions,
        refunds=refunds,
        chargebacks=chargebacks,
        relationships=relationships,
        ground_truth=ground_truth,
        ring_stats=ring_stats,
    )
