"""Leakage-safe temporal feature store.

Design: rather than computing features with joins/rolling windows that are
easy to get subtly wrong (off-by-one future leaks), we do a SINGLE forward
chronological pass over transactions, maintaining per-entity rolling state
(deques of recent events). A transaction's features are computed from state
that reflects ONLY events with event_timestamp < this transaction's
timestamp, and the transaction is applied to the state AFTER its features
are read. This makes "no future information" a structural property of the
algorithm rather than a hope.

Ground-truth columns (ring_id, abuse_type, abuse_label, generator_type) are
never read by this module — it only ever touches transactions/refunds/
chargebacks/customers/merchants/relationships-derivable data.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd

FEATURE_VERSION = "features-v1"

WINDOWS = {
    "1h": 3600,
    "24h": 86400,
    "7d": 7 * 86400,
    "30d": 30 * 86400,
}


class _Windowed:
    """Per-key deque of (timestamp, value) with expiry-based windowed stats."""

    def __init__(self):
        self.data: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    def push(self, key, ts, value):
        self.data[key].append((ts, value))

    def prune(self, key, ts, window_s):
        dq = self.data[key]
        cutoff = ts - window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def count(self, key, ts, window_s):
        self.prune(key, ts, window_s)
        return len(self.data[key])

    def values(self, key, ts, window_s):
        self.prune(key, ts, window_s)
        return [v for _, v in self.data[key]]


def build_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    refunds: pd.DataFrame,
    chargebacks: pd.DataFrame,
) -> pd.DataFrame:
    txns = transactions.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    cust_created = customers.set_index("customer_id")["account_created_at"].to_dict()

    # Precompute refund/chargeback timestamps per transaction_id for O(1) lookup during the pass.
    refunds_by_txn = refunds.groupby("transaction_id")["timestamp"].apply(list).to_dict() if len(refunds) else {}
    cbs_by_txn = chargebacks.groupby("transaction_id")["timestamp"].apply(list).to_dict() if len(chargebacks) else {}

    # Rolling windowed state, keyed by entity id.
    cust_txn_times = _Windowed()          # customer -> [(ts, amount)]
    cust_merchants = _Windowed()          # customer -> [(ts, merchant_id)]
    cust_devices = _Windowed()            # customer -> [(ts, device_id)]
    cust_ips = _Windowed()                # customer -> [(ts, ip_id)]
    device_customers = _Windowed()        # device -> [(ts, customer_id)]
    ip_customers = _Windowed()            # ip -> [(ts, customer_id)]
    instrument_devices = _Windowed()      # instrument -> [(ts, device_id)]
    device_txn_times = _Windowed()        # device -> [(ts, 1)]
    ip_txn_times = _Windowed()            # ip -> [(ts, 1)]
    instrument_txn_times = _Windowed()    # instrument -> [(ts, amount)]
    merchant_txn_times = _Windowed()      # merchant -> [(ts, 1)]

    # Cumulative (all-time, as-of) sets/counters for degree-style graph features.
    cum_device_customers: dict[str, set] = collections.defaultdict(set)
    cum_ip_customers: dict[str, set] = collections.defaultdict(set)
    cum_instrument_customers: dict[str, set] = collections.defaultdict(set)
    cum_customer_devices: dict[str, set] = collections.defaultdict(set)
    cum_customer_ips: dict[str, set] = collections.defaultdict(set)
    cum_customer_instruments: dict[str, set] = collections.defaultdict(set)
    cum_instrument_txn_count: dict[str, int] = collections.defaultdict(int)
    cum_merchant_txn_count: dict[str, int] = collections.defaultdict(int)

    # Refund/chargeback ledgers, applied lazily via a pointer scan since they
    # arrive out of order relative to their parent transaction's row position.
    # We instead check them at read-time using timestamps already known at
    # generation (all refund/chargeback timestamps are AFTER their txn), so
    # we build a per-customer sorted list of (event_ts, kind) up front and
    # binary-search the prefix strictly before the current prediction time.
    disputes_by_customer: dict[str, list] = collections.defaultdict(list)
    disputes_by_instrument: dict[str, list] = collections.defaultdict(list)
    txn_to_customer = txns.set_index("transaction_id")["customer_id"].to_dict()
    txn_to_instrument = txns.set_index("transaction_id")["payment_instrument_id"].to_dict()
    for txn_id, ts_list in refunds_by_txn.items():
        c = txn_to_customer.get(txn_id)
        instr_ = txn_to_instrument.get(txn_id)
        for ts in ts_list:
            if c is not None:
                disputes_by_customer[c].append((pd.Timestamp(ts).value, "refund"))
            if instr_ is not None:
                disputes_by_instrument[instr_].append((pd.Timestamp(ts).value, "refund"))
    for txn_id, ts_list in cbs_by_txn.items():
        c = txn_to_customer.get(txn_id)
        instr_ = txn_to_instrument.get(txn_id)
        for ts in ts_list:
            if c is not None:
                disputes_by_customer[c].append((pd.Timestamp(ts).value, "chargeback"))
            if instr_ is not None:
                disputes_by_instrument[instr_].append((pd.Timestamp(ts).value, "chargeback"))
    for c in disputes_by_customer:
        disputes_by_customer[c].sort()
    for instr_ in disputes_by_instrument:
        disputes_by_instrument[instr_].sort()

    import bisect

    rows = []
    ts_ns = txns["timestamp"].values.astype("datetime64[ns]").astype("int64")
    ts_s = ts_ns // 10**9

    for i in range(len(txns)):
        r = txns.iloc[i]
        t = int(ts_s[i])
        cust = r["customer_id"]
        merch = r["merchant_id"]
        dev = r["device_id"]
        ip = r["ip_id"]
        instr = r["payment_instrument_id"]
        amount = float(r["amount"])

        # ---- customer historical (strictly before t; state not yet updated) ----
        txn_count_1h = cust_txn_times.count(cust, t, WINDOWS["1h"])
        txn_count_24h = cust_txn_times.count(cust, t, WINDOWS["24h"])
        txn_count_7d = cust_txn_times.count(cust, t, WINDOWS["7d"])
        amounts_7d = cust_txn_times.values(cust, t, WINDOWS["7d"])
        avg_amount_7d = float(np.mean(amounts_7d)) if amounts_7d else 0.0
        amount_std_7d = float(np.std(amounts_7d)) if len(amounts_7d) > 1 else 0.0

        merchants_30d = cust_merchants.values(cust, t, WINDOWS["30d"])
        unique_merchants_30d = len(set(merchants_30d))
        devices_30d = cust_devices.values(cust, t, WINDOWS["30d"])
        unique_devices_30d = len(set(devices_30d))
        ips_30d = cust_ips.values(cust, t, WINDOWS["30d"])
        unique_ips_30d = len(set(ips_30d))

        created_at = cust_created.get(cust)
        account_age_days = (
            (pd.Timestamp(t, unit="s") - pd.Timestamp(created_at)).total_seconds() / 86400.0
            if created_at is not None else -1.0
        )

        # dispute rates strictly before t (30d window), using ONLY dispute
        # events whose own timestamp is < t (refund/chargeback of a PAST txn)
        cutoff_ns = t * 10**9
        window_ns = WINDOWS["30d"] * 10**9
        dlist = disputes_by_customer.get(cust, [])
        lo = bisect.bisect_left(dlist, (cutoff_ns - window_ns, ""))
        hi = bisect.bisect_left(dlist, (cutoff_ns, ""))
        recent_disputes = dlist[lo:hi]
        n_refund_30d = sum(1 for _, k in recent_disputes if k == "refund")
        n_cb_30d = sum(1 for _, k in recent_disputes if k == "chargeback")
        denom = max(txn_count_7d, 1)  # proxy denominator: recent activity level
        refund_rate_30d = n_refund_30d / denom
        chargeback_rate_30d = n_cb_30d / denom

        # ---- device / ip / instrument velocity + shared-resource counts ----
        shared_device_count = len(cum_device_customers[dev])
        shared_ip_count = len(cum_ip_customers[ip])
        shared_instrument_count = len(cum_instrument_customers[instr])

        device_unique_customers_1d = len(set(device_customers.values(dev, t, WINDOWS["24h"])))
        device_unique_customers_7d = len(set(device_customers.values(dev, t, WINDOWS["7d"])))
        device_txn_24h = device_txn_times.count(dev, t, WINDOWS["24h"])
        device_txn_1h = device_txn_times.count(dev, t, WINDOWS["1h"])

        ip_unique_customers_1h = len(set(ip_customers.values(ip, t, WINDOWS["1h"])))
        ip_unique_customers_24h = len(set(ip_customers.values(ip, t, WINDOWS["24h"])))
        ip_unique_accounts_7d = len(set(ip_customers.values(ip, t, WINDOWS["7d"])))
        ip_txn_velocity_1h = ip_txn_times.count(ip, t, WINDOWS["1h"])

        instrument_unique_customers = len(cum_instrument_customers[instr])
        instrument_unique_devices = len(set(instrument_devices.values(instr, t, 10 ** 12)))
        instrument_txn_count = cum_instrument_txn_count[instr]
        ilist = disputes_by_instrument.get(instr, [])
        ihi = bisect.bisect_left(ilist, (cutoff_ns, ""))
        n_cb_instr = sum(1 for _, k in ilist[:ihi] if k == "chargeback")
        instrument_cb_rate = n_cb_instr / instrument_txn_count if instrument_txn_count > 0 else 0.0

        # ---- coordination signals ----
        window_amounts = instrument_txn_times.values(instr, t, WINDOWS["24h"]) + \
            device_txn_times.values(dev, t, WINDOWS["24h"])
        if len(window_amounts) >= 2:
            amount_similarity = 1.0 / (1.0 + float(np.std(window_amounts)))
        else:
            amount_similarity = 0.0
        transactions_in_time_window = device_txn_times.count(dev, t, 900) + ip_txn_times.count(ip, t, 900)

        # accounts created near this customer's creation, among others who
        # share this customer's device/ip (uses only creation timestamps of
        # customers already observed transacting before t on the same
        # resource — i.e. no future account lookups)
        neighbor_customers = cum_device_customers[dev] | cum_ip_customers[ip]
        accounts_created_nearby = 0
        if created_at is not None:
            for nc in neighbor_customers:
                nc_created = cust_created.get(nc)
                if nc_created is not None and abs((nc_created - created_at).total_seconds()) <= 3 * 86400:
                    accounts_created_nearby += 1

        # ---- graph-ish features ----
        customer_device_degree = len(cum_customer_devices[cust])
        customer_ip_degree = len(cum_customer_ips[cust])
        customer_instrument_degree = len(cum_customer_instruments[cust])
        two_hop_neighbors = len(neighbor_customers)
        # crude community-size proxy: size of the union of customer sets
        # across this txn's device/ip/instrument (shared-resource cluster)
        community_size = len(cum_device_customers[dev] | cum_ip_customers[ip] | cum_instrument_customers[instr])

        merchant_freq = cum_merchant_txn_count[merch]

        ts_pd = pd.Timestamp(t, unit="s")
        rows.append({
            "transaction_id": r["transaction_id"],
            "customer_id": cust,
            "prediction_time": r["timestamp"],
            "amount": amount,
            "log_amount": float(np.log1p(amount)),
            "hour": ts_pd.hour,
            "day_of_week": ts_pd.dayofweek,
            "weekend": int(ts_pd.dayofweek >= 5),
            "merchant_frequency": merchant_freq,
            "transaction_count_1h": txn_count_1h,
            "transaction_count_24h": txn_count_24h,
            "transaction_count_7d": txn_count_7d,
            "average_amount_7d": avg_amount_7d,
            "amount_std_7d": amount_std_7d,
            "refund_rate_30d": refund_rate_30d,
            "chargeback_rate_30d": chargeback_rate_30d,
            "unique_merchants_30d": unique_merchants_30d,
            "unique_devices_30d": unique_devices_30d,
            "unique_ips_30d": unique_ips_30d,
            "account_age_days": account_age_days,
            "device_unique_customers_1d": device_unique_customers_1d,
            "device_unique_customers_7d": device_unique_customers_7d,
            "device_transactions_24h": device_txn_24h,
            "device_transactions_1h": device_txn_1h,
            "device_customer_degree": shared_device_count,
            "ip_unique_customers_1h": ip_unique_customers_1h,
            "ip_unique_customers_24h": ip_unique_customers_24h,
            "ip_unique_accounts_7d": ip_unique_accounts_7d,
            "ip_transaction_velocity_1h": ip_txn_velocity_1h,
            "instrument_unique_customers": instrument_unique_customers,
            "instrument_unique_devices": instrument_unique_devices,
            "instrument_transaction_count": instrument_txn_count,
            "instrument_historical_chargeback_rate": instrument_cb_rate,
            "accounts_created_nearby": accounts_created_nearby,
            "transactions_in_time_window_15m": transactions_in_time_window,
            "amount_similarity": amount_similarity,
            "shared_device_count": shared_device_count,
            "shared_ip_count": shared_ip_count,
            "shared_instrument_count": shared_instrument_count,
            "customer_device_degree": customer_device_degree,
            "customer_ip_degree": customer_ip_degree,
            "customer_instrument_degree": customer_instrument_degree,
            "two_hop_neighbor_count": two_hop_neighbors,
            "community_size": community_size,
        })

        # ---- NOW apply this transaction to state (future rows will see it) ----
        cust_txn_times.push(cust, t, amount)
        cust_merchants.push(cust, t, merch)
        cust_devices.push(cust, t, dev)
        cust_ips.push(cust, t, ip)
        device_customers.push(dev, t, cust)
        ip_customers.push(ip, t, cust)
        instrument_devices.push(instr, t, dev)
        device_txn_times.push(dev, t, 1)
        ip_txn_times.push(ip, t, 1)
        instrument_txn_times.push(instr, t, amount)
        merchant_txn_times.push(merch, t, 1)

        cum_device_customers[dev].add(cust)
        cum_ip_customers[ip].add(cust)
        cum_instrument_customers[instr].add(cust)
        cum_customer_devices[cust].add(dev)
        cum_customer_ips[cust].add(ip)
        cum_customer_instruments[cust].add(instr)
        cum_instrument_txn_count[instr] += 1
        cum_merchant_txn_count[merch] += 1

    feats = pd.DataFrame(rows)
    feats["feature_version"] = FEATURE_VERSION
    return feats


FEATURE_COLUMNS = [
    "amount", "log_amount", "hour", "day_of_week", "weekend", "merchant_frequency",
    "transaction_count_1h", "transaction_count_24h", "transaction_count_7d",
    "average_amount_7d", "amount_std_7d", "refund_rate_30d", "chargeback_rate_30d",
    "unique_merchants_30d", "unique_devices_30d", "unique_ips_30d", "account_age_days",
    "device_unique_customers_1d", "device_unique_customers_7d", "device_transactions_24h",
    "device_transactions_1h", "device_customer_degree",
    "ip_unique_customers_1h", "ip_unique_customers_24h", "ip_unique_accounts_7d",
    "ip_transaction_velocity_1h",
    "instrument_unique_customers", "instrument_unique_devices", "instrument_transaction_count",
    "instrument_historical_chargeback_rate",
    "accounts_created_nearby", "transactions_in_time_window_15m", "amount_similarity",
    "shared_device_count", "shared_ip_count", "shared_instrument_count",
    "customer_device_degree", "customer_ip_degree", "customer_instrument_degree",
    "two_hop_neighbor_count", "community_size",
]

# Subset considered "graph/coordination" features, used for ablation experiments (Sec 19).
GRAPH_FEATURE_COLUMNS = [
    "device_unique_customers_1d", "device_unique_customers_7d", "device_customer_degree",
    "ip_unique_customers_1h", "ip_unique_customers_24h", "ip_unique_accounts_7d",
    "instrument_unique_customers", "instrument_unique_devices",
    "accounts_created_nearby", "transactions_in_time_window_15m", "amount_similarity",
    "shared_device_count", "shared_ip_count", "shared_instrument_count",
    "customer_device_degree", "customer_ip_degree", "customer_instrument_degree",
    "two_hop_neighbor_count", "community_size",
]
BEHAVIORAL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in GRAPH_FEATURE_COLUMNS]
VELOCITY_FEATURE_COLUMNS = [
    "transaction_count_1h", "transaction_count_24h", "transaction_count_7d",
    "device_transactions_1h", "device_transactions_24h", "ip_transaction_velocity_1h",
    "transactions_in_time_window_15m",
]
DISPUTE_FEATURE_COLUMNS = [
    "refund_rate_30d", "chargeback_rate_30d", "instrument_historical_chargeback_rate",
]
