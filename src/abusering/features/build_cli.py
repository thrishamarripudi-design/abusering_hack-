"""CLI: python -m abusering.features.build_cli --data data_store --out data_store"""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd

from abusering.features.build import FEATURE_COLUMNS, FEATURE_VERSION, build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_store")
    ap.add_argument("--out", type=str, default="data_store")
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_parquet(data / "transactions.parquet")
    customers = pd.read_parquet(data / "customers.parquet")
    merchants = pd.read_parquet(data / "merchants.parquet")
    refunds = pd.read_parquet(data / "refunds.parquet")
    chargebacks = pd.read_parquet(data / "chargebacks.parquet")

    print(
        f"[features] building {FEATURE_VERSION} for {len(transactions)} transactions "
        f"(single forward chronological pass)..."
    )
    t0 = time.time()
    feats = build_features(transactions, customers, merchants, refunds, chargebacks)
    dt = time.time() - t0
    print(f"[features] done in {dt:.1f}s -> {len(feats)} rows, {len(FEATURE_COLUMNS)} feature columns")

    feats.to_parquet(out / "features.parquet", index=False)
    print(f"[features] wrote {out / 'features.parquet'}")


if __name__ == "__main__":
    main()
