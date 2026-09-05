"""CLI: python -m abusering.data.generate --seed 42 --profile demo"""

from __future__ import annotations

import argparse
import json
import pathlib

from abusering.data.config import get_config
from abusering.data.generator import generate_dataset

DEFAULT_OUT = pathlib.Path("data_store")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profile", type=str, default="demo", choices=["demo", "full", "stress"])
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    cfg = get_config(args.profile, seed=args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[generate] profile={cfg.profile} seed={cfg.seed} "
        f"n_customers={cfg.n_customers} n_merchants={cfg.n_merchants} "
        f"n_transactions_target={cfg.n_transactions} n_rings={cfg.n_rings}"
    )

    ds = generate_dataset(cfg)

    ds.customers.to_parquet(out / "customers.parquet", index=False)
    ds.merchants.to_parquet(out / "merchants.parquet", index=False)
    ds.transactions.to_parquet(out / "transactions.parquet", index=False)
    ds.refunds.to_parquet(out / "refunds.parquet", index=False)
    ds.chargebacks.to_parquet(out / "chargebacks.parquet", index=False)
    ds.relationships.to_parquet(out / "relationships.parquet", index=False)
    ds.ground_truth.to_parquet(out / "ground_truth.parquet", index=False)
    ds.ring_stats.to_parquet(out / "ring_stats.parquet", index=False)

    abuse_rate = ds.ground_truth["abuse_label"].mean()
    manifest = {
        "dataset_version": cfg.dataset_version(),
        "profile": cfg.profile,
        "seed": cfg.seed,
        "n_customers": len(ds.customers),
        "n_merchants": len(ds.merchants),
        "n_transactions": len(ds.transactions),
        "n_rings": len(ds.ring_stats),
        "abuse_rate": round(float(abuse_rate), 5),
        "n_refunds": len(ds.refunds),
        "n_chargebacks": len(ds.chargebacks),
        "n_relationships": len(ds.relationships),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[generate] wrote {len(ds.transactions)} transactions, "
        f"{len(ds.ring_stats)} rings, abuse_rate={abuse_rate:.4f} -> {out}/"
    )
    print(f"[generate] manifest: {manifest}")


if __name__ == "__main__":
    main()
