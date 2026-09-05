"""Split by abuse ring, not by transaction (Sec 10).

No ring may appear in more than one split. Legitimate transactions are
independently, randomly partitioned across the same three splits so the
class balance is preserved everywhere.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def make_splits(ground_truth: pd.DataFrame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    ring_ids = sorted(ground_truth.loc[ground_truth.abuse_label == 1, "ring_id"].dropna().unique().tolist())
    rng.shuffle(ring_ids)
    n = len(ring_ids)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["validation"]))
    train_rings = ring_ids[:n_train]
    val_rings = ring_ids[n_train:n_train + n_val]
    test_rings = ring_ids[n_train + n_val:]

    assert set(train_rings) & set(val_rings) == set()
    assert set(train_rings) & set(test_rings) == set()
    assert set(val_rings) & set(test_rings) == set()

    legit_txn_ids = ground_truth.loc[ground_truth.abuse_label == 0, "transaction_id"].to_numpy()
    rng.shuffle(legit_txn_ids)
    n_legit = len(legit_txn_ids)
    n_legit_train = int(round(n_legit * SPLIT_RATIOS["train"]))
    n_legit_val = int(round(n_legit * SPLIT_RATIOS["validation"]))
    legit_train = legit_txn_ids[:n_legit_train]
    legit_val = legit_txn_ids[n_legit_train:n_legit_train + n_legit_val]
    legit_test = legit_txn_ids[n_legit_train + n_legit_val:]

    train_abuse_txns = ground_truth.loc[ground_truth.ring_id.isin(train_rings), "transaction_id"].to_numpy()
    val_abuse_txns = ground_truth.loc[ground_truth.ring_id.isin(val_rings), "transaction_id"].to_numpy()
    test_abuse_txns = ground_truth.loc[ground_truth.ring_id.isin(test_rings), "transaction_id"].to_numpy()

    train_txns = np.concatenate([train_abuse_txns, legit_train])
    val_txns = np.concatenate([val_abuse_txns, legit_val])
    test_txns = np.concatenate([test_abuse_txns, legit_test])

    manifest = {
        "seed": seed,
        "train_ring_ids": train_rings,
        "validation_ring_ids": val_rings,
        "test_ring_ids": test_rings,
        "train_transaction_count": int(len(train_txns)),
        "validation_transaction_count": int(len(val_txns)),
        "test_transaction_count": int(len(test_txns)),
        "train_transaction_ids": train_txns.tolist(),
        "validation_transaction_ids": val_txns.tolist(),
        "test_transaction_ids": test_txns.tolist(),
    }
    return manifest


def verify_no_ring_overlap(manifest: dict) -> None:
    tr, va, te = set(manifest["train_ring_ids"]), set(manifest["validation_ring_ids"]), set(manifest["test_ring_ids"])
    assert not (tr & va), "train/validation ring overlap detected"
    assert not (tr & te), "train/test ring overlap detected"
    assert not (va & te), "validation/test ring overlap detected"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_store")
    ap.add_argument("--out", type=str, default="data_store")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    out = pathlib.Path(args.out)
    ground_truth = pd.read_parquet(data / "ground_truth.parquet")

    manifest = make_splits(ground_truth, seed=args.seed)
    verify_no_ring_overlap(manifest)

    with open(out / "splits.json", "w") as f:
        json.dump(manifest, f)

    print(f"[splits] train_rings={len(manifest['train_ring_ids'])} "
          f"val_rings={len(manifest['validation_ring_ids'])} "
          f"test_rings={len(manifest['test_ring_ids'])}")
    print(f"[splits] train_txns={manifest['train_transaction_count']} "
          f"val_txns={manifest['validation_transaction_count']} "
          f"test_txns={manifest['test_transaction_count']}")
    print("[splits] zero ring overlap verified OK")


if __name__ == "__main__":
    main()
