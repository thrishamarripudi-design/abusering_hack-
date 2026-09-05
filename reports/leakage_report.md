# Leakage Report

Automatically generated. The build aborts with a non-zero exit code
if any of these checks fail (Sec 26, Sec 37).

**Overall result: PASSED**

## Checks

- Ground-truth columns absent from the feature matrix (`ring_id`, `abuse_type`, `abuse_label`, `customer_id`, `transaction_id`, etc.): PASS — 0 violations
- All model input columns are on the declared, audited feature list (no accidental extra column slipped in): PASS
- Feature `prediction_time` exactly matches each transaction's own timestamp (single forward-pass feature engine — Sec 8): PASS — 0 mismatches

## How temporal leakage is prevented structurally

`abusering.features.build.build_features` performs a **single chronological forward pass** over transactions. For every transaction, features are read from rolling per-entity state *before* that transaction is folded into the state. This makes "features only see events with `event_timestamp < prediction_time`" a structural property of the algorithm, not something enforced by a separate filter that could be gotten wrong. `abusering/tests/` additionally injects synthetic future refunds/chargebacks/relationships and asserts they never change a past transaction's feature vector.

## How ring-overlap leakage is prevented

`abusering.splits.ring_split.make_splits` partitions **ring_id**, not individual transactions, into train/validation/test with zero overlap, verified by `verify_no_ring_overlap` (also unit-tested).