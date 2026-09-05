# Dataset Card — AbuseRing Sentinel Synthetic Data

**This is entirely synthetic data.** No real customer, merchant, or transaction data was used anywhere in this project.

## Generation

- Profile: `demo`
- Seed: 42
- Customers: 8835
- Merchants: 1500
- Transactions: 39964
- Abuse rings: 40
- Abuse rate (transaction-level): 0.05915
- Refunds: 1935, Chargebacks: 707
- Relationship edges: 69038

Generation is deterministic given `(profile, seed)`. Reproduce with:

```
python -m abusering.data.generate --profile demo --seed 42
```

A `--profile full` option reproduces the spec's exact target scale (100k customers / 20k merchants / 500k transactions / ~100 rings); the numbers in this project's reports were produced at the `demo` profile for sandbox runtime reasons — see `abusering/data/config.py`.

## Legitimate populations (hard negatives)

Customers are assigned one of seven segments: `normal`, `frequent`, `high_value`, `family_shared_device`, `office_shared_ip`, `hostel_shared_network`, `small_business`. The shared-resource segments deliberately create customers who **look** coordinated (same device or IP as several other real people) but transact normally, so a naive "shared IP = abuse" rule scores much worse than the trained models (see `baseline_rules` in `artifacts/experiments.json`).

## Abuse mechanisms

Six generator families (Sec 6 of the build spec): shared-device rings, shared-payment-instrument rings, coordinated-velocity rings, dispute/refund-abuse rings, coordinated-account-creation rings, and hybrid rings combining two mechanisms. The generator type is never exposed to the model — only `ground_truth.parquet` carries it, for evaluation only.

## Temporal characteristics

Transactions span a configured date range (`abusering/data/config.py`); every transaction timestamp is constrained to be at or after its own customer's account-creation timestamp.

## Split strategy

70/15/15 by **abuse ring**, with independent random 70/15/15 partitioning of legitimate transactions across the same three splits. Zero ring overlap, verified programmatically.

## Limitations

- Synthetic amount/timing distributions are simplified log-normal / uniform approximations, not fit to real transaction data.
- Ring sizes, velocities, and burst windows are drawn from reasonable-looking ranges chosen by the generator's author, not calibrated against real abuse-ring case data.
- At demo scale, some feature windows (e.g. 30-day uniques) see fewer events per entity than they would at full scale.