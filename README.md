# AbuseRing Sentinel

**Coordinated abuse-ring detection for Razorpay's AI Risk Manager track.**

A defense-only financial-risk ML system that detects groups of accounts
colluding via shared devices, IPs, or payment instruments — something
single-transaction fraud models structurally cannot see.

---

## 1. The Problem

Most fraud/risk systems score **one transaction at a time**. That works for
a single stolen card, but coordinated abuse doesn't look like that — it
looks like 15 accounts sharing one device, created within an hour of each
other, transacting in a 9-minute window. No single transaction in that
group looks unusual in isolation. The pattern only exists **between**
accounts.

Naively reacting to "shared device = suspicious" doesn't work either —
families share a device, offices and hostels share an IP, small businesses
transact frequently. A model (or a hand-written rule) that can't tell these
apart from a coordinated ring will drown a risk team in false positives.

## 2. What This Project Solves

1. **Detects coordination, not just anomalies** — using graph/relationship
   features (shared-resource counts, account-creation bursts, two-hop
   neighborhoods) alongside standard behavioral features.
2. **Doesn't punish innocent look-alikes** — the training data deliberately
   includes legitimate shared-device families, shared-IP offices/hostels,
   and small businesses as "hard negatives," so the model has to learn the
   real distinguishing signal, not a shortcut rule.
3. **Ties every decision to money, not just accuracy** — the review/block
   threshold is chosen by minimizing an explicit financial cost model
   (false-positive review cost + false-negative loss + investigation
   cost), not a default 0.5 cutoff.
4. **Is explainable and auditable** — every prediction returns SHAP-based
   evidence ("18 accounts share this device," "14 transactions in a
   9-minute window") and is logged to a permanent audit trail.

## 3. Who Benefits

- **Risk & fraud investigation teams** — ranked, evidence-backed cases
  instead of a black-box score.
- **Finance/compliance leadership** — set risk appetite by editing a cost
  config file, not by asking a data scientist to retune a model.
- **Payment processors / PSPs** — catch promo abuse, referral-bonus
  farming, and coordinated chargeback rings before payout.
- **Legitimate customers, indirectly** — fewer wrongful blocks, because the
  system is explicitly tested against realistic legitimate look-alikes.

---

## 4. How It Was Built — the full process

Built in the order the underlying spec requires (data before models,
models before UI), with every metric coming from an actually-executed run
— nothing in this repo is hardcoded or invented.

| Phase | What was built |
|---|---|
| 1. Data generation | Synthetic customers, merchants, transactions, refunds, chargebacks, and **6 distinct abuse-ring mechanisms** (shared-device, shared-instrument, coordinated-velocity, dispute-abuse, coordinated-account-creation, hybrid) — plus legitimate "hard negative" populations |
| 2. Feature engineering | A **leakage-safe** feature store: a single forward chronological pass so a transaction's features can only ever see events strictly before it — structurally, not by a filter that could have a bug |
| 3. Train/val/test split | Split by **abuse ring**, not by transaction, with zero ring overlap — so the model is tested on rings it has genuinely never seen |
| 4–10 | Baselines (random, majority-class, rule-based) → Logistic Regression, Random Forest, XGBoost, CatBoost → XGBoost with vs. without graph features (the key experiment) → Optuna hyperparameter tuning → cost-based threshold optimization |
| 11–19 | Full ablation study (behavioral-only vs. graph-only vs. combined, minus velocity, minus dispute features) |
| 20 | SHAP explainability for every prediction |
| 22–26 | Auto-generated model card, dataset card, leakage report, model-selection report — written from the actual results, never hand-authored |
| 27–29 | FastAPI inference service + SQLite-backed audit trail |
| 30 | A static Investigation UI (Overview / Investigation / Model Lab / Dataset Lab / Audit tabs) consuming the live API |
| 34 | 23 automated tests, including tests that inject *future* refunds/chargebacks/relationships and assert they never leak backward into a past transaction's features |

## 5. Step-by-Step: What We Finally Achieved

**Dataset** (seed=42, reproducible): 39,964 transactions, 8,835 customers,
1,500 merchants, 40 abuse rings, 5.9% abuse rate (target: ~6%).

**Split**: 28 rings / 28,017 txns → train · 6 rings / 6,024 txns →
validation · 6 rings / 5,923 txns → **locked test** (zero overlap).

**Model selection**: 9 experiments run (3 baselines + 6 ML variants),
selected by **minimum validation expected financial loss** — not the
highest accuracy. Winner: **XGBoost, Optuna-tuned**.

**The key finding**: does relational/graph information actually help?
Tested directly, not assumed — XGBoost with vs. without graph features:

| | Validation PR-AUC |
|---|---|
| Behavioral features only | 0.834 |
| **+ graph/coordination features** | **0.966** |

**+0.132 PR-AUC lift** — graph features are doing real work.

**Locked test-set result** (rings the model never saw, evaluated once):

| Metric | Value | What it means |
|---|---|---|
| Precision | 0.819 | 82% of flagged transactions were genuinely abuse |
| Recall | 0.830 | Caught 83% of all real abuse in unseen rings |
| PR-AUC | 0.908 | Strong ranking quality despite abuse being rare (~6%) |
| False positives | 52 / 5,640 legit | 0.9% false-positive rate |
| False negatives | 48 / 283 abuse | |
| Threshold | 0.436 | Chosen to minimize ₹ cost, not accuracy |
| Expected financial loss | 97,414 (cost units) | The number the whole system was optimized to minimize |

**Engineering**: 23/23 tests passing, clean lint, reproducible pipeline
(same seed → same numbers), FastAPI + Investigation UI both verified
end-to-end against real data.

---

## 6. Quickstart

```bash
pip install -e ".[dev]"
make data && make features && make splits && make experiments && make reports && make test
make api          # terminal 1 — FastAPI on :8000
make frontend     # terminal 2 — Investigation UI on :8080
```


## 7. Repository Layout

```
src/abusering/
  data/        synthetic generator (6 abuse-ring mechanisms + legit look-alikes)
  features/    leakage-safe temporal feature store
  splits/      ring-based train/val/test split
  models/      baselines + model training
  evaluation/  metrics, cost model, threshold optimization, SHAP, experiment orchestration
  reports/     programmatic report generation
  api/         FastAPI service + audit trail
frontend/      static Investigation UI
tests/         23 tests
reports/       generated: model_selection.md, model_card.md, dataset_card.md, leakage_report.md
```

## 8. Honest Limitations

- Trained and evaluated on **synthetic data** — real-world abuse patterns
  and legitimate distributions will differ.
- Demo-scale dataset (~40k transactions). The `full` profile
  (500k transactions, spec's exact target) generates correctly in ~7s but
  the full model-training run needs more runtime than this project's dev
  sandbox allowed for a single command.
- No GNN — the tabular model's measured graph-feature uplift (+0.132
  PR-AUC) already captures most of the relational signal at far lower
  engineering/serving complexity.
- SQLite instead of PostgreSQL for the audit trail (swappable via one
  environment variable, no code changes).
