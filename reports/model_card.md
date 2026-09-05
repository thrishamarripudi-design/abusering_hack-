# Model Card — AbuseRing Sentinel

**Model:** `E3b_xgboost_optuna_tuned`  
**Model version:** risk-v1.0  
**Feature version:** features-v1  
**Selected threshold:** 0.4159  

## Intended use

Flag transactions for **coordinated-abuse-risk review** in a human-in-the-loop investigation workflow. Output is a risk score and evidence, not an automated block/allow decision and not a fraud or criminal determination.

## Non-intended use

- Not a standalone auto-decline system.
- Not a criminal-fraud accusation tool — terminology throughout this system is "coordinated abuse risk", never "confirmed fraud".
- Not validated on real production traffic — trained and tested entirely on synthetic data (see dataset card).

## Training data

Synthetic dataset, profile=`demo`, seed=42, 39964 transactions, 40 abuse rings, abuse rate=0.05915. See `reports/dataset_card.md`.

## Labels

Binary `abuse_label`: 1 if a transaction belongs to a synthetically generated coordinated abuse ring, 0 otherwise. Ground truth is generated independently of the model's input features (Sec 7); ring/abuse-type metadata is never used as a feature.

## Split strategy

70/15/15 split **by abuse ring** (not by transaction), so the model is evaluated on entirely unseen rings, not just unseen transactions from already-seen rings. See `data_store/splits.json`.

## Metrics (locked test set, unseen rings)

- Precision: 0.8185, Recall: 0.8445, F1: 0.8313
- PR-AUC: 0.9059, ROC-AUC: 0.9864
- Brier score: 0.0128
- False positive rate: 0.0094
- Expected financial loss: 95460

## Limitations

- Synthetic training/eval data; real-world abuse patterns, legitimate-lookalike distributions, and class balance will differ.
- Demo-scale data volume (see `data_store/manifest.json`).
- No adversarial robustness testing against an attacker who knows the model's feature set.
- Calibration (Brier score above) has not been separately verified against a held-out recalibration set.

## False-positive risk

A false positive triggers a review/investigation cost (configured at 100.0 per review) and potential customer friction — not an automatic account suspension in this system's intended design.

## False-negative risk

A missed coordinated-abuse transaction is costed at 85% of transaction amount in the financial cost model used for threshold selection — tune `configs/cost_config.json` to your actual risk appetite.

## Known biases

The synthetic generator's legitimate "hard negative" populations (shared-device families, shared-IP offices/hostels, small businesses) are hand-designed approximations of real-world look-alike behavior and may not cover every legitimate pattern that exists in production traffic.