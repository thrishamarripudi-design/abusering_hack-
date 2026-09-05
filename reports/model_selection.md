# Model Selection Report

Generated programmatically from `artifacts/experiments.json`. Every number below came from an executed run — nothing here is invented.

## 1. What models were tested, and why?

| Experiment | Why tested | # features | Val PR-AUC | Val ROC-AUC | Val precision | Val recall | Val F1 | Threshold | Val expected loss |
|---|---|---|---|---|---|---|---|---|---|
| baseline_random | Sanity floor: uninformed classifier. | 41 | 0.0653 | 0.4981 | 0.0548 | 0.0104 | 0.0175 | 0.9900 | 218776 |
| baseline_majority | Sanity floor: always predict legitimate. | 41 | 0.0637 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 | 178322 |
| baseline_rules | Reasonable non-ML operational approach used for comparison. | 5 | 0.4343 | 0.7336 | 0.9675 | 0.3880 | 0.5539 | 0.2030 | 138660 |
| E1_logistic_regression | Interpretable linear baseline; coefficient sanity check. | 41 | 0.8282 | 0.9821 | 0.9459 | 0.4557 | 0.6151 | 0.9702 | 130302 |
| E2_random_forest | Nonlinear interactions, robust tree baseline. | 41 | 0.9256 | 0.9890 | 0.9463 | 0.8255 | 0.8818 | 0.5990 | 94562 |
| E3_xgboost | Strong structured/tabular model on engineered risk features. | 41 | 0.9667 | 0.9958 | 0.9169 | 0.9193 | 0.9181 | 0.5643 | 71840 |
| E4_catboost | Compare another strong GBDT approach / categorical handling. | 41 | 0.9613 | 0.9940 | 0.9540 | 0.8646 | 0.9071 | 0.7623 | 79319 |
| E5_xgboost_no_graph | Does relational/graph information actually help? (A/B vs E3) | 22 | 0.8340 | 0.9767 | 0.8690 | 0.5703 | 0.6887 | 0.8960 | 128330 |
| E3b_xgboost_optuna_tuned | XGBoost with Optuna-tuned hyperparameters (all features). | 41 | 0.9691 | 0.9958 | 0.9225 | 0.9297 | 0.9261 | 0.4159 | 69017 |

Baselines (random, majority-class, rule-based) establish the floor any ML model must beat; see `artifacts/experiments.json` for their validation numbers alongside the ML experiments above.

## 2. Which model won, and why?

**Winner: `E3b_xgboost_optuna_tuned`**, selected by **minimum validation expected financial loss** (Sec 18) — 69017 on the validation set, using cost config {"false_positive_cost": 500.0, "investigation_cost": 100.0, "loss_rate_on_missed_abuse": 0.85}. Secondary criteria (PR-AUC, recall, precision, calibration) were consulted but did not override the financial-loss ranking.

Selected (locked) threshold: **0.4159**, chosen by grid search over candidate thresholds on the VALIDATION set only (Sec 17), then frozen before the test set was touched.

## 3. Did graph/relational features actually help? (Sec 19, E5)

XGBoost with all features (E3) vs. XGBoost with graph/coordination features removed (E5_xgboost_no_graph): **PR-AUC uplift = +0.1328** on the validation set. This is a substantial, not marginal, improvement — graph/coordination signals (shared-device/IP/instrument counts, two-hop neighborhood size, account-creation-burst proximity) carry real information for this task that transaction-level behavioral features alone do not capture.

## 4. Ablation results (Sec 19)

| Ablation | # features | Val PR-AUC | Val ROC-AUC | Val precision | Val recall | Val expected loss |
|---|---|---|---|---|---|---|
| A_behavioral_only | 22 | 0.8340 | 0.9767 | 0.8690 | 0.5703 | 128330 |
| B_graph_only | 19 | 0.7153 | 0.9379 | 1.0000 | 0.6146 | 115992 |
| C_behavioral_plus_graph | 41 | 0.9667 | 0.9958 | 0.9169 | 0.9193 | 71840 |
| D_remove_velocity | 34 | 0.9655 | 0.9956 | 0.9182 | 0.9062 | 75043 |
| E_remove_graph | 22 | 0.8340 | 0.9767 | 0.8690 | 0.5703 | 128330 |
| F_remove_dispute_features | 38 | 0.9568 | 0.9939 | 0.9568 | 0.8646 | 77883 |

- **A vs B**: behavioral-only vs. graph-only — shows each family's standalone signal.
- **C**: behavioral + graph together (full feature set).
- **D**: full set minus velocity features — isolates velocity's contribution.
- **E**: full set minus graph features (identical to A by construction).
- **F**: full set minus dispute/chargeback-history features.

## 5. Locked test-set result (touched exactly once, Sec 10/17/18)

Model `E3b_xgboost_optuna_tuned` at the pre-registered threshold `0.4159`, evaluated on previously-unseen abuse rings held out of all training and threshold-selection steps:

- Precision: **0.8185**
- Recall: **0.8445**
- F1: **0.8313**
- PR-AUC: **0.9059**
- ROC-AUC: **0.9864**
- False positives: 53, False negatives: 44
- Expected financial loss (locked test set): 95460

Note the test-set expected loss and recall differ from validation — this is the honest, expected effect of evaluating on entirely new, previously unseen abuse rings rather than re-testing on rings the model's hyperparameters/threshold were already exposed to via validation. We report it as measured, not adjusted.

## 6. Weaknesses / limitations

- Trained and evaluated on **synthetic** data; see `dataset_card.md`.
- Demo-scale run (see `data_store/manifest.json` for exact counts) — the spec's full-scale profile (100k customers / 500k transactions / 100 rings) is implemented and reachable via `--profile full` but was not the profile used to produce the numbers in this report, for sandbox runtime reasons.
- Coordinated-velocity and account-creation-burst rings are the hardest to catch with behavioral-only features (see ablation A) — this is exactly the case graph/coordination features are meant to address.
- No GNN (E6) was implemented — see `README.md` for the justification.