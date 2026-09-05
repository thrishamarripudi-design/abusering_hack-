"""python -m abusering.evaluation.run_experiments

Runs Phases 4-10, 12, 19 of the spec: baselines, E1-E5, hyperparameter
tuning (Optuna) for the best-looking model family, threshold/cost
optimization, ablations, MLflow logging, and model selection based on
minimum VALIDATION expected financial loss. The TEST set is touched exactly
once, at the very end, using the already-locked model + threshold.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import mlflow
import pandas as pd

from abusering.evaluation.leakage import run_all_checks
from abusering.evaluation.metrics import (
    compute_metrics,
    load_cost_config,
    optimize_threshold,
    precision_at_recall,
    recall_at_precision,
)
from abusering.features.build import (
    BEHAVIORAL_FEATURE_COLUMNS,
    DISPUTE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    GRAPH_FEATURE_COLUMNS,
    VELOCITY_FEATURE_COLUMNS,
)
from abusering.models import baselines
from abusering.models import train as tr


def load_all(data_dir: pathlib.Path):
    features = pd.read_parquet(data_dir / "features.parquet")
    ground_truth = pd.read_parquet(data_dir / "ground_truth.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    return features, ground_truth, transactions, splits


def build_xy(features: pd.DataFrame, ground_truth: pd.DataFrame, txn_ids: list, feature_cols: list):
    sub = features[features.transaction_id.isin(set(txn_ids))].merge(
        ground_truth[["transaction_id", "abuse_label"]], on="transaction_id", how="left"
    )
    X = sub[feature_cols].fillna(0).to_numpy(dtype=float)
    y = sub["abuse_label"].to_numpy(dtype=int)
    amounts = sub["amount"].to_numpy(dtype=float)
    return X, y, amounts, sub["transaction_id"].to_numpy()


def evaluate_split(y_true, y_score, threshold, amounts, cost_cfg):
    y_pred = (y_score >= threshold).astype(int)
    m = compute_metrics(y_true, y_score, y_pred, amounts, cost_cfg)
    m["precision_at_recall_0.5"] = precision_at_recall(y_true, y_score, 0.5)
    m["recall_at_precision_0.9"] = recall_at_precision(y_true, y_score, 0.9)
    return m


def run_optuna_xgb(X_train, y_train, X_val, y_val, n_trials=20, seed=42):
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    base_spw = neg / pos

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", base_spw * 0.5, base_spw * 1.5),
            random_state=seed,
            n_jobs=-1,
            eval_metric="aucpr",
        )
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        score = clf.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score

        return average_precision_score(y_val, score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_store")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--reports", type=str, default="reports")
    ap.add_argument("--cost-config", type=str, default="configs/cost_config.json")
    ap.add_argument("--optuna-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data)
    artifacts_dir = pathlib.Path(args.artifacts)
    reports_dir = pathlib.Path(args.reports)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    mlflow.set_experiment("abusering_sentinel")

    cost_cfg = load_cost_config(args.cost_config)
    features, ground_truth, transactions, splits = load_all(data_dir)

    # --- leakage gate: fail loudly if declared feature columns are unsafe ---
    leak = run_all_checks(FEATURE_COLUMNS, features_df=features, transactions_df=transactions)
    if not leak["passed"]:
        raise SystemExit(f"LEAKAGE CHECK FAILED, aborting experiments: {leak}")
    print("[experiments] leakage gate passed")

    X_train, y_train, amt_train, _ = build_xy(
        features, ground_truth, splits["train_transaction_ids"], FEATURE_COLUMNS
    )
    X_val, y_val, amt_val, _ = build_xy(
        features, ground_truth, splits["validation_transaction_ids"], FEATURE_COLUMNS
    )
    X_test, y_test, amt_test, _ = build_xy(
        features, ground_truth, splits["test_transaction_ids"], FEATURE_COLUMNS
    )
    print(
        f"[experiments] train n={len(y_train)} pos={y_train.sum()} | "
        f"val n={len(y_val)} pos={y_val.sum()} | test n={len(y_test)} pos={y_test.sum()}"
    )

    val_df = features[features.transaction_id.isin(set(splits["validation_transaction_ids"]))].reset_index(
        drop=True
    )
    val_df = val_df.merge(ground_truth[["transaction_id", "abuse_label"]], on="transaction_id")

    results = {}

    def log_experiment(
        name, purpose, y_score_val, feature_set_used, extra=None, model_bundle=None, hyperparams=None
    ):
        thr_result = optimize_threshold(y_val, y_score_val, amt_val, cost_cfg)
        thr = thr_result["selected_threshold"]
        val_metrics = evaluate_split(y_val, y_score_val, thr, amt_val, cost_cfg)

        from sklearn.metrics import precision_recall_curve as _prc

        pr_p, pr_r, pr_thr = _prc(y_val, y_score_val)
        # subsample to keep experiments.json a reasonable size
        step = max(1, len(pr_p) // 150)
        pr_curve = {
            "precision": pr_p[::step].tolist(),
            "recall": pr_r[::step].tolist(),
        }

        entry = {
            "name": name,
            "purpose": purpose,
            "feature_set": feature_set_used,
            "n_features": len(feature_set_used),
            "selected_threshold": thr,
            "validation_metrics": val_metrics,
            "threshold_curve": thr_result[
                "curve"
            ],  # full curve: precision/recall/F1/FP/FN/loss per threshold
            "pr_curve": pr_curve,
            "hyperparams": hyperparams or {},
        }
        if extra:
            entry.update(extra)
        results[name] = entry

        with mlflow.start_run(run_name=name):
            mlflow.log_param("purpose", purpose)
            mlflow.log_param("n_features", len(feature_set_used))
            mlflow.log_param("dataset_version", "demo-seed42")
            mlflow.log_param("seed", args.seed)
            if hyperparams:
                for k, v in hyperparams.items():
                    mlflow.log_param(f"hp_{k}", v)
            mlflow.log_param("selected_threshold", thr)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"val_{k}", v)

        if model_bundle is not None:
            with open(artifacts_dir / f"model_{name}.pkl", "wb") as f:
                pickle.dump(model_bundle, f)
        print(
            f"[experiments] {name}: PR-AUC={val_metrics['pr_auc']:.4f} "
            f"recall={val_metrics['recall']:.3f} precision={val_metrics['precision']:.3f} "
            f"expected_loss={val_metrics['expected_financial_loss']:.0f} thr={thr:.3f}"
        )
        return entry

    t0 = time.time()

    # --- Baseline 0: random ---
    score = baselines.random_baseline_scores(len(y_val), args.seed, y_val.mean())
    log_experiment("baseline_random", "Sanity floor: uninformed classifier.", score, FEATURE_COLUMNS)

    # --- Baseline 1: majority class ---
    score = baselines.majority_baseline_scores(len(y_val))
    log_experiment("baseline_majority", "Sanity floor: always predict legitimate.", score, FEATURE_COLUMNS)

    # --- Baseline 2: rule-based detector ---
    score = baselines.rule_based_scores(val_df)
    log_experiment(
        "baseline_rules",
        "Reasonable non-ML operational approach used for comparison.",
        score,
        [
            "shared_device_count",
            "shared_ip_count",
            "ip_transaction_velocity_1h",
            "shared_instrument_count",
            "transactions_in_time_window_15m",
        ],
    )

    # --- E1: Logistic Regression ---
    bundle = tr.train_logreg(X_train, y_train, seed=args.seed)
    score = tr.predict_logreg(bundle, X_val)
    log_experiment(
        "E1_logistic_regression",
        "Interpretable linear baseline; coefficient sanity check.",
        score,
        FEATURE_COLUMNS,
        model_bundle=bundle,
    )

    # --- E2: Random Forest ---
    bundle = tr.train_random_forest(X_train, y_train, seed=args.seed)
    score = tr.predict_random_forest(bundle, X_val)
    log_experiment(
        "E2_random_forest",
        "Nonlinear interactions, robust tree baseline.",
        score,
        FEATURE_COLUMNS,
        model_bundle=bundle,
    )

    # --- E3: XGBoost (default params, all features incl. graph) ---
    bundle = tr.train_xgboost(X_train, y_train, seed=args.seed)
    score = tr.predict_xgboost(bundle, X_val)
    log_experiment(
        "E3_xgboost",
        "Strong structured/tabular model on engineered risk features.",
        score,
        FEATURE_COLUMNS,
        model_bundle=bundle,
    )

    # --- E4: CatBoost ---
    bundle = tr.train_catboost(X_train, y_train, seed=args.seed)
    score = tr.predict_catboost(bundle, X_val)
    log_experiment(
        "E4_catboost",
        "Compare another strong GBDT approach / categorical handling.",
        score,
        FEATURE_COLUMNS,
        model_bundle=bundle,
    )

    # --- E5: XGBoost WITHOUT graph features (behavioral only) for direct A/B ---
    behav_idx = [FEATURE_COLUMNS.index(c) for c in BEHAVIORAL_FEATURE_COLUMNS]
    bundle = tr.train_xgboost(X_train[:, behav_idx], y_train, seed=args.seed)
    score = tr.predict_xgboost(bundle, X_val[:, behav_idx])
    log_experiment(
        "E5_xgboost_no_graph",
        "Does relational/graph information actually help? (A/B vs E3)",
        score,
        BEHAVIORAL_FEATURE_COLUMNS,
        model_bundle=bundle,
    )

    graph_uplift_pr_auc = (
        results["E3_xgboost"]["validation_metrics"]["pr_auc"]
        - results["E5_xgboost_no_graph"]["validation_metrics"]["pr_auc"]
    )
    print(f"[experiments] graph-feature PR-AUC uplift (E3 - E5_no_graph) = {graph_uplift_pr_auc:+.4f}")

    # --- Hyperparameter optimization (Optuna) on the model family with the
    #     best PR-AUC so far, tuned against the VALIDATION set (Sec 13) ---
    print(f"[experiments] running Optuna ({args.optuna_trials} trials) on XGBoost (all features)...")
    best_params, best_val_ap = run_optuna_xgb(
        X_train, y_train, X_val, y_val, n_trials=args.optuna_trials, seed=args.seed
    )
    bundle = tr.train_xgboost(X_train, y_train, seed=args.seed, params=best_params)
    score = tr.predict_xgboost(bundle, X_val)
    log_experiment(
        "E3b_xgboost_optuna_tuned",
        "XGBoost with Optuna-tuned hyperparameters (all features).",
        score,
        FEATURE_COLUMNS,
        model_bundle=bundle,
        hyperparams=best_params,
    )

    dt = time.time() - t0
    print(f"[experiments] all experiments trained in {dt:.1f}s")

    # ---------------- Ablations (Sec 19) ----------------
    ablation_defs = {
        "A_behavioral_only": BEHAVIORAL_FEATURE_COLUMNS,
        "B_graph_only": GRAPH_FEATURE_COLUMNS,
        "C_behavioral_plus_graph": FEATURE_COLUMNS,
        "D_remove_velocity": [c for c in FEATURE_COLUMNS if c not in VELOCITY_FEATURE_COLUMNS],
        "E_remove_graph": BEHAVIORAL_FEATURE_COLUMNS,
        "F_remove_dispute_features": [c for c in FEATURE_COLUMNS if c not in DISPUTE_FEATURE_COLUMNS],
    }
    ablation_results = {}
    for name, cols in ablation_defs.items():
        idx = [FEATURE_COLUMNS.index(c) for c in cols]
        bundle = tr.train_xgboost(X_train[:, idx], y_train, seed=args.seed)
        score = tr.predict_xgboost(bundle, X_val[:, idx])
        thr_result = optimize_threshold(y_val, score, amt_val, cost_cfg)
        m = evaluate_split(y_val, score, thr_result["selected_threshold"], amt_val, cost_cfg)
        ablation_results[name] = {
            "n_features": len(cols),
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "expected_financial_loss": m["expected_financial_loss"],
        }
        print(
            f"[ablation] {name}: n_feat={len(cols)} PR-AUC={m['pr_auc']:.4f} "
            f"expected_loss={m['expected_financial_loss']:.0f}"
        )

    # ---------------- Model selection (Sec 18): min VALIDATION expected loss ----------------
    candidates = {k: v for k, v in results.items() if not k.startswith("baseline_")}
    winner_name = min(
        candidates, key=lambda k: candidates[k]["validation_metrics"]["expected_financial_loss"]
    )
    print(f"[experiments] WINNER (by validation expected financial loss): {winner_name}")

    # ---------------- Locked test evaluation (Sec 10, 17, 18) — touched ONCE ----------------
    winner_entry = results[winner_name]
    winner_cols = winner_entry["feature_set"]
    winner_thr = winner_entry["selected_threshold"]
    with open(artifacts_dir / f"model_{winner_name}.pkl", "rb") as f:
        winner_bundle = pickle.load(f)

    predict_fn = {
        "E1_logistic_regression": tr.predict_logreg,
        "E2_random_forest": tr.predict_random_forest,
        "E3_xgboost": tr.predict_xgboost,
        "E4_catboost": tr.predict_catboost,
        "E5_xgboost_no_graph": tr.predict_xgboost,
        "E3b_xgboost_optuna_tuned": tr.predict_xgboost,
    }[winner_name]

    idx = [FEATURE_COLUMNS.index(c) for c in winner_cols]
    X_test_winner = X_test[:, idx]
    test_score = predict_fn(winner_bundle, X_test_winner)
    test_metrics = evaluate_split(y_test, test_score, winner_thr, amt_test, cost_cfg)
    print(f"[experiments] LOCKED TEST metrics for {winner_name} @ threshold={winner_thr:.3f}: {test_metrics}")

    with mlflow.start_run(run_name=f"{winner_name}_LOCKED_TEST"):
        mlflow.log_param("winner", winner_name)
        mlflow.log_param("selected_threshold", winner_thr)
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)

    # ---------------- Score EVERY transaction (train+val+test) with the
    # winner model, for the API's /overview endpoint — avoids rescoring at
    # request time and keeps the API read-only over cached artifacts. ----
    X_all, y_all, amt_all, ids_all = build_xy(
        features,
        ground_truth,
        splits["train_transaction_ids"]
        + splits["validation_transaction_ids"]
        + splits["test_transaction_ids"],
        FEATURE_COLUMNS,
    )
    all_scores = predict_fn(winner_bundle, X_all[:, idx])
    scored_all = pd.DataFrame(
        {
            "transaction_id": ids_all,
            "risk_score": all_scores,
            "predicted_abuse": (all_scores >= winner_thr).astype(int),
            "amount": amt_all,
        }
    )
    scored_all.to_parquet(artifacts_dir / "scored_transactions.parquet", index=False)
    print(
        f"[experiments] wrote {artifacts_dir / 'scored_transactions.parquet'} "
        f"({len(scored_all)} scored transactions, {scored_all['predicted_abuse'].sum()} flagged)"
    )

    # ---------------- Persist everything ----------------
    summary = {
        "dataset_version": "see data_store/manifest.json",
        "feature_version": "features-v1",
        "seed": args.seed,
        "cost_config": cost_cfg,
        "experiments": results,
        "ablations": ablation_results,
        "graph_feature_pr_auc_uplift_E3_minus_E5": graph_uplift_pr_auc,
        "winner": winner_name,
        "winner_threshold": winner_thr,
        "locked_test_metrics": test_metrics,
        "leakage_check": leak,
        "runtime_seconds": dt,
    }
    with open(artifacts_dir / "experiments.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[experiments] wrote {artifacts_dir / 'experiments.json'}")


if __name__ == "__main__":
    main()
