"""python -m uvicorn abusering.api.main:app --reload

FastAPI inference service. Loads the winner model selected by
run_experiments.py, serves risk scoring with SHAP evidence, and writes
every prediction to the audit trail (Sec 27, 29).
"""

from __future__ import annotations

import json
import pathlib
import pickle
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from abusering.api.db import PredictionAudit, get_session, init_db
from abusering.evaluation.explain import explain_prediction
from abusering.features.build import FEATURE_VERSION
from abusering.models import train as tr

ARTIFACTS_DIR = pathlib.Path("artifacts")
DATA_DIR = pathlib.Path("data_store")

app = FastAPI(title="AbuseRing Sentinel", version="1.0")

# The Investigation UI (frontend/) is a static page served separately (e.g.
# `python -m http.server` from frontend/) and calls this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {}


def _predict_fn_for(name: str):
    if name.startswith("E1"):
        return tr.predict_logreg
    if name.startswith("E2"):
        return tr.predict_random_forest
    if name.startswith("E4"):
        return tr.predict_catboost
    return tr.predict_xgboost  # E3, E3b, E5


@app.on_event("startup")
def load_artifacts():
    init_db()
    exp_path = ARTIFACTS_DIR / "experiments.json"
    if not exp_path.exists():
        _state["ready"] = False
        _state["error"] = (
            "artifacts/experiments.json not found — run "
            "`python -m abusering.evaluation.run_experiments` first."
        )
        return
    with open(exp_path) as f:
        summary = json.load(f)
    winner = summary["winner"]
    with open(ARTIFACTS_DIR / f"model_{winner}.pkl", "rb") as f:
        bundle = pickle.load(f)

    _state.update({
        "ready": True,
        "winner": winner,
        "threshold": summary["experiments"][winner]["selected_threshold"],
        "feature_set": summary["experiments"][winner]["feature_set"],
        "bundle": bundle,
        "predict_fn": _predict_fn_for(winner),
        "model_version": "risk-v1.0",
        "feature_version": FEATURE_VERSION,
        "locked_test_metrics": summary["locked_test_metrics"],
        "summary": summary,
    })

    # feature store snapshot for inference-time lookups
    if (DATA_DIR / "features.parquet").exists():
        _state["features"] = pd.read_parquet(DATA_DIR / "features.parquet")
        gt = pd.read_parquet(DATA_DIR / "ground_truth.parquet")
        _state["ground_truth"] = gt
    if (DATA_DIR / "transactions.parquet").exists():
        _state["transactions"] = pd.read_parquet(DATA_DIR / "transactions.parquet")
    if (DATA_DIR / "customers.parquet").exists():
        _state["customers"] = pd.read_parquet(DATA_DIR / "customers.parquet")
    if (DATA_DIR / "ring_stats.parquet").exists():
        _state["ring_stats"] = pd.read_parquet(DATA_DIR / "ring_stats.parquet")
    if (DATA_DIR / "manifest.json").exists():
        with open(DATA_DIR / "manifest.json") as f:
            _state["dataset_manifest"] = json.load(f)
    if (DATA_DIR / "splits.json").exists():
        with open(DATA_DIR / "splits.json") as f:
            _state["splits"] = json.load(f)
    if (ARTIFACTS_DIR / "scored_transactions.parquet").exists():
        _state["scored"] = pd.read_parquet(ARTIFACTS_DIR / "scored_transactions.parquet")


class PredictRequest(BaseModel):
    transaction_id: str


def _risk_band(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok" if _state.get("ready") else "not_ready", "error": _state.get("error")}


@app.get("/model/info")
def model_info():
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    return {
        "model_version": _state["model_version"],
        "feature_version": _state["feature_version"],
        "winning_model": _state["winner"],
        "selected_threshold": _state["threshold"],
        "n_features": len(_state["feature_set"]),
        "locked_test_metrics": _state["locked_test_metrics"],
    }


@app.get("/experiments")
def experiments():
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    return _state["summary"]["experiments"]


@app.get("/metrics/benchmark")
def metrics_benchmark():
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    return {
        "winner": _state["winner"],
        "locked_test_metrics": _state["locked_test_metrics"],
        "ablations": _state["summary"]["ablations"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    feats = _state.get("features")
    if feats is None:
        raise HTTPException(503, "feature store not loaded")

    row = feats[feats.transaction_id == req.transaction_id]
    if len(row) == 0:
        raise HTTPException(
            404,
            f"transaction_id {req.transaction_id!r} not found in the feature "
            "store. This demo endpoint scores previously-generated "
            "transactions; a production deployment would compute features "
            "for a live transaction from the streaming feature store "
            "(abusering.features.build) instead.",
        )

    t0 = time.time()
    feature_cols = _state["feature_set"]
    X = row[feature_cols].fillna(0).to_numpy(dtype=float)
    score = float(_state["predict_fn"](_state["bundle"], X)[0])
    prediction = "coordinated_abuse_risk" if score >= _state["threshold"] else "normal"

    evidence = {"top_positive_signals": [], "top_negative_signals": []}
    try:
        model = _state["bundle"]["model"]
        if hasattr(model, "get_booster") or type(model).__name__ == "CatBoostClassifier":
            evidence = explain_prediction(model, X[0], feature_cols, top_k=5)
    except Exception as e:  # SHAP is best-effort evidence, never blocks a response
        evidence = {"error": f"explanation unavailable: {e}"}

    latency_ms = (time.time() - t0) * 1000

    session = get_session()
    try:
        audit = PredictionAudit(
            transaction_id=req.transaction_id,
            model_version=_state["model_version"],
            feature_version=_state["feature_version"],
            risk_score=score,
            prediction=prediction,
            threshold=_state["threshold"],
            top_signals=evidence.get("top_positive_signals", []),
            inference_latency_ms=latency_ms,
        )
        session.add(audit)
        session.commit()
    finally:
        session.close()

    return {
        "risk_score": round(score, 6),
        "prediction": prediction,
        "risk_band": _risk_band(score),
        "model_version": _state["model_version"],
        "feature_version": _state["feature_version"],
        "top_signals": evidence.get("top_positive_signals", []),
        "evidence": evidence,
        "inference_latency_ms": round(latency_ms, 3),
    }


@app.get("/investigations/{transaction_id}")
def investigation(transaction_id: str):
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    feats = _state.get("features")
    gt = _state.get("ground_truth")
    row = feats[feats.transaction_id == transaction_id]
    if len(row) == 0:
        raise HTTPException(404, f"transaction_id {transaction_id!r} not found")
    row = row.iloc[0]

    feature_cols = _state["feature_set"]
    X = row[feature_cols].fillna(0).to_numpy(dtype=float)
    score = float(_state["predict_fn"](_state["bundle"], X.reshape(1, -1))[0])
    prediction = "coordinated_abuse_risk" if score >= _state["threshold"] else "normal"

    truth_row = gt[gt.transaction_id == transaction_id]
    ground_truth = None
    if len(truth_row):
        tr_row = truth_row.iloc[0]
        ground_truth = {
            "abuse_label": int(tr_row["abuse_label"]),
            "ring_id": tr_row["ring_id"],
            "abuse_type": tr_row["abuse_type"],
        }  # exposed here for demo/investigation purposes ONLY, never as a model input

    session = get_session()
    try:
        history = (
            session.query(PredictionAudit)
            .filter(PredictionAudit.transaction_id == transaction_id)
            .order_by(PredictionAudit.timestamp.desc())
            .all()
        )
        audit_events = [
            {
                "timestamp": str(h.timestamp), "risk_score": h.risk_score,
                "prediction": h.prediction, "model_version": h.model_version,
            }
            for h in history
        ]
    finally:
        session.close()

    return {
        "transaction_id": transaction_id,
        "risk_score": round(score, 6),
        "prediction": prediction,
        "risk_band": _risk_band(score),
        "transaction": {
            "amount": float(row["amount"]),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
        },
        "evidence_features": {c: float(row[c]) for c in [
            "shared_device_count", "shared_ip_count", "shared_instrument_count",
            "transactions_in_time_window_15m", "accounts_created_nearby",
            "two_hop_neighbor_count", "community_size",
        ]},
        "ground_truth_for_evaluation_only": ground_truth,
        "audit_trail": audit_events,
    }


@app.get("/overview")
def overview():
    """Aggregate stats for the UI's Overview tab (Sec 30). All numbers come
    from artifacts/scored_transactions.parquet (real, cached model scores)
    and artifacts/experiments.json — nothing computed ad hoc here."""
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    scored = _state.get("scored")
    if scored is None:
        raise HTTPException(503, "artifacts/scored_transactions.parquet not found — rerun run_experiments.py")

    gt = _state.get("ground_truth")
    flagged = scored[scored.predicted_abuse == 1]
    summary = _state["summary"]

    n_known_rings = None
    if gt is not None:
        n_known_rings = int(gt["ring_id"].dropna().nunique())

    return {
        "transactions_monitored": int(len(scored)),
        "predicted_abuse_count": int(flagged.shape[0]),
        "predicted_abuse_rate": float(flagged.shape[0] / len(scored)) if len(scored) else 0.0,
        "known_abuse_ring_clusters_evaluation_only": n_known_rings,
        "potential_financial_exposure_flagged_amount": float(flagged["amount"].sum()),
        "model_version": _state["model_version"],
        "winning_model": _state["winner"],
        "locked_test_metrics": _state["locked_test_metrics"],
        "baseline_rules_validation_metrics": summary["experiments"]["baseline_rules"]["validation_metrics"],
        "selected_model_validation_metrics": summary["experiments"][_state["winner"]]["validation_metrics"],
    }


@app.get("/model/curves")
def model_curves(experiment: str = None):
    """PR curve + threshold/cost curve for the Model Lab tab. Defaults to
    the winning model if no experiment name is given."""
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    exps = _state["summary"]["experiments"]
    name = experiment or _state["winner"]
    if name not in exps:
        raise HTTPException(404, f"unknown experiment {name!r}. Options: {list(exps)}")
    e = exps[name]
    return {
        "name": name,
        "pr_curve": e.get("pr_curve"),
        "threshold_curve": e.get("threshold_curve"),
        "selected_threshold": e["selected_threshold"],
        "validation_metrics": e["validation_metrics"],
    }


@app.get("/dataset/summary")
def dataset_summary():
    """Class distribution, ring distribution, legitimate-lookalike segment
    counts, and split composition for the Dataset Lab tab (Sec 30). Read
    directly from data_store/ — never fabricated."""
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    gt = _state.get("ground_truth")
    customers = _state.get("customers")
    ring_stats = _state.get("ring_stats")
    splits = _state.get("splits")
    manifest = _state.get("dataset_manifest")
    if gt is None or customers is None or ring_stats is None or splits is None:
        raise HTTPException(503, "dataset artifacts not fully loaded")

    class_dist = gt["abuse_label"].value_counts().to_dict()
    ring_type_dist = ring_stats["abuse_type"].value_counts().to_dict()
    segment_dist = customers["customer_segment"].value_counts().to_dict()

    split_composition = {
        "train": {"n_rings": len(splits["train_ring_ids"]), "n_transactions": splits["train_transaction_count"]},
        "validation": {"n_rings": len(splits["validation_ring_ids"]), "n_transactions": splits["validation_transaction_count"]},
        "test": {"n_rings": len(splits["test_ring_ids"]), "n_transactions": splits["test_transaction_count"]},
    }

    return {
        "manifest": manifest,
        "class_distribution": {"legitimate": int(class_dist.get(0, 0)), "abuse": int(class_dist.get(1, 0))},
        "ring_type_distribution": {str(k): int(v) for k, v in ring_type_dist.items()},
        "legitimate_lookalike_segment_distribution": {str(k): int(v) for k, v in segment_dist.items()},
        "split_composition": split_composition,
        "ring_stats": ring_stats.to_dict(orient="records"),
    }


@app.get("/audit/recent")
def audit_recent(limit: int = 50):
    """Most recent prediction-audit events for the Audit tab (Sec 30, 29)."""
    if not _state.get("ready"):
        raise HTTPException(503, _state.get("error", "not ready"))
    session = get_session()
    try:
        rows = (
            session.query(PredictionAudit)
            .order_by(PredictionAudit.timestamp.desc())
            .limit(min(limit, 500))
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": str(r.timestamp),
                "transaction_id": r.transaction_id,
                "model_version": r.model_version,
                "feature_version": r.feature_version,
                "risk_score": r.risk_score,
                "prediction": r.prediction,
                "threshold": r.threshold,
                "top_signals": r.top_signals,
                "inference_latency_ms": r.inference_latency_ms,
            }
            for r in rows
        ]
    finally:
        session.close()
