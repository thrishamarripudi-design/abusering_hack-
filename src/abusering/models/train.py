"""Model training utilities for E1-E5."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_logreg(X_train, y_train, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Xs, y_train)
    return {"model": clf, "scaler": scaler}


def predict_logreg(bundle, X):
    Xs = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(Xs)[:, 1]


def train_random_forest(X_train, y_train, seed=42, n_estimators=300, max_depth=10):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )
    clf.fit(X_train, y_train)
    return {"model": clf}


def predict_random_forest(bundle, X):
    return bundle["model"].predict_proba(X)[:, 1]


def train_xgboost(X_train, y_train, seed=42, params: dict | None = None):
    import xgboost as xgb
    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    scale_pos_weight = neg / pos
    p = dict(
        n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=2, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=seed,
        eval_metric="aucpr", n_jobs=-1,
    )
    if params:
        p.update(params)
    clf = xgb.XGBClassifier(**p)
    clf.fit(X_train, y_train)
    return {"model": clf}


def predict_xgboost(bundle, X):
    return bundle["model"].predict_proba(X)[:, 1]


def train_catboost(X_train, y_train, seed=42):
    from catboost import CatBoostClassifier
    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    clf = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.08,
        class_weights=[1.0, neg / pos], random_seed=seed, verbose=False,
    )
    clf.fit(X_train, y_train)
    return {"model": clf}


def predict_catboost(bundle, X):
    return bundle["model"].predict_proba(X)[:, 1]


PREDICT_FNS = {
    "logistic_regression": predict_logreg,
    "random_forest": predict_random_forest,
    "xgboost": predict_xgboost,
    "catboost": predict_catboost,
}
