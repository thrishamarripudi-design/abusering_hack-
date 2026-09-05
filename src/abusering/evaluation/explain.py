"""SHAP explainability (Sec 20). Every explanation is grounded in actual
feature values and actual SHAP contributions — never invented."""

from __future__ import annotations

import numpy as np


def explain_prediction(model, X_row: np.ndarray, feature_names: list[str], top_k: int = 5) -> dict:
    """Return top positive/negative contributing features for one row,
    using the model's native SHAP-compatible tree explainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row.reshape(1, -1))
    if isinstance(sv, list):  # some wrappers return [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv).reshape(-1)

    contributions = list(zip(feature_names, sv, X_row.tolist()))
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_positive = [
        {"feature": f, "shap_value": float(v), "feature_value": float(val)}
        for f, v, val in contributions[:top_k]
        if v > 0
    ]
    top_negative = [
        {"feature": f, "shap_value": float(v), "feature_value": float(val)}
        for f, v, val in contributions[-top_k:][::-1]
        if v < 0
    ]
    return {"top_positive_signals": top_positive, "top_negative_signals": top_negative}
