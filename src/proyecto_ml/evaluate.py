"""Métricas para clasificación binaria desbalanceada."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_k(y_true, y_score, k_frac: float = 0.1) -> float:
    """Proporción de positivos capturados en el top ``k_frac`` de puntajes.

    Responde: "si el negocio solo puede gestionar al 10 % con mayor riesgo,
    ¿qué fracción de los casos reales encuentra?".
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_score) * k_frac)))
    top_idx = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[top_idx].sum() / positives)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "base_rate": float(y_true.mean()),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_at_10pct": recall_at_k(y_true, y_score, 0.10),
    }
