import math

from proyecto_ml.evaluate import compute_metrics, recall_at_k


def test_recall_at_k_perfecto():
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert recall_at_k(y, scores, 0.1) == 1.0


def test_recall_at_k_sin_positivos_es_nan():
    assert math.isnan(recall_at_k([0, 0], [0.1, 0.2]))


def test_compute_metrics_claves_y_una_clase():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert {"roc_auc", "pr_auc", "f1", "recall_at_10pct", "base_rate"} <= m.keys()
    assert math.isnan(m["roc_auc"])
