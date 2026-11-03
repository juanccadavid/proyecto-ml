import math

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import COLUMNAS_COMPARATIVA, TIEMPO, compute_metrics, recall_at_k, resaltar, tabla_comparativa
from proyecto_ml.generador import GeneradorAsociados


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


def test_tabla_comparativa_como_pycaret():
    df = GeneradorAsociados(n=300, seed=3).generar()
    modelos = a.modelos_supervisados()
    tabla = tabla_comparativa(modelos, a.preparar(df), df[a.TARGET], folds=3, ordenar="Recall")

    assert list(tabla.columns) == ["Model", *COLUMNAS_COMPARATIVA]
    assert sorted(tabla.index) == sorted(modelos)
    assert tabla.loc["random_forest", "Model"] == "RandomForestClassifier"
    assert tabla["Recall"].is_monotonic_decreasing  # ordenada por la columna pedida
    assert tabla[COLUMNAS_COMPARATIVA[:-1]].le(1).all().all()  # métricas en [-1, 1]
    assert tabla.loc["ridge", "AUC"] > 0.8  # Ridge sin predict_proba: AUC con decision_function (no 0)
    assert (tabla[TIEMPO] > 0).all()

    rapida = tabla_comparativa(modelos, a.preparar(df), df[a.TARGET], folds=3, ordenar=TIEMPO)
    assert rapida[TIEMPO].is_monotonic_increasing  # en tiempo, el más rápido arriba
    # pandas junta en una regla CSS todas las celdas resaltadas: al menos una por columna (puede haber empates).
    selector = resaltar(tabla).to_html().split("background-color: #ffff00")[0].rsplit("}", 1)[-1]
    assert selector.count("_col") >= len(COLUMNAS_COMPARATIVA)
