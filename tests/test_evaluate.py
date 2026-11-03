import math

import numpy as np

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import (
    COLUMNAS_COMPARATIVA,
    TIEMPO,
    afinar,
    compute_metrics,
    figura_busqueda,
    formato_params,
    mejores_por_grupo,
    recall_at_k,
    resaltar,
    resultados_busqueda,
    resultados_por_fold,
    tabla_comparativa,
)
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


def test_afinar_usa_las_mismas_particiones_que_la_tabla():
    df = GeneradorAsociados(n=300, seed=3).generar()
    X, y = a.preparar(df), df[a.TARGET]
    grilla = {"model__alpha": [0.1, 1.0, 100.0], "model__class_weight": [None, "balanced"]}
    busqueda = afinar(a.modelo_ridge(), grilla, X, y, folds=3, metrica="F1")

    assert busqueda.best_params_["model__alpha"] in grilla["model__alpha"]
    top = resultados_busqueda(busqueda, top=4)
    assert list(top.columns) == ["params", *COLUMNAS_COMPARATIVA] and len(top) == 4
    assert top["F1"].is_monotonic_decreasing and top.loc[0, "F1"] == busqueda.best_score_
    assert formato_params({"model__alpha": 100.0, "model__class_weight": None}) == "alpha=100, class_weight=None"

    # El modelo por defecto (alpha=1, balanced) puntúa igual en la búsqueda y en la tabla comparativa.
    metricas = COLUMNAS_COMPARATIVA[:-1]  # sin el tiempo, que varía entre corridas
    tabla = tabla_comparativa({"ridge": a.modelo_ridge()}, X, y, folds=3)
    todas = resultados_busqueda(busqueda, top=6)
    por_defecto = todas[todas["params"] == "alpha=1, class_weight=balanced"].iloc[0]
    assert np.allclose(por_defecto[metricas].astype(float), tabla.loc["ridge", metricas].astype(float))

    fig = figura_busqueda({"ridge": busqueda}, {"ridge": "alpha"})
    etiquetas = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
    assert etiquetas == ["class_weight=None", "class_weight=balanced", "por defecto"]  # None no se pierde (pandas 3)


def test_tabla_por_fold_un_pipeline_por_preprocesamiento():
    from sklearn.pipeline import Pipeline

    from proyecto_ml.features import Preprocesador

    df = GeneradorAsociados(n=300, seed=3).generar()
    X, y = a.preparar(df), df[a.TARGET]
    modelo = Pipeline([("preprocess", Preprocesador(a.NUMERICAS, a.CATEGORICAS)), ("model", a.modelo_ridge()[-1])])
    grilla = {"preprocess__encoding": ["onehot", "ordinal"], "model__alpha": [0.1, 10.0]}
    busqueda = afinar(modelo, grilla, X, y, folds=3, metrica="AUC")

    por_fold = resultados_por_fold(busqueda, ["encoding"], "Ridge", metricas=("Accuracy", "AUC"))
    assert list(por_fold.columns) == ["pipeline", "fold", "Accuracy", "AUC"]
    assert list(por_fold["pipeline"].unique()) == ["Ridge_onehot", "Ridge_ordinal"]
    assert list(por_fold["fold"]) == [1, 2, 3, 1, 2, 3]

    # Cada grupo usa su mejor alpha, y sus puntajes son los de esas particiones en cv_results_.
    mejores = mejores_por_grupo(busqueda, ["encoding"], "Ridge")
    r = busqueda.cv_results_
    for pipeline, params in mejores.items():
        i = r["params"].index(params)
        mismo_grupo = [
            j for j, p in enumerate(r["params"]) if p["preprocess__encoding"] == params["preprocess__encoding"]
        ]
        assert r["mean_test_AUC"][i] == max(r["mean_test_AUC"][j] for j in mismo_grupo)
        filas = por_fold[por_fold["pipeline"] == pipeline]
        assert np.allclose(filas["Accuracy"], [r[f"split{k}_test_Accuracy"][i] for k in range(3)])
