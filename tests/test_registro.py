"""Registro en MLflow: nombres y prefijos de métricas, corridas anidadas y el alias del campeón."""

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import registro as r
from proyecto_ml.generador import GeneradorAsociados


def test_nombres_y_prefijos():
    assert r.nombre_metrica("Prec.") == "prec"
    assert r.nombre_metrica("TT (Sec)") == "tt_sec"
    metricas = r.con_prefijo("cv", {"roc_auc": 0.9, "F1": 0.5, "vacio": float("nan"), "Model": "x", "ok": True})
    assert metricas == {"cv_roc_auc": 0.9, "cv_f1": 0.5}  # sin NaN, textos ni booleanos


def test_tabla_y_union():
    tabla = pd.DataFrame({"AUC": [0.9, 0.8], "Recall": [0.5, 0.6]}, index=["ridge", "logreg"])
    cv = r.metricas_tabla(tabla, "cv")
    assert cv["ridge"] == {"cv_auc": 0.9, "cv_recall": 0.5}
    unido = r.unir(cv, {"ridge": {"nuevos_auc": 0.7}})
    assert unido["ridge"] == {"cv_auc": 0.9, "cv_recall": 0.5, "nuevos_auc": 0.7} and "logreg" in unido


@pytest.mark.integration
def test_corridas_y_alias_del_campeon(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    df = GeneradorAsociados(n=200, seed=1).generar()
    X, y = a.preparar(df), df[a.TARGET]
    modelo = a.modelos_supervisados()["logreg"].fit(X, y)
    agrupador = a.AgrupadorAbandono().fit(df)

    r.configurar("prueba")
    artefacto = tmp_path / "resumen.json"
    artefacto.write_text("{}", encoding="utf-8")
    with r.corrida("padre", {"filas": 200}, artefactos=[artefacto]) as padre:
        ids = r.registrar_modelos(
            {"logreg": modelo, "kmeans": agrupador}, {"datos": "prueba"}, {"logreg": {"cv_pr_auc": 0.9}}
        )
        with r.corrida("campeon_logreg", anidada=True):
            info = r.registrar_modelo(modelo, registrar_como=r.MODELO_REGISTRADO)
    r.asignar_alias(info.registered_model_version)

    hija = mlflow.get_run(ids["logreg"])
    assert hija.data.tags["mlflow.parentRunId"] == padre
    assert hija.data.metrics["cv_pr_auc"] == 0.9
    assert hija.data.params["model__C"] == "1.0" and hija.data.params["datos"] == "prueba"
    assert mlflow.get_run(ids["kmeans"]).data.params["model__n_clusters"] == "2"  # el KMeans del agrupador
    cargado = mlflow.sklearn.load_model(f"models:/{r.MODELO_REGISTRADO}@{r.ALIAS_CAMPEON}")
    assert (cargado.predict_proba(X) == modelo.predict_proba(X)).all()
