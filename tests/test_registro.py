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


# ---------------------------------------------------------------- versión de los datos


def test_huella_cambia_con_una_sola_celda():
    from proyecto_ml.generador import FALTANTES

    df = a.foto(GeneradorAsociados(n=300, seed=1, faltantes=FALTANTES).generar(), 3)
    huella = r.huella(df)
    assert len(huella) == 12 and huella == r.huella(df.copy())  # mismos datos, misma huella
    assert huella == r.huella(GeneradorAsociados(n=300, seed=1, faltantes=FALTANTES).generar().pipe(a.foto, 3))

    otro = df.copy()
    otro.iloc[-1, otro.columns.get_loc("SALARIO")] += 1  # una celda, en la última fila (MLflow solo mira 10.000)
    assert r.huella(otro) != huella
    texto = df.copy()
    texto.iloc[0, texto.columns.get_loc("CARGO")] = None  # un vacío en una categórica
    assert r.huella(texto) != huella
    assert r.huella(df.astype({"ESTRATO": "float64"})) != huella  # mismo valor, otro tipo
    assert r.huella(df.iloc[::-1]) != huella  # el orden de las filas cuenta
    assert r.huella(df.astype({"CARGO": object})) == r.huella(df.astype({"CARGO": "string"}))


def test_huella_del_contrato(tmp_path):
    contrato = tmp_path / "contrato.yaml"
    contrato.write_text("columnas: [a]\n", encoding="utf-8")
    antes = r.huella_contrato(contrato)
    contrato.write_text("columnas: [a, b]\n", encoding="utf-8")
    assert r.huella_contrato(contrato) != antes and len(r.huella_contrato()) == 12


@pytest.mark.integration
def test_registrar_datos_en_mlflow(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    df = GeneradorAsociados(n=120, seed=2).generar()
    r.configurar("prueba_datos")
    with r.corrida("con_datos", r.parametros_datos(df)) as run_id:
        huella = r.registrar_datos(df, "asociados_prueba", "eleccion", "GeneradorAsociados(seed=2)")
    corrida = mlflow.get_run(run_id)
    (entrada,) = corrida.inputs.dataset_inputs
    assert entrada.dataset.name == "asociados_prueba" and entrada.dataset.digest == huella == r.huella(df)
    assert [t.value for t in entrada.tags if t.key == "mlflow.data.context"] == ["eleccion"]
    assert '"num_rows": 120' in entrada.dataset.profile and "SALARIO" in entrada.dataset.schema
    assert "GeneradorAsociados(seed=2)" in entrada.dataset.source  # la fuente, no los datos
    assert corrida.data.params["datos_huella"] == huella and len(corrida.data.params["contrato_huella"]) == 12


def test_misma_huella_en_memoria_y_en_archivo(tmp_path, monkeypatch):
    """El CI genera los datos de desarrollo en memoria; en el PC se leen del parquet: deben dar la misma huella."""
    from proyecto_ml import campeon as c
    from proyecto_ml import config, generador

    monkeypatch.setitem(generador.DESARROLLO, "n", 300)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    memoria, fuente = c.datos_eleccion()
    assert fuente == "datos_desarrollo() en memoria"
    GeneradorAsociados(**generador.DESARROLLO).guardar(tmp_path, generador.NOMBRE_DESARROLLO)  # make datos-desarrollo
    archivo, fuente = c.datos_eleccion()
    assert fuente == f"{generador.NOMBRE_DESARROLLO}.parquet"
    assert r.huella(archivo) == r.huella(memoria)


def test_version_codigo_con_variable_y_sin_git(monkeypatch):
    import subprocess

    from proyecto_ml import registro as r

    monkeypatch.setenv("GIT_COMMIT", "0123456789abcdef0123")
    assert r.version_codigo()["git.commit"] == "0123456789ab"  # GIT_COMMIT manda (CI, Docker)
    monkeypatch.delenv("GIT_COMMIT")
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def sin_git(*args, **kw):
        raise OSError("git no está instalado")

    monkeypatch.setattr(subprocess, "run", sin_git)
    assert r.version_codigo() == {"git.commit": "desconocido", "git.sucio": "desconocido"}  # nunca falla
    assert r.contexto_reproducible()["git.commit"] == "desconocido"
