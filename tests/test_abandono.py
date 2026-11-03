"""Pruebas de los modelos de abandono: supervisados (Ridge, logística, bagging, boosting) y KMeans."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from proyecto_ml import abandono as a
from proyecto_ml.generador import FALTANTES, GeneradorAsociados


@pytest.fixture(scope="module")
def entrenamiento() -> pd.DataFrame:
    return GeneradorAsociados(n=400, seed=1).generar()


@pytest.fixture(scope="module")
def nuevos() -> pd.DataFrame:
    """Asociados que los modelos nunca ven al entrenar."""
    return GeneradorAsociados(n=400, seed=2).generar()


def test_variables_sin_fuga_ni_datos_personales(entrenamiento):
    X = a.preparar(entrenamiento)
    assert list(X.columns) == a.NUMERICAS + a.CATEGORICAS
    assert not set(X.columns) & set(a.FUGA)
    assert not set(X.columns) & {"CEDULA", "NOMBRES", "EMAIL", "TELEFONO1", "NUMEROCUENTA", "IDASOCIADOS"}
    assert X["EDAD"].between(18, 70).all()


def test_columnas_de_fuga_no_cambian_las_predicciones(entrenamiento, nuevos):
    ridge = a.modelo_ridge().fit(a.preparar(entrenamiento), entrenamiento[a.TARGET])
    agrupador = a.AgrupadorAbandono().fit(entrenamiento)
    alterado = nuevos.copy()
    alterado["ESTADO"], alterado["MOTIVORETIRO"], alterado["FECHARETIRO"] = "ACTIVO", "", pd.NaT
    assert np.array_equal(ridge.predict(a.preparar(nuevos)), ridge.predict(a.preparar(alterado)))
    assert np.array_equal(agrupador.predict(nuevos), agrupador.predict(alterado))


def test_ridge_generaliza_a_asociados_nuevos(entrenamiento, nuevos):
    ridge = a.modelo_ridge().fit(a.preparar(entrenamiento), entrenamiento[a.TARGET])
    m = a.metricas_supervisado(ridge, nuevos)
    assert m["roc_auc"] > 0.8
    assert m["recall"] > 0.6
    assert len(a.coeficientes(ridge, 5)) == 5


@pytest.fixture(scope="module")
def supervisados(entrenamiento) -> dict:
    X, y = a.preparar(entrenamiento), entrenamiento[a.TARGET]
    return {nombre: modelo.fit(X, y) for nombre, modelo in a.modelos_supervisados().items()}


def test_hay_un_modelo_de_bagging_y_uno_de_boosting(supervisados):
    assert list(supervisados) == list(a.TITULOS)
    assert isinstance(supervisados["random_forest"].named_steps["model"], RandomForestClassifier)  # bagging
    assert isinstance(supervisados["hist_gb"].named_steps["model"], HistGradientBoostingClassifier)  # boosting


@pytest.mark.parametrize("nombre", list(a.TITULOS))
def test_supervisados_generalizan_a_asociados_nuevos(supervisados, nuevos, nombre):
    modelo = supervisados[nombre]
    m = a.metricas_supervisado(modelo, nuevos)
    assert m["roc_auc"] > 0.8
    assert m["recall"] > 0.6
    # El umbral de cada tipo de puntaje coincide con lo que decide predict().
    X = a.preparar(nuevos)
    decide = (a.puntaje(modelo, X) >= a.UMBRAL[a.metodo_puntaje(modelo)]).astype(int)
    assert (decide == modelo.predict(X)).mean() > 0.99  # empates exactos en el umbral pueden diferir


def test_kmeans_se_entrena_sin_el_objetivo(entrenamiento):
    sin_objetivo = entrenamiento.drop(columns=a.FUGA)
    agrupador = a.AgrupadorAbandono().fit(sin_objetivo)
    assert set(agrupador.predict(sin_objetivo)) == {0, 1}
    assert set(agrupador.perfil_.index) == {"riesgo", "estable"}


def test_kmeans_separa_a_los_que_abandonan(entrenamiento, nuevos):
    agrupador = a.AgrupadorAbandono().fit(entrenamiento.drop(columns=a.TARGET))
    m = a.metricas_agrupacion(agrupador, nuevos)
    assert m["recall"] > 0.7  # la mayoría de los que se van quedan en el grupo de riesgo
    assert m["ari"] > 0.3  # mucho mejor que agrupar al azar (ARI ≈ 0)
    perfil = agrupador.perfil_
    assert perfil.loc["riesgo", "#INTENCIONRETIROS_ULT3MESES"] > perfil.loc["estable", "#INTENCIONRETIROS_ULT3MESES"]


def test_modelos_toleran_datos_faltantes():
    """Con vacíos (como llegan los datos reales) el pipeline imputa con lo aprendido en train."""
    sucio = GeneradorAsociados(n=600, seed=21, faltantes=FALTANTES).generar()
    nuevos_sucios = GeneradorAsociados(n=600, seed=22, faltantes=FALTANTES).generar()
    X = a.preparar(sucio)
    assert X.isna().any().any()  # hay vacíos de verdad...
    assert not X[a.CATEGORICAS].isin(["nan", "None", "<NA>"]).any().any()  # ...y no texto "nan"

    for modelo in a.modelos_supervisados().values():
        assert a.metricas_supervisado(modelo.fit(X, sucio[a.TARGET]), nuevos_sucios)["roc_auc"] > 0.8
    agrupador = a.AgrupadorAbandono().fit(sucio)
    assert a.metricas_agrupacion(agrupador, nuevos_sucios)["recall"] > 0.7


def test_figura_matrices_confusion(entrenamiento, nuevos, supervisados):
    agrupador = a.AgrupadorAbandono().fit(entrenamiento)
    fig = a.figura_matrices_confusion(supervisados, agrupador, nuevos)
    ejes = fig.get_axes()  # 5 paneles en una cuadrícula de 2x3: el sexto se quita
    assert [ax.get_title() for ax in ejes] == [*a.TITULOS.values(), "KMeans (sin objetivo)"]
    for ax in ejes:
        conteos = [int(t.get_text().split("\n")[0]) for t in ax.texts]
        assert sum(conteos) == len(nuevos)  # las 4 celdas suman todos los asociados


def test_tabla_predicciones_como_predict_model(supervisados, nuevos):
    modelos = {n: supervisados[n] for n in ("ridge", "logreg")}
    tabla = a.tabla_predicciones(modelos, nuevos)

    assert list(tabla.columns) == [
        *a.NUMERICAS,
        *a.CATEGORICAS,
        a.TARGET,
        "ridge_label",
        "ridge_score",
        "logreg_label",
        "logreg_score",
    ]
    assert len(tabla) == len(nuevos)
    assert tabla["logreg_score"].between(0, 1).all()  # probabilidad de abandono
    # Ridge no da probabilidad: su score es la distancia a la frontera, y > 0 es exactamente "abandona".
    assert ((tabla["ridge_score"] > 0).astype(int) == tabla["ridge_label"]).all()
    # Sin la columna objetivo (asociados actuales, sin desenlace) también funciona.
    assert a.TARGET not in a.tabla_predicciones(modelos, nuevos.drop(columns=a.TARGET))


def test_guardar_modelo_en_joblib_por_partes(supervisados, nuevos, tmp_path):
    import joblib

    rutas = a.guardar_modelo(supervisados["ridge"], tmp_path, "ridge")
    assert {r.name for r in rutas.values()} == {"ridge.joblib", "ridge_preprocesamiento.joblib", "ridge_modelo.joblib"}

    X = a.preparar(nuevos)
    pipeline = joblib.load(rutas["pipeline"])
    preprocesamiento, modelo = joblib.load(rutas["preprocesamiento"]), joblib.load(rutas["modelo"])
    # Por partes da exactamente lo mismo que el pipeline completo.
    assert np.array_equal(modelo.predict(preprocesamiento.transform(X)), pipeline.predict(X))
    assert type(modelo).__name__ == "RidgeClassifier"


def test_recorte_opcional_solo_en_columnas_continuas(entrenamiento, nuevos):
    X, y = a.preparar(entrenamiento), entrenamiento[a.TARGET]
    for nombre, modelo in a.modelos_supervisados(clip_outliers="iqr").items():
        modelo.fit(X, y)
        ramas = modelo.named_steps["preprocess"].named_transformers_
        assert list(ramas["clip"].feature_names_in_) == a.COLUMNAS_RECORTE, nombre
        assert "outliers" not in ramas["num"].named_steps  # los conteos de comportamiento no se tocan
        assert a.metricas_supervisado(modelo, nuevos)["roc_auc"] > 0.8, nombre
    # Por defecto, sin recorte (no cambia nada de lo que ya había).
    sin = a.modelos_supervisados()["ridge"].fit(X, y).named_steps["preprocess"].named_transformers_
    assert "clip" not in sin
