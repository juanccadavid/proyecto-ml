"""Pruebas de los modelos de abandono: Ridge (supervisado) y KMeans (sin objetivo)."""

import numpy as np
import pandas as pd
import pytest

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
    m = a.metricas_ridge(ridge, nuevos)
    assert m["roc_auc"] > 0.8
    assert m["recall"] > 0.6
    assert len(a.coeficientes(ridge, 5)) == 5


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

    ridge = a.modelo_ridge().fit(X, sucio[a.TARGET])
    assert a.metricas_ridge(ridge, nuevos_sucios)["roc_auc"] > 0.8
    agrupador = a.AgrupadorAbandono().fit(sucio)
    assert a.metricas_agrupacion(agrupador, nuevos_sucios)["recall"] > 0.7
