"""Pruebas de los modelos de abandono: Ridge (supervisado) y KMeans (sin objetivo)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml.config import ROOT_DIR

_spec = importlib.util.spec_from_file_location("gen", ROOT_DIR / "scripts" / "generar_asociados_ejemplo.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(scope="module")
def entrenamiento() -> pd.DataFrame:
    return gen.make_asociados(n=400, seed=1)


@pytest.fixture(scope="module")
def nuevos() -> pd.DataFrame:
    """Asociados que los modelos nunca ven al entrenar."""
    return gen.make_asociados(n=400, seed=2)


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
