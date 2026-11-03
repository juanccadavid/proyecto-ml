"""El campeón con sus umbrales (campeon.ModeloZonas) y KMeans con la fila cruda."""

import joblib
import numpy as np
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import registro as r
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def datos():
    entrenamiento = GeneradorAsociados(n=300, seed=6, tasa_retiro=0.25).generar()
    nuevos = GeneradorAsociados(n=120, seed=7, tasa_retiro=0.25).generar()
    return entrenamiento, nuevos


@pytest.fixture(scope="module")
def logreg(datos):
    return a.modelos_supervisados()["logreg"].fit(*a.separar(datos[0]))


def test_predict_da_la_zona_desde_la_fila_cruda(datos, logreg):
    _, nuevos = datos
    modelo = c.ModeloZonas(logreg, umbral=0.3, umbral_alto=0.7)
    puntaje = a.puntaje(logreg, nuevos)
    esperado = [c.zona(p, 0.3, 0.7) for p in puntaje]
    assert modelo.predict(nuevos).tolist() == esperado
    assert modelo.predict(a.preparar(nuevos)).tolist() == esperado  # también con la fila ya preparada
    assert set(esperado) <= set(c.ZONAS)
    assert (modelo.alerta(nuevos) == (puntaje >= 0.3)).all()
    assert np.allclose(a.puntaje(modelo, nuevos), puntaje)  # predict_proba se delega


def test_sin_umbral_alto_no_hay_zona_se_retira(datos, logreg):
    zonas = c.ModeloZonas(logreg, umbral=0.0, umbral_alto=None).predict(datos[1])
    assert "se retira" not in zonas and set(zonas) == {"zona gris"}


def test_ridge_expone_decision_function_y_no_predict_proba(datos):
    ridge = a.modelo_ridge().fit(*a.separar(datos[0]))
    modelo = c.ModeloZonas(ridge, umbral=0.0, umbral_alto=None)
    assert not hasattr(modelo, "predict_proba") and hasattr(modelo, "decision_function")
    assert modelo.metodo == "decision_function"


def test_guardar_y_cargar(tmp_path, datos):
    entrenamiento, nuevos = datos
    elegido = c.elegir(entrenamiento, folds=3, repeticiones=1)
    c.guardar(elegido, tmp_path)
    modelo, agrupador, info = c.cargar(tmp_path)
    assert isinstance(modelo, c.ModeloZonas) and modelo.umbral == info["umbral"]
    assert modelo.predict(nuevos).tolist() == elegido.zona(nuevos)
    # un campeón guardado antes (solo el pipeline) se envuelve con los umbrales de campeon.json
    joblib.dump(elegido.modelo, tmp_path / c.ARCHIVO_MODELO)
    viejo, _, _ = c.cargar(tmp_path)
    assert isinstance(viejo, c.ModeloZonas) and viejo.predict(nuevos).tolist() == elegido.zona(nuevos)


def test_kmeans_con_la_fila_cruda_o_preparada(datos):
    entrenamiento, nuevos = datos
    agrupador = a.AgrupadorAbandono().fit(entrenamiento)
    assert [n for n, _ in agrupador.pipeline.steps][:2] == ["preparar", "comportamiento"]
    assert (agrupador.predict(nuevos) == agrupador.grupos(a.preparar(nuevos))).all()
    assert set(agrupador.perfil_.columns) == set(a.COMPORTAMIENTO)


@pytest.mark.integration
def test_mlflow_devuelve_la_zona(tmp_path, monkeypatch, datos, logreg):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    _, nuevos = datos
    modelo = c.ModeloZonas(logreg, umbral=0.3, umbral_alto=0.7)
    r.configurar("prueba_zonas")
    with r.corrida("campeon"):
        info = r.registrar_modelo(modelo, registrar_como="abandono_prueba")
    cargado = mlflow.pyfunc.load_model(f"models:/abandono_prueba/{info.registered_model_version}")
    assert list(cargado.predict(nuevos)) == modelo.predict(nuevos).tolist()  # fila cruda -> zona
