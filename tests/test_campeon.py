"""Modelo campeón: umbrales fuera de partición, tres zonas, elección por PR-AUC y archivos para el simulador."""

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def foto():
    return a.foto(GeneradorAsociados(n=900, seed=1).generar(), 3)


@pytest.fixture(scope="module")
def elegido(foto):
    return c.elegir(foto, repeticiones=2, info={"datos": "prueba", "horizonte_meses": 3})


def test_umbrales_f2_y_precision_minima():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    puntaje = np.r_[rng.beta(2, 8, 900), rng.beta(6, 3, 100)]  # los retiros puntúan más alto
    u = c.elegir_umbrales(y, puntaje, beta=2, precision_min=0.8)
    pred = puntaje >= u["umbral"]
    assert u["recall"] == pytest.approx(pred[y == 1].mean())
    assert u["precision"] == pytest.approx(y[pred].mean())
    assert u["umbral_alto"] >= u["umbral"]
    alto = puntaje >= u["umbral_alto"]
    assert y[alto].mean() >= 0.8 and u["recall_alto"] <= u["recall"]
    # F2 premia el recall: su umbral es más bajo (o igual) que el de F1
    assert u["umbral"] <= c.elegir_umbrales(y, puntaje, beta=1)["umbral"]


def test_sin_zona_segura_si_ningun_umbral_llega_a_la_precision():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    u = c.elegir_umbrales(y, rng.random(400), precision_min=0.99)  # puntaje al azar
    assert u["umbral_alto"] is None


@pytest.mark.parametrize(
    ("puntaje", "esperada"), [(0.95, "se retira"), (0.5, "zona gris"), (0.1, "se queda"), (0.3, "zona gris")]
)
def test_zonas(puntaje, esperada):
    assert c.zona(puntaje, umbral=0.3, umbral_alto=0.8) == esperada


def test_sin_umbral_alto_todo_lo_alertado_es_zona_gris():
    assert c.zona(0.99, 0.3, None) == "zona gris"


def test_elige_el_de_mejor_pr_auc_y_lo_deja_entrenado(elegido, foto):
    tabla = elegido.tabla()
    assert list(tabla.index)[0] == elegido.nombre == tabla[c.CRITERIO].idxmax()
    assert set(tabla.index) == set(a.TITULOS)
    assert elegido.candidatos[elegido.nombre].puntajes.shape == (2, len(foto))
    zonas = elegido.zona(a.preparar(foto))
    assert set(zonas) <= set(c.ZONAS) and "se queda" in zonas
    # KMeans: el grupo de riesgo se retira más que el estable (sin haber visto el objetivo)
    assert elegido.grupos["riesgo"]["tasa_retiro"] > elegido.grupos["estable"]["tasa_retiro"]
    assert elegido.info["horizonte_meses"] == 3 and elegido.info["filas"] == len(foto)


def test_guardar_y_cargar(elegido, foto, tmp_path):
    rutas = c.guardar(elegido, tmp_path, extra={"nota": "prueba"})
    assert {r.name for r in rutas.values()} == {c.ARCHIVO_MODELO, c.ARCHIVO_GRUPOS, c.ARCHIVO_INFO}
    modelo, agrupador, info = c.cargar(tmp_path)
    X = a.preparar(foto)
    assert np.allclose(a.puntaje(modelo, X), a.puntaje(elegido.modelo, X))
    assert (agrupador.grupos(X) == elegido.agrupador.predict(foto)).all()
    assert info["modelo"] == elegido.nombre and info["umbral"] == elegido.umbral and info["nota"] == "prueba"
    assert set(info["grupos"]) == {"riesgo", "estable"}
    assert c.cargar(tmp_path / "vacia") is None


def test_figura_de_umbrales(elegido):
    fig = c.figura_umbrales(elegido)
    assert [t.name for t in fig.data] == ["Precisión", "Recall"]


def test_datos_eleccion_con_ruta(tmp_path):
    df = GeneradorAsociados(n=60, seed=3).generar()
    ruta = tmp_path / "asociados.parquet"
    df.to_parquet(ruta, index=False)
    datos, fuente = c.datos_eleccion(ruta)
    assert fuente == "asociados.parquet" and len(datos) == 60
    assert isinstance(datos, pd.DataFrame)
