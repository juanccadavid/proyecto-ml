"""Segmentación no supervisada (KMeans y DBSCAN, mapa) y predicción con el campeón guardado (predict.py)."""

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import predict as p
from proyecto_ml import segmentacion as s
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def datos():
    return a.foto(GeneradorAsociados(n=600, seed=5, tasa_retiro=0.2).generar(), 3)


@pytest.fixture(scope="module")
def segmentacion(datos):
    return s.Segmentacion().fit(datos)


def test_kmeans_elige_k_por_silueta_y_numera_por_senal(datos, segmentacion):
    assert segmentacion.k_ == int(segmentacion.tabla_k_["silueta"].idxmax())
    assert list(segmentacion.tabla_k_.index) == list(s.RANGO_K)
    segmento = segmentacion.kmeans(datos)
    assert set(segmento) == set(range(1, segmentacion.k_ + 1))
    perfil = s.perfil(datos, segmento)
    senal = sum(perfil[v] * peso for v, peso in a.SENAL_RIESGO.items())
    assert senal.idxmax() == 1  # el segmento 1 es el de más señales de salida (sin mirar el retiro)
    assert perfil["asociados"].sum() == len(datos) and perfil["pct"].sum() == pytest.approx(1)


def test_dbscan_predice_igual_que_al_entrenar_y_marca_atipicos(datos, segmentacion):
    assert (segmentacion.dbscan(datos) == segmentacion.dbscan_.labels_).all()
    raro = a.preparar(datos.head(1)).copy()
    raro[s.VARIABLES] = 25.0  # comportamiento que nadie tiene: fuera de toda zona densa
    assert segmentacion.dbscan(raro)[0] == s.ATIPICO
    assert segmentacion.eps_ == pytest.approx(s.eps_por_vecinos(segmentacion.transform(datos), 2 * len(s.VARIABLES)))


def test_los_atipicos_concentran_el_retiro(datos, segmentacion):
    tabla = s.comparar_con_retiro(datos, segmentacion)
    assert list(tabla.index) == ["KMeans: segmento 1", "DBSCAN: atípicos"]
    assert tabla.loc["DBSCAN: atípicos", "tasa_retiro"] > datos[a.TARGET].mean()  # validación, sin haberlo visto
    assert ((tabla["recall"] >= 0) & (tabla["recall"] <= 1)).all()


def test_k_fijo_y_mapa(datos):
    seg = s.Segmentacion(k=4).fit(datos)
    assert seg.k_ == 4 and seg.tabla_k_ is None
    plano = seg.plano(datos)
    assert plano.shape == (len(datos), 2) and np.isfinite(plano).all()
    fig = s.figura_mapa(seg, datos, asociado=a.preparar(datos.head(1)))
    botones = fig.layout.updatemenus[0].buttons
    assert [b.label.split(" ")[0] for b in botones] == ["KMeans", "DBSCAN"]
    for boton in botones:  # cada botón deja visible su método y la estrella del asociado
        assert len(boton.args[0]["visible"]) == len(fig.data) and boton.args[0]["visible"][-1]
    assert fig.data[-1].name == "este asociado"
    assert len(s.figura_k(s.tabla_k(seg.transform(datos), range(2, 4)), 2).data) == 2


def test_contorno_cerrado():
    puntos = np.random.default_rng(0).normal(size=(50, 2))
    borde = s._contorno(puntos)
    assert (borde[0] == borde[-1]).all() and len(borde) >= 4
    assert len(s._contorno(np.array([[0.0, 0.0], [1.0, 1.0]]))) == 3  # sin área: no falla


# ---------------------------------------------------------------- predict.py


def test_predecir_con_el_campeon_guardado(tmp_path, datos):
    c.guardar(c.elegir(datos, folds=3, repeticiones=1), tmp_path)
    nuevos = GeneradorAsociados(n=40, seed=8).generar()
    salida = p.predecir(nuevos, tmp_path)
    assert list(salida.columns) == [*p.IDENTIFICADORES, *p.COLUMNAS]
    assert set(salida["zona"]) <= set(c.ZONAS) and set(salida["grupo_kmeans"]) <= {"riesgo", "estable"}
    modelo, _, _ = c.cargar(tmp_path)
    assert np.allclose(salida["puntaje"], a.puntaje(modelo, nuevos.drop(columns=a.TARGET)))

    archivo = tmp_path / "nuevos.parquet"
    nuevos.to_parquet(archivo)
    p.main(["--datos", str(archivo), "--salida", str(tmp_path / "pred.csv"), "--modelos", str(tmp_path)])
    assert len(pd.read_csv(tmp_path / "pred.csv")) == len(nuevos)


def test_predecir_sin_campeon(tmp_path):
    with pytest.raises(FileNotFoundError, match="make campeon"):
        p.predecir(GeneradorAsociados(n=5, seed=1).generar(), tmp_path)
