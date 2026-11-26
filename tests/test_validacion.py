"""Fuerza de la señal del generador, escenarios menos favorables y partición temporal."""

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import validacion as v
from proyecto_ml.generador import DESARROLLO, PRODUCTOS, GeneradorAsociados, datos_desarrollo


def test_fuerza_uno_no_cambia_nada():
    base = GeneradorAsociados(n=300, seed=5).generar()
    igual = GeneradorAsociados(n=300, seed=5, fuerza_senal=1.0).generar()
    pd.testing.assert_frame_equal(base, igual)


def test_fuerza_escala_la_diferencia_entre_retirados_y_activos():
    columna = "#SOLICITUD_RETIROPARCIALAHORROS"
    base, senal, extra = PRODUCTOS[columna]  # P(tener) en activos y en "retirados"; 1 + Poisson(extra) si tiene
    diferencia = {}
    for f in (1.0, 0.5, 0.0):
        df = GeneradorAsociados(n=8000, seed=5, fuerza_senal=f).generar()
        medias = df.groupby(a.objetivo_binario(df))[columna].mean()
        diferencia[f] = medias[1] - medias[0]
    esperada = (senal - base) * (1 + extra)  # con f = 1; escala con f
    assert diferencia[1.0] == pytest.approx(esperada, abs=0.12)
    assert diferencia[0.5] == pytest.approx(esperada / 2, abs=0.1)
    assert abs(diferencia[0.0]) < 0.05  # indistinguibles


def test_fuerza_negativa_no_se_acepta():
    with pytest.raises(ValueError, match="fuerza_senal"):
        GeneradorAsociados(fuerza_senal=-0.1)


def test_datos_desarrollo():
    df = datos_desarrollo()
    assert len(df) == DESARROLLO["n"]
    assert a.objetivo_binario(df).mean() == pytest.approx(DESARROLLO["tasa_retiro"], abs=0.02)


@pytest.fixture(scope="module")
def logistica():
    df = GeneradorAsociados(n=800, seed=1).generar()
    return a.modelos_supervisados()["logreg"].fit(a.preparar(df), a.objetivo_binario(df))


def test_escenarios_degradan_al_bajar_la_senal(logistica):
    resultados = v.evaluar_escenarios({"logreg": logistica}, n=600)
    assert list(resultados["escenario"]) == [e.nombre for e in v.ESCENARIOS]
    auc = resultados.set_index("escenario")["roc_auc"]
    assert auc.iloc[1] > auc.iloc[2] > auc.iloc[3]  # señal 100 % > 50 % > 25 %
    tasa = resultados.set_index("escenario")["base_rate"]
    assert tasa.iloc[0] == pytest.approx(0.30, abs=0.05) and tasa.iloc[1] == pytest.approx(0.10, abs=0.04)

    tabla = v.tabla_escenarios(resultados)
    assert list(tabla.index) == [e.nombre for e in v.ESCENARIOS]
    assert list(tabla.columns) == [("logreg", "roc_auc"), ("logreg", "precision"), ("logreg", "recall")]


def test_escenarios_usan_el_umbral_dado(logistica):
    sin = v.evaluar_escenarios({"logreg": logistica}, escenarios=v.ESCENARIOS[:1], n=400)
    bajo = v.evaluar_escenarios({"logreg": logistica}, {"logreg": 0.05}, escenarios=v.ESCENARIOS[:1], n=400)
    assert bajo["recall"].iloc[0] >= sin["recall"].iloc[0]


def test_particion_temporal_entrena_con_la_foto_vieja():
    df = GeneradorAsociados(n=1500, seed=2).generar()
    entrenamiento, prueba = v.particion_temporal(
        df, horizonte_meses=3, meses_prueba=3, corte=pd.Timestamp("2026-03-31")
    )
    # Corte 2026-03-31 (el del extracto real): la foto de prueba es el corte y la de entrenamiento, 3 meses antes.
    assert (entrenamiento[a.FECHA_OBSERVACION] == pd.Timestamp("2025-12-31")).all()
    assert (prueba[a.FECHA_OBSERVACION] == pd.Timestamp("2026-03-31")).all()
    assert len(entrenamiento) == len(prueba) == len(df)
    assert entrenamiento[a.TARGET].equals(prueba[a.TARGET])
    with pytest.raises(ValueError):
        v.particion_temporal(df, meses_prueba=0)


def test_con_un_solo_extracto_no_se_inventa_una_validacion_fuera_de_tiempo():
    df = GeneradorAsociados(n=1500, seed=5, tasa_retiro=0.3).generar()
    vieja, reciente = v.particion_temporal(df, horizonte_meses=3, meses_prueba=3)
    assert vieja["NROIDENTIFICACION"].equals(reciente["NROIDENTIFICACION"])
    assert vieja[a.CONTEOS_SIN_FECHA].equals(reciente[a.CONTEOS_SIN_FECHA])
    assert vieja[a.TARGET].equals(reciente[a.TARGET])


def test_desplazamiento_distingue_un_lote_distinto_y_no_uno_igual():
    referencia = GeneradorAsociados(n=400, seed=1, tasa_retiro=0.1).generar_extracto()
    igual = GeneradorAsociados(n=400, seed=2, tasa_retiro=0.1).generar_extracto()
    parecido = v.desplazamiento(referencia, igual, folds=3)
    assert parecido["roc_auc"] < 0.65  # mismo generador, otra semilla: casi indistinguibles
    assert set(parecido["por_variable"]) <= set(a.NUMERICAS + a.CATEGORICAS)
    assert not set(a.IDENTIFICADORES) & set(parecido["por_variable"])
    distinto = igual.assign(SALARIO=igual["SALARIO"] * 3)  # el mismo lote con otro nivel de salario
    d = v.desplazamiento(referencia, distinto, folds=3)
    assert d["roc_auc"] > 0.9 and next(iter(d["por_variable"])) == "SALARIO"  # la que más distingue
    assert d["roc_auc_comportamiento"] < 0.65  # el comportamiento no cambió
    distribuciones = v.comparar_distribuciones(referencia, distinto)
    assert distribuciones.loc["SALARIO", "lote_p99"] > distribuciones.loc["SALARIO", "ref_p99"]
    assert distribuciones.loc["SALARIO", "lote_fuera_ref_p01_p99"] > 0

    duplicado = pd.concat([referencia, referencia.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        v.desplazamiento(duplicado, igual, folds=3)
