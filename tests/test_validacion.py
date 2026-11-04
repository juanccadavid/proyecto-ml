"""Fuerza de la señal del generador, escenarios menos favorables y partición temporal."""

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import validacion as v
from proyecto_ml.generador import DESARROLLO, GeneradorAsociados, datos_desarrollo


def test_fuerza_uno_no_cambia_nada():
    base = GeneradorAsociados(n=300, seed=5).generar()
    igual = GeneradorAsociados(n=300, seed=5, fuerza_senal=1.0).generar()
    pd.testing.assert_frame_equal(base, igual)


def test_fuerza_escala_la_diferencia_entre_retirados_y_activos():
    diferencia = {}
    for f in (1.0, 0.5, 0.0):
        df = GeneradorAsociados(n=4000, seed=5, fuerza_senal=f).generar()
        medias = df.groupby(a.TARGET)["#INTENCIONRETIROS_ULT3MESES"].mean()
        diferencia[f] = medias[1] - medias[0]
    assert diferencia[1.0] == pytest.approx(1.3, abs=0.15)  # 1.5 - 0.2 en el generador
    assert diferencia[0.5] == pytest.approx(0.65, abs=0.15)
    assert abs(diferencia[0.0]) < 0.1  # indistinguibles


def test_fuerza_negativa_no_se_acepta():
    with pytest.raises(ValueError, match="fuerza_senal"):
        GeneradorAsociados(fuerza_senal=-0.1)


def test_datos_desarrollo():
    df = datos_desarrollo()
    assert len(df) == DESARROLLO["n"]
    assert df[a.TARGET].mean() == pytest.approx(DESARROLLO["tasa_retiro"], abs=0.02)


@pytest.fixture(scope="module")
def logistica():
    df = GeneradorAsociados(n=800, seed=1).generar()
    return a.modelos_supervisados()["logreg"].fit(a.preparar(df), df[a.TARGET])


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
    entrenamiento, prueba = v.particion_temporal(df, horizonte_meses=3, meses_prueba=3)
    assert (entrenamiento[a.FECHA_OBSERVACION] == pd.Timestamp("2026-04-30")).all()
    assert (prueba[a.FECHA_OBSERVACION] == pd.Timestamp("2026-07-31")).all()
    # En la foto vieja, quien se retira después de su horizonte (2026-07-31) cuenta como activo.
    retiro = pd.to_datetime(entrenamiento["FECHARETIRO"])
    tardios = retiro > pd.Timestamp("2026-07-31")
    assert tardios.any() and (entrenamiento.loc[tardios, a.TARGET] == 0).all()
    assert entrenamiento[a.TARGET].sum() > 0 and prueba[a.TARGET].sum() > 0
    with pytest.raises(ValueError):
        v.particion_temporal(df, meses_prueba=0)


def test_con_un_solo_extracto_las_dos_fotos_son_las_mismas_personas():
    """Lo que documenta el README ("prueba de estrés, no validación fuera de tiempo"): con un solo extracto, los
    retiros de la foto reciente son asociados que en la vieja cuentan como activos, con los mismos conteos."""
    df = GeneradorAsociados(n=1500, seed=5, tasa_retiro=0.3).generar()
    vieja, reciente = v.particion_temporal(df, horizonte_meses=3, meses_prueba=3)
    assert set(reciente["CEDULA"]) <= set(vieja["CEDULA"])
    retiros = reciente.loc[reciente[a.TARGET] == 1, "CEDULA"]
    en_vieja = vieja.set_index("CEDULA").loc[retiros]
    assert len(retiros) and (en_vieja[a.TARGET] == 0).all()  # en la vieja, activos
    conteos = reciente.set_index("CEDULA").loc[retiros, a.CONTEOS_SIN_FECHA]
    assert conteos.equals(en_vieja[a.CONTEOS_SIN_FECHA])  # mismos números en las dos fotos
