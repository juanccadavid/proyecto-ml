"""Generador sintético alineado con el contrato definitivo."""

from pathlib import Path

import pandas as pd

from proyecto_ml import contrato as k
from proyecto_ml.generador import (
    DESARROLLO,
    FALTANTES,
    FECHA_CORTE,
    GRUPOS,
    TARGET_SINTETICO,
    GeneradorAsociados,
    asociados_nuevos,
    datos_desarrollo,
    diccionario,
)


def test_esquema_completo_y_en_orden():
    df = GeneradorAsociados(n=50, seed=2).generar_extracto()
    esperadas = [c for columnas in GRUPOS.values() for c in columnas]
    assert list(df.columns) == esperadas
    assert len(df.columns) == 44
    assert k.validar(df, k.cargar()) == []


def test_diccionario_cubre_todas_las_columnas():
    tabla = diccionario()
    assert tabla["COLUMNA"].tolist() == [c for columnas in GRUPOS.values() for c in columnas]
    assert tabla["GRUPO"].value_counts().to_dict() == {
        "SOCIODEMOGRAFICO": 29,
        "LABORAL": 7,
        "PRODUCTOS": 7,
        "RETIRO": 1,
    }


def test_datos_evidentemente_ficticios_y_seudonimos_unicos():
    df = GeneradorAsociados(n=50, seed=3).generar_extracto()
    assert df["NROIDENTIFICACION"].str.fullmatch(r"[0-9a-f]{10}").all()  # como el identificador del extracto
    assert df["NROIDENTIFICACION"].is_unique
    assert set(df["PRIMERAPELLIDO"]) <= {"PRUEBA", "FICTICIO", "EJEMPLO", "DEMO", "SINTETICO", "MUESTRA"}
    assert df["EMAIL"].str.endswith(".invalid").all()


def test_edades_y_antiguedades_coherentes_con_fechas():
    df = GeneradorAsociados(n=120, seed=4).generar_extracto()
    corte = FECHA_CORTE
    edad = ((corte - pd.to_datetime(df["FECHANACIMIENTO"])).dt.days / 365.25).astype(int)
    assert (df["EDAD"] == edad).all()
    assert (pd.to_datetime(df["FECHAINGRESO"]) > pd.to_datetime(df["FECHANACIMIENTO"])).all()
    assert (pd.to_datetime(df["FECHAINGRESOLABORA"]) > pd.to_datetime(df["FECHANACIMIENTO"])).all()


def test_faltantes_no_tocan_identificador_ni_conteo_historico():
    gen = GeneradorAsociados(n=200, seed=8, faltantes={"SALARIO": 0.2, "ANTIGUEDAD_LA_MESES": 0.2})
    df = gen.generar()
    assert df[["NROIDENTIFICACION", "#RETIROS"]].notna().all().all()
    assert sum(gen.reporte_faltantes_.values()) > 0


def test_datos_desarrollo_reproducibles():
    a, b = datos_desarrollo(), datos_desarrollo()
    assert len(a) == DESARROLLO["n"]
    assert TARGET_SINTETICO in a
    assert set(a[TARGET_SINTETICO].unique()) == {0, 1}
    assert not a[TARGET_SINTETICO].equals((a["#RETIROS"] > 0).astype(int))
    pd.testing.assert_frame_equal(a, b)


def test_asociados_nuevos_tienen_la_prevalencia_de_desarrollo():
    """Validar con otra tasa de retiro cambia la prevalencia: precisión, PR-AUC y umbral dejan de ser comparables."""
    nuevos, desarrollo = asociados_nuevos(5000, seed=2026), datos_desarrollo()
    assert abs(nuevos[TARGET_SINTETICO].mean() - desarrollo[TARGET_SINTETICO].mean()) < 0.02
    assert abs(nuevos[TARGET_SINTETICO].mean() - DESARROLLO["tasa_retiro"]) < 0.02
    assert asociados_nuevos(300, seed=2026, faltantes=FALTANTES).isna().any().any()


def test_los_scripts_validan_con_nuevos_de_la_misma_prevalencia():
    """Antes abandono, afinar, optuna y pycaret entrenaban con 10 % de retiros y validaban con 27 %."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    nombres = ["modelo_abandono", "afinar_modelos", "afinar_optuna", "comparar_pycaret", "elegir_campeon"]
    for nombre in [*nombres, "regresion_salario"]:
        codigo = (scripts / f"{nombre}.py").read_text(encoding="utf-8")
        assert "asociados_nuevos(" in codigo, nombre
        assert "GeneradorAsociados(" not in codigo, nombre
