"""Generador sintético alineado con el contrato definitivo."""

import pandas as pd

from proyecto_ml import contrato as k
from proyecto_ml.generador import DESARROLLO, GRUPOS, GeneradorAsociados, datos_desarrollo, diccionario


def test_esquema_completo_y_en_orden():
    df = GeneradorAsociados(n=50, seed=2).generar()
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
    df = GeneradorAsociados(n=50, seed=3).generar()
    assert df["NROIDENTIFICACION"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert df["NROIDENTIFICACION"].is_unique
    assert set(df["PRIMERAPELLIDO"]) <= {"PRUEBA", "FICTICIO", "EJEMPLO", "DEMO", "SINTETICO", "MUESTRA"}
    assert df["EMAIL"].str.endswith(".invalid").all()


def test_edades_y_antiguedades_coherentes_con_fechas():
    df = GeneradorAsociados(n=120, seed=4).generar()
    corte = pd.Timestamp("2026-10-31")
    edad = ((corte - pd.to_datetime(df["FECHANACIMIENTO"])).dt.days / 365.25).astype(int)
    assert (df["EDAD"] == edad).all()
    assert (pd.to_datetime(df["FECHAINGRESO"]) > pd.to_datetime(df["FECHANACIMIENTO"])).all()
    assert (pd.to_datetime(df["FECHAINGRESOLABORA"]) > pd.to_datetime(df["FECHANACIMIENTO"])).all()


def test_faltantes_no_tocan_identificador_ni_objetivo():
    gen = GeneradorAsociados(n=200, seed=8, faltantes={"SALARIO": 0.2, "ANTIGUEDAD_LA_MESES": 0.2})
    df = gen.generar()
    assert df[["NROIDENTIFICACION", "#RETIROS"]].notna().all().all()
    assert sum(gen.reporte_faltantes_.values()) > 0


def test_datos_desarrollo_reproducibles():
    a, b = datos_desarrollo(), datos_desarrollo()
    assert len(a) == DESARROLLO["n"]
    pd.testing.assert_frame_equal(a, b)
