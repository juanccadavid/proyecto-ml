"""Generación del contrato para Pandera, SQL y web."""

import pandas as pd

from proyecto_ml import contrato as k
from proyecto_ml.generador import GeneradorAsociados


def test_contrato_cubre_el_esquema_del_generador():
    df = GeneradorAsociados(n=50, seed=42).generar()
    assert list(df.columns) == k.cargar().nombres


def test_esquema_pandera_con_los_tres_niveles():
    contrato = k.cargar()
    df = GeneradorAsociados(n=50, seed=42).generar()
    assert k.validar(df, contrato) == []
    assert len(contrato.reglas) == 2
    assert len(contrato.distribucion) == 5


def test_nombres_en_la_bd():
    contrato = k.cargar()
    db = {c.nombre: c.db for c in contrato.columnas}
    assert db["NROIDENTIFICACION"] == "nroidentificacion"
    assert db["#SOLICITUDCREDITOS_ULT3MES"] == "num_solicitudcreditos_ult3mes"
    assert db["#RETIROS"] == "num_retiros"


def test_registros_para_la_bd_y_vuelta():
    contrato = k.cargar()
    df = GeneradorAsociados(n=4, seed=2).generar()
    registros = k.a_registros_db(df, contrato, origen="script")
    assert registros[0]["origen"] == "script"
    vuelta = k.desde_registros_db(registros, contrato)
    assert list(vuelta.columns) == contrato.nombres
    assert len(vuelta) == len(df)


def test_sql_incluye_restricciones_y_seguridad():
    sql = k.sql(k.cargar()).lower()
    assert "nroidentificacion text not null unique" in sql
    assert "constraint ingreso_fondo_posterior_nacimiento" in sql
    assert "enable row level security" in sql
    assert "create policy equipo_inserta" in sql


def test_limpiar_extracto_sigue_cumpliendo_el_contrato():
    contrato = k.cargar()
    df = GeneradorAsociados(n=50, seed=4).generar()
    limpio = k.limpiar_extracto(df)
    assert k.validar(limpio, contrato) == []
    assert pd.api.types.is_datetime64_any_dtype(limpio["FECHANACIMIENTO"])
