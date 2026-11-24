"""Auditoría del pipeline con el esquema v2."""

from proyecto_ml import abandono as a
from proyecto_ml import contrato as k
from proyecto_ml.generador import GeneradorAsociados


def test_el_contrato_es_obligatorio():
    df = GeneradorAsociados(n=80, seed=3).generar_extracto().drop(columns="EDAD")
    errores = k.validar(df, k.cargar())
    assert errores and any("EDAD" in error for error in errores)


def test_identificadores_y_fuga_no_son_predictores():
    entradas = set(a.NUMERICAS + a.CATEGORICAS)
    assert entradas.isdisjoint(a.IDENTIFICADORES + a.FUGA)


def test_campos_de_comportamiento_existen_en_el_contrato():
    assert set(a.COMPORTAMIENTO) <= set(k.cargar().nombres)
