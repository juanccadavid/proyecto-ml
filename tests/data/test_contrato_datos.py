"""Contrato v2 del extracto anonimizado definitivo."""

from __future__ import annotations

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import contrato as k
from proyecto_ml.generador import GRUPOS, GeneradorAsociados


@pytest.fixture
def contrato():
    return k.cargar()


@pytest.fixture
def lote():
    return GeneradorAsociados(n=80, seed=9, tasa_retiro=0.1).generar_extracto()


def test_esquema_definitivo_tiene_44_columnas_en_orden(contrato):
    esperadas = [columna for columnas in GRUPOS.values() for columna in columnas]
    assert len(esperadas) == 44
    assert contrato.nombres == esperadas
    assert contrato.version == "2.1.0"


def test_lote_sintetico_cumple_los_tres_niveles(lote, contrato):
    assert k.validar(lote, contrato) == []


@pytest.mark.parametrize("columna", ["NROIDENTIFICACION", "EDAD", "ANTIGUEDAD_F_MESES", "#RETIROS"])
def test_columnas_requeridas_no_admiten_vacios(lote, contrato, columna):
    alterado = lote.copy()
    alterado.loc[0, columna] = None
    assert any(columna in error for error in k.validar(alterado, contrato))


def test_identificador_anonimizado_es_unico(lote, contrato):
    alterado = lote.copy()
    alterado.loc[1, "NROIDENTIFICACION"] = alterado.loc[0, "NROIDENTIFICACION"]
    assert any("NROIDENTIFICACION" in error for error in k.validar(alterado, contrato))


@pytest.mark.parametrize(
    ("columna", "valor"),
    [("EDAD", 121), ("ANTIGUEDAD_F_MESES", -1), ("SALARIO", -1), ("#RETIROS", -1)],
)
def test_rangos_rechazan_valores_invalidos(lote, contrato, columna, valor):
    alterado = lote.copy()
    alterado.loc[0, columna] = valor
    assert any(columna in error for error in k.validar(alterado, contrato))


def test_retiros_admite_conteos_mayores_que_uno(lote, contrato):
    alterado = lote.copy()
    alterado.loc[0, "#RETIROS"] = 3
    assert k.validar(alterado, contrato) == []
    assert a.preparar(alterado).loc[0, "#RETIROS"] == 3


@pytest.mark.parametrize("columna", ["FECHAINGRESO", "FECHAINGRESOLABORA"])
def test_ingreso_debe_ser_posterior_al_nacimiento(lote, contrato, columna):
    alterado = lote.copy()
    alterado.loc[0, columna] = pd.to_datetime(alterado.loc[0, "FECHANACIMIENTO"]) - pd.Timedelta(days=1)
    assert any("posterior_nacimiento" in error for error in k.validar(alterado, contrato))


def test_contrato_ml_es_el_mismo_esquema_anonimizado(contrato):
    ml = k.para_ml(contrato)
    assert ml.nombres == contrato.nombres
    assert k.columnas_pii(contrato) == []


def test_sql_y_web_salen_del_mismo_contrato(contrato):
    sql = k.sql(contrato)
    web = k.js(contrato)
    for columna in ("NROIDENTIFICACION", "ANTIGUEDAD_F_MESES", "#SOLICITUDCREDITOS_ULT3MES"):
        assert columna.lower().replace("#", "num_") in sql.lower()
        assert columna in web
    assert '"#RETIROS"' in web
    assert '"min": 0' in web
