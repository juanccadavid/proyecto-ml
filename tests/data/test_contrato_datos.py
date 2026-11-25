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
    assert contrato.version == "2.2.0"


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


def test_ingreso_al_fondo_debe_ser_posterior_al_nacimiento(lote, contrato):
    alterado = lote.copy()
    alterado.loc[0, "FECHAINGRESO"] = pd.to_datetime(alterado.loc[0, "FECHANACIMIENTO"]) - pd.Timedelta(days=1)
    assert any("posterior_nacimiento" in error for error in k.validar(alterado, contrato))


def test_ingreso_laboral_imposible_es_advertencia_y_no_bloquea(lote, contrato):
    # La fecha laboral la reporta la empresa y el extracto real trae algunas imposibles: se avisa cuántas, sin
    # bloquear, porque el modelo no la usa (usa la antigüedad, y la vacía si es imposible).
    alterado = lote.copy()
    alterado.loc[0, "FECHAINGRESOLABORA"] = pd.to_datetime(alterado.loc[0, "FECHANACIMIENTO"]) - pd.Timedelta(days=1)
    alterado.loc[1, "FECHAINGRESOLABORA"] = pd.Timestamp("1889-02-01")
    assert k.validar(alterado, contrato) == []
    avisos = k.advertencias(alterado, contrato)
    assert any("ingreso_laboral_posterior_nacimiento: 2 fila(s)" in aviso for aviso in avisos)
    assert any("ingreso_laboral_desde_1950: 1 fila(s)" in aviso for aviso in avisos)
    assert "ingreso_laboral" not in k.sql(contrato)  # la BD tolera lo mismo que el contrato


def test_extracto_real_con_alias_codigos_numericos_y_espacios(lote, contrato):
    # Así llega DataAnonimizada.xlsx: nombres con tildes o con errata, códigos que Excel lee como números,
    # códigos con espacios al final y estrato en blanco.
    real = lote.rename(
        columns={
            "EDAD": "EDAD_AÑOS",
            "ANTIGUEDAD_F_MESES": "ANTIGÜEDAD_F_MESES",
            "ANTIGUEDAD_LA_MESES": "ANTIGÜEDAD_LA_MESES",
            "FECHAINGRESOLABORA": "FECHAINGRESALABORA",
        }
    ).astype({"TIPOCONTRATO": object, "ESTRATO": object, "ESTADOCIVIL": object})
    real.loc[0, "TIPOCONTRATO"], real.loc[1, "TIPOCONTRATO"] = 2, "O  "
    real.loc[0, "ESTRATO"], real.loc[1, "ESTRATO"] = 1.0, "  "
    real.loc[0, "ESTADOCIVIL"] = "S "
    assert k.validar(real, contrato) == []
    limpio = k.limpiar_extracto(real)
    assert limpio.columns.tolist() == contrato.nombres
    assert limpio.loc[[0, 1], "TIPOCONTRATO"].tolist() == ["2", "O"]
    assert limpio.loc[0, "ESTRATO"] == "1" and pd.isna(limpio.loc[1, "ESTRATO"])
    assert limpio.loc[0, "ESTADOCIVIL"] == "S"


def test_codigo_fuera_del_diccionario_no_cumple(lote, contrato):
    alterado = lote.copy()
    alterado.loc[0, "ESTUDIOS"] = "BACHILLER"  # el esquema anterior: el extracto usa códigos (O, U, G...)
    assert any("ESTUDIOS" in error for error in k.validar(alterado, contrato))


def test_fechas_de_texto_sin_ambiguedad():
    # El sistema del fondo escribe ISO; una fecha digitada a mano llega como dd/mm/aaaa. Aplicarle "día primero"
    # a una ISO voltearía mes y día (2026-03-12 sería el 3 de diciembre).
    s = pd.Series(["2026-03-12 10:00:00.597", "03/12/2020", pd.Timestamp("2020-01-05"), None, "  "], dtype=object)
    assert k.fecha_canonica(s).tolist() == [
        pd.Timestamp("2026-03-12 10:00:00.597"),
        pd.Timestamp("2020-12-03"),
        pd.Timestamp("2020-01-05"),
        pd.NaT,
        pd.NaT,
    ]


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
