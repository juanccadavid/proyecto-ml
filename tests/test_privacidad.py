"""Fronteras de privacidad del esquema anonimizado definitivo."""

from pathlib import Path

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import predict as p
from proyecto_ml.generador import GeneradorAsociados


def test_identificadores_y_contacto_no_entran_al_modelo():
    prohibidas = {
        "NROIDENTIFICACION",
        "PRIMERAPELLIDO",
        "SEGUNDOAPELLIDO",
        "NOMBRES",
        "SEGUNDONOMBRE",
        "NOMBREINTEGRADO",
        "DIRECCION",
        "EMAIL",
        "CELULAR",
        "ESTADO",
        "FECHAULTACTUALIZACION",
    }
    assert prohibidas.isdisjoint(a.NUMERICAS + a.CATEGORICAS)
    assert "#RETIROS" in a.NUMERICAS


def test_modo_real_aplica_lista_blanca_del_contrato():
    df = GeneradorAsociados(n=40, seed=1).generar_extracto()
    p.revisar_real(df)
    with pytest.raises(p.DatosPersonales, match="no conoce"):
        p.revisar_real(df.assign(COLUMNA_NO_AUTORIZADA="x"))


def test_modo_real_exige_todos_los_predictores():
    df = GeneradorAsociados(n=40, seed=1).generar_extracto().drop(columns="EDAD")
    with pytest.raises(p.DatosPersonales, match="EDAD"):
        p.revisar_real(df)


def test_salida_de_prediccion_solo_define_agregados_operativos():
    assert p.COLUMNAS == ["puntaje", "zona", "alerta", "grupo_kmeans"]
    assert "NROIDENTIFICACION" not in p.COLUMNAS


def test_archivo_anonimizado_no_se_agrega_a_gitignore():
    texto = Path(".gitignore").read_text(encoding="utf-8")
    assert "*.xlsx" in texto or "data/" in texto


def test_preparar_no_muta_el_lote():
    df = GeneradorAsociados(n=40, seed=2).generar_extracto()
    original = df.copy(deep=True)
    a.preparar(df)
    pd.testing.assert_frame_equal(df, original)
