"""Preparación y modelos con las variables definitivas."""

import numpy as np
import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml.generador import GeneradorAsociados


def datos(n=500, seed=5):
    return GeneradorAsociados(n=n, seed=seed, tasa_retiro=0.15, fuerza_senal=2.0).generar()


def test_preparar_solo_usa_predictores_permitidos():
    df = datos(80)
    X = a.preparar(df)
    assert list(X.columns) == a.NUMERICAS + a.CATEGORICAS
    assert not set(X) & set(a.IDENTIFICADORES + a.FUGA)
    assert {"EDAD", "ANTIGUEDAD_F_MESES", "ANTIGUEDAD_LA_MESES"} <= set(X)


def test_numericas_y_categoricas_quedan_tipadas():
    X = a.preparar(datos(80))
    assert all(pd.api.types.is_float_dtype(X[c]) for c in a.NUMERICAS)
    assert all(pd.api.types.is_string_dtype(X[c]) for c in a.CATEGORICAS)


def test_foto_no_inventa_retiros_ni_filtra_filas():
    df = datos(80)
    vista = a.foto(df, horizonte_meses=3, corte=pd.Timestamp("2026-10-31"))
    assert len(vista) == len(df)
    assert vista[a.TARGET].equals(df[a.TARGET])
    assert (vista[a.FECHA_OBSERVACION] == pd.Timestamp("2026-07-31")).all()


def test_modelos_supervisados_entrenan_y_predicen():
    df = datos(500)
    X, y = a.separar(df)
    for modelo in a.modelos_supervisados().values():
        entrenado = modelo.fit(X, y)
        assert len(entrenado.predict(X.iloc[:12])) == 12
        assert np.isfinite(a.puntaje(entrenado, X.iloc[:12])).all()


def test_identificadores_y_fuga_no_cambian_predicciones():
    df = datos(500)
    X, y = a.separar(df)
    modelo = a.modelos_supervisados()["logreg"].fit(X, y)
    base = a.puntaje(modelo, X.iloc[:20])
    alterado = X.iloc[:20].copy()
    for columna in a.IDENTIFICADORES + ["ESTADO", "FECHAULTACTUALIZACION"]:
        if columna in alterado:
            alterado[columna] = "VALOR_IGNORADO"
    np.testing.assert_allclose(base, a.puntaje(modelo, alterado))


def test_agrupador_usa_solo_comportamiento():
    df = datos(500)
    agrupador = a.AgrupadorAbandono().fit(df)
    grupos = agrupador.predict(df)
    assert set(grupos) <= {0, 1}
    assert len(grupos) == len(df)
