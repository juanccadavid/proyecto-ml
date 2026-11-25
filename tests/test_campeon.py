"""Modelo campeón: umbrales fuera de partición, tres zonas, elección por PR-AUC y archivos para el simulador."""

import json

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import campeon_figuras as cf
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def foto():
    return a.foto(GeneradorAsociados(n=900, seed=1).generar(), 3)


@pytest.fixture(scope="module")
def elegido(foto):
    return c.elegir(foto, repeticiones=2, info={"datos": "prueba", "horizonte_meses": 3})


def test_umbrales_f2_y_precision_minima():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    puntaje = np.r_[rng.beta(2, 8, 900), rng.beta(6, 3, 100)]  # los retiros puntúan más alto
    u = c.elegir_umbrales(y, puntaje, beta=2, precision_min=0.8)
    pred = puntaje >= u["umbral"]
    assert u["recall"] == pytest.approx(pred[y == 1].mean())
    assert u["precision"] == pytest.approx(y[pred].mean())
    assert u["umbral_alto"] >= u["umbral"]
    alto = puntaje >= u["umbral_alto"]
    assert y[alto].mean() >= 0.8 and u["recall_alto"] <= u["recall"]
    # F2 premia el recall: su umbral es más bajo (o igual) que el de F1
    assert u["umbral"] <= c.elegir_umbrales(y, puntaje, beta=1)["umbral"]


def test_sin_zona_segura_si_ningun_umbral_llega_a_la_precision():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    u = c.elegir_umbrales(y, rng.random(400), precision_min=0.99)  # puntaje al azar
    assert u["umbral_alto"] is None


@pytest.mark.parametrize(
    ("puntaje", "esperada"), [(0.95, "se retira"), (0.5, "zona gris"), (0.1, "se queda"), (0.3, "zona gris")]
)
def test_zonas(puntaje, esperada):
    assert c.zona(puntaje, umbral=0.3, umbral_alto=0.8) == esperada


def test_sin_umbral_alto_todo_lo_alertado_es_zona_gris():
    assert c.zona(0.99, 0.3, None) == "zona gris"


def test_elige_el_de_mejor_pr_auc_y_lo_deja_entrenado(elegido, foto):
    tabla = elegido.tabla()
    assert list(tabla.index)[0] == elegido.nombre == tabla[c.CRITERIO].idxmax()
    assert set(tabla.index) == set(c.lista_candidatos(a.separar(foto)[0]))  # los 4 + técnicas concluyentes
    assert elegido.candidatos[elegido.nombre].puntajes.shape == (2, len(foto))
    zonas = elegido.zona(a.preparar(foto))
    assert set(zonas) <= set(c.ZONAS) and "se queda" in zonas
    # KMeans: el grupo de riesgo se retira más que el estable (sin haber visto el objetivo)
    assert elegido.grupos["riesgo"]["tasa_retiro"] > elegido.grupos["estable"]["tasa_retiro"]
    assert elegido.info["horizonte_meses"] == 3 and elegido.info["filas"] == len(foto)


def test_guardar_y_cargar(elegido, foto, tmp_path):
    rutas = c.guardar(elegido, tmp_path, extra={"nota": "prueba"})
    assert {r.name for r in rutas.values()} == {c.ARCHIVO_MODELO, c.ARCHIVO_GRUPOS, c.ARCHIVO_INFO}
    modelo, agrupador, info = c.cargar(tmp_path)
    X = a.preparar(foto)
    assert np.allclose(a.puntaje(modelo, X), a.puntaje(elegido.modelo, X))
    assert (agrupador.grupos(X) == elegido.agrupador.predict(foto)).all()
    assert info["modelo"] == elegido.nombre and info["umbral"] == elegido.umbral and info["nota"] == "prueba"
    assert info["variables"] == a.NUMERICAS + a.CATEGORICAS
    assert set(info["grupos"]) == {"riesgo", "estable"}
    assert c.cargar(tmp_path / "vacia") is None

    info["variables"].append("VARIABLE_OBSOLETA")
    rutas["info"].write_text(json.dumps(info), encoding="utf-8")
    assert c.cargar(tmp_path) is None


def test_figura_de_umbrales(elegido):
    fig = cf.figura_umbrales(elegido)
    assert [t.name for t in fig.data] == ["Precisión", "Recall"]


def test_datos_eleccion_con_ruta(tmp_path):
    df = GeneradorAsociados(n=60, seed=3).generar()
    ruta = tmp_path / "asociados.parquet"
    df.to_parquet(ruta, index=False)
    datos, fuente = c.datos_eleccion(ruta)
    assert fuente == "asociados.parquet" and len(datos) == 60
    assert isinstance(datos, pd.DataFrame)


def test_datos_reales_sin_etiqueta_futura_no_entrenan(tmp_path):
    ruta = tmp_path / "extracto.parquet"
    GeneradorAsociados(n=60, seed=4).generar_extracto().to_parquet(ruta, index=False)
    with pytest.raises(c.DatosInvalidos, match="conteo histórico|acumulado durante la permanencia"):
        c.datos_eleccion(ruta, origen="real_anonimizado")


def test_datos_reales_sirven_para_lo_que_no_usa_la_etiqueta(tmp_path):
    # Segmentación y regresión del salario no usan la etiqueta futura: con el extracto real sí corren. Se escribe
    # como llega el real: Excel, nombres con tildes y una actualización a las 23:00 del día del corte.
    extracto = GeneradorAsociados(n=60, seed=4).generar_extracto()
    extracto = extracto.rename(columns={"EDAD": "EDAD_AÑOS", "FECHAINGRESOLABORA": "FECHAINGRESALABORA"})
    extracto.loc[0, "FECHAULTACTUALIZACION"] = a.FECHA_CORTE + pd.Timedelta(hours=23)
    ruta = tmp_path / "extracto.xlsx"
    extracto.to_excel(ruta, index=False)
    datos, fuente = c.datos_eleccion(ruta, origen="real_anonimizado", supervisado=False)
    assert fuente == "extracto.xlsx" and len(datos) == 60 and a.TARGET not in datos
    assert {"EDAD", "FECHAINGRESOLABORA"} <= set(datos.columns)


def test_datos_reales_con_fechas_posteriores_al_corte_no_pasan(tmp_path):
    extracto = GeneradorAsociados(n=60, seed=4).generar_extracto()
    extracto.loc[0, "FECHAINGRESO"] = a.FECHA_CORTE + pd.Timedelta(days=1)  # corte mal puesto en el .env
    ruta = tmp_path / "extracto.parquet"
    extracto.to_parquet(ruta, index=False)
    with pytest.raises(c.DatosInvalidos, match="FECHAINGRESO tiene fechas después de FECHA_CORTE"):
        c.datos_eleccion(ruta, origen="real_anonimizado", supervisado=False)


def test_valores_imposibles_quedan_vacios():
    df = GeneradorAsociados(n=4, seed=4).generar_extracto()
    df.loc[0, "SALARIO"] = 987_654  # relleno del extracto
    df.loc[1, "SALARIO"] = 32_421_080_000  # error de digitación
    df.loc[2, ["EDAD", "ANTIGUEDAD_LA_MESES"]] = [30, 1514]  # 126 años de antigüedad laboral
    X = a.preparar(df)
    assert X.loc[[0, 1], "SALARIO"].isna().all()
    assert pd.isna(X.loc[2, "ANTIGUEDAD_LA_MESES"])
    assert pd.notna(X.loc[3, "SALARIO"])


def test_estrato_en_blanco_es_categoria_y_cargo_no_entra():
    df = GeneradorAsociados(n=4, seed=4).generar_extracto()
    df.loc[0, "ESTRATO"] = None
    X = a.preparar(df)
    assert X.loc[0, "ESTRATO"] == a.ESTRATO_SIN_DATO  # categoría propia: no se imputa con la moda
    assert "CARGO" not in X.columns and "CODPROFESION" in X.columns  # CARGO repite CODPROFESION
