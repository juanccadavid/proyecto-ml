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
    assert set(info["grupos"]) == {"riesgo", "estable"}
    assert info["variables"] == a.NUMERICAS + a.CATEGORICAS
    assert c.cargar(tmp_path / "vacia") is None


def test_no_carga_un_campeon_entrenado_con_otras_variables(elegido, tmp_path):
    # Pasó de verdad: al sacar CARGO del modelo, el campeón guardado fallaba al predecir con una traza cruda.
    c.guardar(elegido, tmp_path)
    ruta = tmp_path / c.ARCHIVO_INFO
    info = json.loads(ruta.read_text(encoding="utf-8"))
    info["variables"] = [*info["variables"], "CARGO"]
    ruta.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(c.CampeonIncompatible, match="CARGO"):
        c.cargar(tmp_path)
    del info["variables"]  # un campeón guardado antes de registrar las variables no se puede revisar: se acepta
    ruta.write_text(json.dumps(info), encoding="utf-8")
    assert c.cargar(tmp_path) is not None


def test_cargar_sincroniza_umbrales_y_firma_depende_del_contenido_real(elegido, foto, tmp_path):
    import os

    import joblib

    rutas = c.guardar(elegido, tmp_path)
    firma_inicial = c.firma(tmp_path)
    firma_mod_inicial = c.firma_modelo(tmp_path)
    assert firma_inicial and firma_mod_inicial

    # 1) Cambiar umbral en campeon.json conservando ruta actualiza ModeloZonas.predict y cambia firma, no firma_modelo.
    info = json.loads(rutas["info"].read_text(encoding="utf-8"))
    info["umbral"], info["umbral_alto"] = 0.9999, 0.99999
    rutas["info"].write_text(json.dumps(info), encoding="utf-8")
    assert c.firma(tmp_path) != firma_inicial
    assert c.firma_modelo(tmp_path) == firma_mod_inicial
    modelo_nuevo, _, _ = c.cargar(tmp_path)
    assert modelo_nuevo.umbral == pytest.approx(0.9999)
    assert (modelo_nuevo.predict(a.preparar(foto.head(20))) == "se queda").all()

    # 2) Sobrescribir abandono_kmeans.joblib conservando mtime cambia firma por contenido de bytes.
    firma_antes_kmeans = c.firma(tmp_path)
    stat_viejo = rutas["grupos"].stat()
    otro_kmeans = a.AgrupadorAbandono(random_state=99).fit(GeneradorAsociados(n=200, seed=99).generar())
    joblib.dump(otro_kmeans, rutas["grupos"])
    os.utime(rutas["grupos"], ns=(stat_viejo.st_atime_ns, stat_viejo.st_mtime_ns))
    assert c.firma(tmp_path) != firma_antes_kmeans
    assert c.firma_modelo(tmp_path) == firma_mod_inicial


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


def test_datos_reales_con_fechas_posteriores_al_corte_no_pasan(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "fecha_corte", lambda: "2026-03-31")
    extracto = GeneradorAsociados(n=60, seed=4).generar_extracto()
    extracto.loc[0, "FECHAINGRESO"] = pd.Timestamp("2026-03-31") + pd.Timedelta(days=1)
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


def test_elegir_rechaza_asociados_repetidos(foto):
    # La CV y la interna de TargetEncoder parten por filas: el mismo asociado en dos filas filtraría su etiqueta.
    repetido = pd.concat([foto, foto.head(5)], ignore_index=True)
    with pytest.raises(ValueError, match="repetidos"):
        c.elegir(repetido, honestos=False)


def test_predecir_real_avisa_si_el_campeon_es_sintetico(elegido, tmp_path, monkeypatch):
    from proyecto_ml import predict as p

    c.guardar(elegido, tmp_path, extra={"origen_datos": "sintetico"})
    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    assert p.aviso_campeon_sintetico(tmp_path) is None  # con sintéticos no hace falta
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    assert "SINTÉTICOS" in p.aviso_campeon_sintetico(tmp_path) and "NO están validados" in p.aviso_campeon_sintetico(
        tmp_path
    )


def test_scripts_exploratorios_no_usan_datos_reales_ni_lotes_sin_etiqueta(tmp_path, monkeypatch):
    # make abandono, afinar, optuna y pycaret validan contra asociados del generador: con datos reales mezclarían
    # orígenes, así que se niegan; y sin la etiqueta sintética no hay qué aprender.
    ruta = tmp_path / "desarrollo.parquet"
    GeneradorAsociados(n=60, seed=3).generar().to_parquet(ruta, index=False)
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    with pytest.raises(c.DatosInvalidos, match="exploratorio"):
        c.datos_exploracion(ruta, "make abandono")
    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    assert len(c.datos_exploracion(ruta, "make abandono")) == 60
    sin_etiqueta = tmp_path / "extracto.parquet"
    GeneradorAsociados(n=60, seed=3).generar_extracto().to_parquet(sin_etiqueta, index=False)
    with pytest.raises(c.DatosInvalidos, match="etiqueta sintética"):
        c.datos_exploracion(sin_etiqueta, "make abandono")


def test_scripts_exploratorios_usan_por_defecto_los_datos_de_desarrollo(tmp_path, monkeypatch):
    # Antes el valor por defecto era DATOS_ASOCIADOS (sin etiqueta desde el contrato v2) y make abandono, afinar,
    # optuna y pycaret fallaban sin argumentos.
    from proyecto_ml import config
    from proyecto_ml.datos_campeon import nombre_fuente

    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)  # sin archivo de desarrollo: el generador en memoria
    en_memoria = c.datos_exploracion(None, "make abandono")
    assert a.TARGET in en_memoria and len(en_memoria) == 5000
    assert nombre_fuente(None) == "datos_desarrollo() en memoria"
    GeneradorAsociados(n=80, seed=3).generar().to_parquet(tmp_path / "asociados_desarrollo.parquet", index=False)
    assert len(c.datos_exploracion(None, "make abandono")) == 80  # con archivo, el de make datos-desarrollo
    assert nombre_fuente(None) == "asociados_desarrollo.parquet"
