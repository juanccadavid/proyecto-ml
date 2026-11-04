"""Privacidad y contrato: anonimización segura, contrato del dataset anonimizado para ML, predicciones sin datos
personales, una sola fecha de corte y el Brier solo con probabilidades. Todo con datos sintéticos."""

import importlib
import re

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import anonymize as an
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml import contrato as k
from proyecto_ml import predict as p
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import GeneradorAsociados

SAL = "clave-de-prueba-0123456789"


@pytest.fixture(scope="module")
def extracto():
    return a.cargar("data/ejemplo/asociados_ejemplo.parquet")  # 50 asociados ficticios, con su "PII" inventada


@pytest.fixture(scope="module")
def anonimo(extracto):
    return an.anonimizar_extracto(extracto, salt=SAL)


# ---------------------------------------------------------------- anonimización


def test_anonimizar_quita_la_pii_y_cumple_el_contrato_ml(extracto, anonimo):
    personales = set(k.columnas_pii())
    assert not personales & set(anonimo.columns)
    assert {"CEDULA_SEUDO", "IDASOCIADOS_SEUDO"} <= set(anonimo.columns)
    assert k.validar(anonimo, k.para_ml()) == []
    assert len(anonimo) == len(extracto)


def test_seudonimo_consistente_de_64_hex_y_no_numerico(extracto, anonimo):
    seudo = anonimo["CEDULA_SEUDO"]
    assert seudo.str.fullmatch(r"[0-9a-f]{64}").all() and not seudo.str.fullmatch(r"\d+").any()
    assert pd.api.types.is_string_dtype(seudo)  # texto, nunca un número
    otra_vez = an.anonimizar_extracto(extracto, salt=SAL)["CEDULA_SEUDO"]
    assert (seudo == otra_vez).all()  # mismo identificador, mismo seudónimo
    assert an.hash_value(1037000000, SAL) == an.hash_value(1037000000.0, SAL) == an.hash_value(" 1037000000 ", SAL)
    assert an.hash_value(1037000000, SAL) != an.hash_value(1037000000, SAL + "x")
    assert seudo.nunique() == extracto["CEDULA"].nunique()  # sin colisiones


def test_sin_clave_o_con_identificador_vacio_falla(extracto):
    with pytest.raises(an.AnonimizacionInsegura, match="ANON_SALT"):
        an.anonimizar_extracto(extracto, salt="corta")
    con_vacio = extracto.copy()
    con_vacio.loc[con_vacio.index[0], "CEDULA"] = None
    with pytest.raises(an.AnonimizacionInsegura, match="CEDULA: 1 identificador"):
        an.anonimizar_extracto(con_vacio, salt=SAL)


def test_columna_desconocida_exige_una_decision(extracto):
    con_extra = extracto.assign(CORREO_PERSONAL="ana@correo.com")
    with pytest.raises(an.AnonimizacionInsegura, match="CORREO_PERSONAL") as error:
        an.anonimizar_extracto(con_extra, salt=SAL)
    assert "ana@correo.com" not in str(error.value)
    assert "CORREO_PERSONAL" not in an.anonimizar_extracto(con_extra, salt=SAL, eliminar=["CORREO_PERSONAL"])
    with pytest.raises(an.AnonimizacionInsegura, match="conservar"):
        an.anonimizar_extracto(extracto, salt=SAL, conservar=["NOMBRES"])  # una columna personal no se conserva


def test_un_dato_personal_copiado_en_otra_columna_detiene_todo(extracto):
    copiado = extracto.copy()
    copiado["OPERADORACTUALIZO"] = copiado["EMAIL"]  # el correo, en una columna que se conserva
    with pytest.raises(an.AnonimizacionInsegura, match="OPERADORACTUALIZO: 50 fila") as error:
        an.anonimizar_extracto(copiado, salt=SAL)
    assert not any(correo in str(error.value) for correo in extracto["EMAIL"])  # cantidades, no valores


def test_el_script_no_escribe_nada_si_falla_ni_guarda_la_clave(tmp_path, monkeypatch, extracto):
    import importlib.util

    ruta = config.ROOT_DIR / "scripts" / "anonimizar.py"
    spec = importlib.util.spec_from_file_location("anonimizar", ruta)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    entrada, salida = tmp_path / "extracto.parquet", tmp_path / "anon.parquet"
    extracto.assign(OBSERVACIONES="texto libre").to_parquet(entrada)

    monkeypatch.setenv("ANON_SALT", SAL)
    with pytest.raises(SystemExit, match="No se escribió nada"):
        script.main(["--input", str(entrada), "--output", str(salida)])  # OBSERVACIONES sin decisión
    assert not salida.exists() and not list(tmp_path.glob(".anon.parquet.*"))  # ni el archivo ni temporales

    script.main(["--input", str(entrada), "--output", str(salida), "--eliminar", "OBSERVACIONES"])
    escrito = pd.read_parquet(salida)
    assert not set(k.columnas_pii()) & set(escrito.columns) and "OBSERVACIONES" not in escrito
    assert SAL.encode() not in salida.read_bytes()  # la clave no queda en el archivo
    monkeypatch.setenv("ANON_SALT", "")
    with pytest.raises(SystemExit, match="ANON_SALT"):
        script.main(
            ["--input", str(entrada), "--output", str(tmp_path / "otro.parquet"), "--eliminar", "OBSERVACIONES"]
        )
    assert not (tmp_path / "otro.parquet").exists()


# ---------------------------------------------------------------- contrato del dataset anonimizado para ML


def test_contrato_ml_sin_pii_pero_con_todas_las_variables_del_modelo():
    ml = k.para_ml()
    assert not set(k.columnas_pii()) & set(ml.nombres)
    assert set(a.NUMERICAS + a.CATEGORICAS) - {"EDAD", "ANTIGUEDAD"} <= set(ml.nombres)
    assert {"FECHANACIMIENTO", "FECHAINGRESOLABORA", "FECHARETIRO", a.TARGET} <= set(ml.nombres)
    exigidas = {col.nombre for col in ml.columnas if col.requerido}
    origen = {col.nombre for col in k.cargar().columnas if col.requerido and not col.pii}
    assert origen <= exigidas  # no relaja nada fuera de la PII
    assert "nombreintegrado_coherente" not in [r["id"] for r in ml.reglas]  # usa nombres: no aplica


def test_contrato_ml_detecta_faltantes_tipos_y_pii_que_quedo(anonimo):
    sin_variable = anonimo.drop(columns="#CANTIDAD_SERVICIOS")
    assert any("Faltan columnas" in e and "#CANTIDAD_SERVICIOS" in e for e in k.validar(sin_variable, k.para_ml()))
    mal_tipo = anonimo.astype({"SALARIO": object})
    mal_tipo.loc[mal_tipo.index[0], "SALARIO"] = "mucho"
    assert any("SALARIO" in e and "tipo" in e for e in k.validar(mal_tipo, k.para_ml()))
    con_pii = anonimo.assign(NOMBRES="ANA")
    assert any("fuera del contrato" in e and "NOMBRES" in e for e in k.validar(con_pii, k.para_ml()))
    seudo_malo = anonimo.assign(CEDULA_SEUDO="1037000000")
    assert k.validar(seudo_malo, k.para_ml())  # una cédula no pasa por seudónimo


def test_las_columnas_personales_no_son_variables_del_modelo(anonimo):
    assert not set(k.columnas_pii()) & set(a.NUMERICAS + a.CATEGORICAS)
    datos = anonimo
    modelo = a.modelos_supervisados()["logreg"].fit(*a.separar(datos))  # entrena sin ninguna columna personal
    assert len(modelo.predict_proba(a.separar(datos)[0])) == len(datos)


def test_modo_real_acepta_el_anonimizado_y_rechaza_el_extracto_con_pii(tmp_path, extracto, anonimo):
    real, crudo = tmp_path / "anon.parquet", tmp_path / "crudo.parquet"
    anonimo.to_parquet(real)
    extracto.to_parquet(crudo)
    df, fuente = c.datos_eleccion(real, origen="real_anonimizado")
    assert fuente == "anon.parquet" and len(df) == len(anonimo)
    with pytest.raises(c.DatosInvalidos, match="contrato-ml") as error:
        c.datos_eleccion(crudo, origen="real_anonimizado")
    assert not any(str(v) in str(error.value) for v in extracto["CEDULA"])
    assert c.datos_eleccion(crudo, origen="sintetico")[1] == "crudo.parquet"  # el sintético sigue con el de origen


# ---------------------------------------------------------------- predicción sin datos personales


@pytest.fixture(scope="module")
def campeon_guardado(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("campeon")
    foto = a.foto(GeneradorAsociados(n=600, seed=11, tasa_retiro=0.25).generar(), 3)
    c.guardar(c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1), carpeta)
    return carpeta


def test_prediccion_sin_cedulas_ni_datos_personales(campeon_guardado, extracto, anonimo):
    sintetico = p.predecir(extracto, campeon_guardado, origen="sintetico")
    assert list(sintetico.columns) == ["fila", *p.COLUMNAS]  # sin CEDULA ni IDASOCIADOS crudos
    real = p.predecir(anonimo, campeon_guardado, origen="real_anonimizado")
    assert list(real.columns) == ["fila", "CEDULA_SEUDO", "IDASOCIADOS_SEUDO", *p.COLUMNAS]
    assert not set(k.columnas_pii()) & set(real.columns)
    assert (real["CEDULA_SEUDO"].to_numpy() == anonimo["CEDULA_SEUDO"].to_numpy()).all()  # se puede cruzar


def test_prediccion_rechaza_cedulas_crudas(campeon_guardado, extracto, anonimo):
    disfrazada = anonimo.assign(CEDULA_SEUDO=extracto["CEDULA"].astype(str).to_numpy())
    with pytest.raises(p.DatosPersonales, match="CEDULA_SEUDO: 50 valor") as error:
        p.predecir(disfrazada, campeon_guardado)
    assert not any(str(v) in str(error.value) for v in extracto["CEDULA"])
    with pytest.raises(p.DatosPersonales, match="anonimízalo"):
        p.predecir(extracto, campeon_guardado, origen="real_anonimizado")  # en modo real, el extracto con PII


# ---------------------------------------------------------------- una sola fecha de corte


def test_contrato_y_pipeline_usan_la_misma_fecha_de_corte(extracto):
    assert "fecha_corte" not in next(d for d in k.cargar().distribucion if d["id"] == "edad_mediana_plausible")
    assert str(a.FECHA_CORTE.date()) == config.fecha_corte()
    medida = k.DISTRIBUCION["edad_mediana_plausible"][0](extracto, {})
    assert medida == pytest.approx(a.preparar(extracto)["EDAD"].median())  # EDAD a FECHA_CORTE en los dos


def test_cambiar_fecha_corte_mueve_todo_junto(monkeypatch, extracto):
    monkeypatch.setenv("FECHA_CORTE", "2027-10-31")  # un año después
    try:
        importlib.reload(a)  # FECHA_CORTE se lee al importar
        assert str(a.FECHA_CORTE.date()) == "2027-10-31" == config.fecha_corte()
        medida = k.DISTRIBUCION["edad_mediana_plausible"][0](extracto, {})
        assert medida == pytest.approx(a.preparar(extracto)["EDAD"].median())
        assert a.fecha_observacion(3) == pd.Timestamp("2027-07-31")
    finally:
        monkeypatch.delenv("FECHA_CORTE")
        importlib.reload(a)
    assert str(a.FECHA_CORTE.date()) == config.FECHA_CORTE_DEFECTO


def test_el_generador_no_pasa_de_la_fecha_de_corte_por_defecto():
    df = GeneradorAsociados(n=500, seed=2).generar()
    corte = pd.Timestamp(config.FECHA_CORTE_DEFECTO)
    for col in ("FECHARETIRO", "FECHAULTACTUALIZACION", "FECHAINGRESOLABORA", "FECHANACIMIENTO"):
        assert pd.to_datetime(df[col]).max() <= corte, col  # FECHATERMINACONTRATO sí puede ser futura


def test_edad_respecto_de_la_fecha_configurada():
    df = pd.DataFrame({"FECHANACIMIENTO": ["1996-10-31"], "FECHAINGRESOLABORA": ["2020-10-31"]})
    for c_ in a.NUMERICAS:
        if c_ not in ("EDAD", "ANTIGUEDAD"):
            df[c_] = 0
    for c_ in a.CATEGORICAS:
        df[c_] = "x"
    assert a.preparar(df)["EDAD"].iloc[0] == pytest.approx(30, abs=0.01)  # 1996-10-31 -> 2026-10-31


# ---------------------------------------------------------------- Brier solo con probabilidades


def test_brier_solo_con_probabilidades():
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9])
    con = compute_metrics(y, proba, metodo="predict_proba")
    assert con["brier"] == pytest.approx(np.mean((proba - y) ** 2))
    distancia = np.array([0.05, 0.2, 0.7, 0.55, 0.1, 0.95])  # en [0, 1], pero es una decision_function
    assert np.isnan(compute_metrics(y, distancia, metodo="decision_function")["brier"])
    assert np.isnan(compute_metrics(y, proba)["brier"])  # sin decir el método, no se asume probabilidad
    with pytest.raises(ValueError, match="fuera de"):
        compute_metrics(y, proba * 3 - 1, metodo="predict_proba")
    sin_metodo = compute_metrics(y, proba)
    assert {key: v for key, v in con.items() if key != "brier"} == pytest.approx(
        {key: v for key, v in sin_metodo.items() if key != "brier"}
    )  # las demás métricas no cambian


def test_patron_del_seudonimo_coincide_con_el_hmac():
    assert re.fullmatch(k.PATRON_SEUDONIMO, an.hash_value("123", SAL))
