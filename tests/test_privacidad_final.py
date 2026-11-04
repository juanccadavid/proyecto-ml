"""Correcciones finales de privacidad: validar antes de escribir, PII incrustada, identificadores en las
predicciones, Brier solo con probabilidades y la tasa de retiro como advertencia. Todo con datos ficticios."""

import importlib.util
import warnings

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


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("anonimizar", config.ROOT_DIR / "scripts" / "anonimizar.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _sin_valores(mensaje: str, valores) -> bool:
    return not any(str(v) in mensaje for v in valores if len(str(v)) >= 4)


# ---------------------------------------------------------------- 1. validar antes de escribir


def test_salida_que_no_cumple_el_contrato_ml_no_se_escribe(tmp_path, monkeypatch, extracto, script):
    """CODSUCURSAL no es personal (pasa la verificación de PII) pero no está en el contrato para ML: antes se
    escribía el archivo y solo se avisaba; ahora se detiene sin escribir."""
    entrada, salida = tmp_path / "extracto.parquet", tmp_path / "anon.parquet"
    extracto.assign(CODSUCURSAL="S01").to_parquet(entrada)
    monkeypatch.setenv("ANON_SALT", SAL)
    escrituras = []
    monkeypatch.setattr(script, "escribir_atomico", lambda df, destino: escrituras.append(destino))
    with pytest.raises(SystemExit, match="No se escribió nada. La salida no cumple el contrato") as error:
        script.main(["--input", str(entrada), "--output", str(salida), "--conservar", "CODSUCURSAL"])
    assert "CODSUCURSAL" in str(error.value)  # dice la columna
    assert escrituras == [] and not salida.exists() and not list(tmp_path.glob(".anon.parquet.*"))


def test_un_dataset_invalido_no_sobrescribe_la_salida_existente(tmp_path, monkeypatch, extracto, script):
    entrada, salida = tmp_path / "extracto.parquet", tmp_path / "anon.parquet"
    salida.write_bytes(b"version anterior")
    monkeypatch.setenv("ANON_SALT", SAL)
    malo = extracto.copy()
    malo.loc[malo.index[0], "ESTADO"] = "VALOR_RARO_XYZ"  # rompe el contrato (no es PII)
    malo.to_parquet(entrada)
    with pytest.raises(SystemExit, match="No se escribió nada") as error:
        script.main(["--input", str(entrada), "--output", str(salida)])
    assert "ESTADO" in str(error.value) and "VALOR_RARO_XYZ" not in str(error.value)  # columna, no valor
    assert salida.read_bytes() == b"version anterior"  # intacta
    extracto.assign(OBSERVACIONES="libre").to_parquet(entrada)  # y una falla de PII tampoco la toca
    with pytest.raises(SystemExit, match="No se escribió nada"):
        script.main(["--input", str(entrada), "--output", str(salida)])
    assert salida.read_bytes() == b"version anterior" and not list(tmp_path.glob(".anon.parquet.*"))


def test_salida_valida_se_escribe_y_cumple_el_contrato_ml(tmp_path, monkeypatch, extracto, script, capsys):
    entrada, salida = tmp_path / "extracto.parquet", tmp_path / "anon.parquet"
    extracto.to_parquet(entrada)
    monkeypatch.setenv("ANON_SALT", SAL)
    script.main(["--input", str(entrada), "--output", str(salida)])
    assert k.validar(pd.read_parquet(salida), k.para_ml()) == []
    consola = capsys.readouterr().out
    assert "cumple el contrato" in consola and _sin_valores(consola, extracto["CEDULA"])


# ---------------------------------------------------------------- 2. PII residual


def _con_texto(extracto: pd.DataFrame, textos) -> pd.DataFrame:
    """OPERADORACTUALIZO (texto del contrato que se conserva) con ``textos`` en las primeras filas."""
    df = extracto.copy()
    df["OPERADORACTUALIZO"] = df["OPERADORACTUALIZO"].astype(object)
    for i, texto in enumerate(textos):
        df.loc[df.index[i], "OPERADORACTUALIZO"] = texto
    return df


def _problemas(df: pd.DataFrame, **kw) -> str:
    with pytest.raises(an.AnonimizacionInsegura) as error:
        an.anonimizar_extracto(df, salt=SAL, **kw)
    return str(error.value)


def test_pii_exacta_copiada_se_detecta(extracto):
    mensaje = _problemas(_con_texto(extracto, [str(extracto["CEDULA"].iloc[5])]))
    assert "OPERADORACTUALIZO: 1 fila(s) con valores de columnas personales" in mensaje
    assert _sin_valores(mensaje, extracto["CEDULA"])


@pytest.mark.parametrize(
    "plantilla",
    ["actualizado por cc {}", "CC: {}.", "ver {} (titular)"],
    ids=["frase", "puntuacion", "parentesis"],
)
def test_cedula_incrustada_en_texto_se_detecta(extracto, plantilla):
    cedula = int(extracto["CEDULA"].iloc[3])
    con_puntos = f"{cedula:,}".replace(",", ".")  # 1.037.000.000: como se escribe a mano
    mensaje = _problemas(_con_texto(extracto, [plantilla.format(cedula), plantilla.format(con_puntos)]))
    assert "OPERADORACTUALIZO: 2 fila(s) con un número de una columna personal" in mensaje
    assert _sin_valores(mensaje, [cedula, con_puntos])


def test_telefono_nombre_y_correo_incrustados_se_detectan(extracto):
    fila = extracto.iloc[7]
    telefono = str(fila["TELEFONO1"])
    textos = [
        f"llamar al {telefono[:3]}-{telefono[3:6]}-{telefono[6:]}",  # con guiones
        f"Habló con {fila['NOMBREINTEGRADO'].lower()} ayer",  # nombre completo, en minúscula
        f"escribir a <{fila['EMAIL']}> por favor",
    ]
    mensaje = _problemas(_con_texto(extracto, textos))
    assert "OPERADORACTUALIZO: 1 fila(s) con un número de una columna personal" in mensaje
    assert "OPERADORACTUALIZO: 2 fila(s) con un nombre, dirección o correo del extracto" in mensaje
    assert _sin_valores(mensaje, [telefono, fila["NOMBREINTEGRADO"], fila["EMAIL"], fila["EMAIL"].lower()])


def test_identificadores_legitimos_y_falsos_positivos_razonables_no_se_marcan(extracto):
    personales = {str(v) for col in k.columnas_pii() for v in extracto[col].dropna()}
    codigos = [f"USR{n:07d}" for n in range(9_100_001, 9_100_011)]  # códigos internos de 7 dígitos
    assert not {cod[3:] for cod in codigos} & personales  # de verdad no son de nadie del extracto
    nombre_suelto = str(extracto["NOMBRES"].iloc[0])  # un nombre suelto dentro de un texto: no se marca
    textos = [*codigos, f"BARRIO SANTA {nombre_suelto}", "radicado 2026-10-31 folio 123", "ext 12"]
    salida = an.anonimizar_extracto(_con_texto(extracto, textos), salt=SAL)
    assert salida["OPERADORACTUALIZO"].iloc[0] == "USR9100001"  # se conserva tal cual


def test_correo_de_alguien_fuera_del_extracto_y_columna_sensible_por_nombre(extracto):
    mensaje = _problemas(_con_texto(extracto, ["contacto: otra.persona@ejemplo.org"]))
    assert "OPERADORACTUALIZO: 1 fila(s) con forma de correo electrónico" in mensaje
    assert "otra.persona" not in mensaje
    con_celular = extracto.assign(CELULAR_ALTERNO="sin dato")
    mensaje = _problemas(con_celular, conservar=["CELULAR_ALTERNO"])
    assert "parecen datos personales" in mensaje and "CELULAR_ALTERNO" in mensaje
    assert an.columnas_sensibles(["CODSUCURSAL", "NOMBRECARGO", "CORREO_2", "Telefono Casa"]) == [
        "CORREO_2",
        "Telefono Casa",
    ]


def test_la_deteccion_es_configurable(extracto):
    cedula = int(extracto["CEDULA"].iloc[3])
    df = _con_texto(extracto, [f"cc {cedula}"])
    _problemas(df)  # por defecto se detecta
    largo = len(str(cedula)) + 1  # con un mínimo más largo que la cédula, esa búsqueda no la ve (y se documenta)
    an.anonimizar_extracto(df, salt=SAL, deteccion=an.Deteccion(largo_min_digitos=largo))
    con_celular = extracto.assign(CELULAR_ALTERNO="sin dato")
    sin_nombres = an.Deteccion(nombres_sensibles=None)
    assert "CELULAR_ALTERNO" in an.anonimizar_extracto(
        con_celular, salt=SAL, conservar=["CELULAR_ALTERNO"], deteccion=sin_nombres
    )


# ---------------------------------------------------------------- 3. identificadores en las predicciones


@pytest.fixture(scope="module")
def campeon_guardado(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("campeon")
    foto = a.foto(GeneradorAsociados(n=600, seed=11, tasa_retiro=0.25).generar(), 3)
    c.guardar(c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1), carpeta)
    return carpeta


def test_modo_real_solo_admite_columnas_del_contrato_ml(campeon_guardado, anonimo):
    real = "real_anonimizado"
    with pytest.raises(p.DatosPersonales, match="no conoce") as error:
        p.predecir(anonimo.assign(CELULAR="3001112233"), campeon_guardado, origen=real)
    assert "CELULAR" in str(error.value) and "3001112233" not in str(error.value)
    falso = anonimo.assign(OTRO_SEUDO=anonimo["CEDULA_SEUDO"])  # forma válida, pero el contrato no lo reconoce
    with pytest.raises(p.DatosPersonales, match="OTRO_SEUDO"):
        p.predecir(falso, campeon_guardado, origen=real)
    with pytest.raises(p.DatosPersonales, match="ningún seudónimo"):
        p.predecir(anonimo.drop(columns=["CEDULA_SEUDO", "IDASOCIADOS_SEUDO"]), campeon_guardado, origen=real)


def test_modo_sintetico_conserva_el_comportamiento(campeon_guardado, extracto):
    salida = p.predecir(extracto.assign(EXTRA=1), campeon_guardado, origen="sintetico")
    assert list(salida.columns) == ["fila", *p.COLUMNAS]  # sin cédulas, con columnas extra permitidas


def test_csv_de_predicciones_en_modo_real_sin_pii(tmp_path, monkeypatch, campeon_guardado, extracto, anonimo):
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    entrada, salida = tmp_path / "anon.parquet", tmp_path / "pred.csv"
    anonimo.to_parquet(entrada)
    p.main(["--datos", str(entrada), "--salida", str(salida), "--modelos", str(campeon_guardado)])
    pred = pd.read_csv(salida, dtype={"CEDULA_SEUDO": str, "IDASOCIADOS_SEUDO": str})
    assert list(pred.columns) == ["fila", "CEDULA_SEUDO", "IDASOCIADOS_SEUDO", *p.COLUMNAS]
    activos = anonimo[anonimo["FECHARETIRO"].isna()]
    assert (pred["CEDULA_SEUDO"] == activos["CEDULA_SEUDO"].to_numpy()).all()  # se cruza por el seudónimo
    texto = salida.read_text(encoding="utf-8")
    personales = [extracto[col].dropna().astype(str) for col in ("CEDULA", "EMAIL", "NOMBREINTEGRADO", "TELEFONO1")]
    assert _sin_valores(texto, pd.concat(personales))


# ---------------------------------------------------------------- 4. Brier solo con probabilidades


def test_brier_con_probabilidades_validas():
    y = np.array([0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
    assert compute_metrics(y, proba, metodo="predict_proba")["brier"] == pytest.approx(np.mean((proba - y) ** 2))


def test_un_decision_score_entre_0_y_1_no_es_probabilidad():
    y = np.array([0, 1, 1, 0, 1])
    distancia = np.array([0.1, 0.8, 0.6, 0.3, 0.9])  # mismos números, pero de decision_function
    assert np.isnan(compute_metrics(y, distancia, metodo="decision_function")["brier"])
    assert np.isnan(compute_metrics(y, distancia)["brier"])  # sin decir el método tampoco
    # las demás métricas no dependen del método
    assert compute_metrics(y, distancia)["pr_auc"] == compute_metrics(y, distancia, metodo="predict_proba")["pr_auc"]


@pytest.mark.parametrize(
    ("score", "metodo", "mensaje"),
    [
        ([0.1, 1.4, 0.6, -0.2, 0.9], "predict_proba", "fuera de"),
        ([0.1, np.nan, 0.6, 0.3, 0.9], "predict_proba", "NaN o infinitos"),
        ([0.1, np.inf, 0.6, 0.3, 0.9], "predict_proba", "NaN o infinitos"),
        ([0.1, 0.8, 0.6, 0.3, 0.9], "proba", "metodo debe ser"),  # un nombre mal escrito no da NaN en silencio
    ],
    ids=["fuera_de_rango", "nan", "infinito", "metodo_desconocido"],
)
def test_casos_invalidos_del_brier(score, metodo, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        compute_metrics([0, 1, 1, 0, 1], score, metodo=metodo)


def test_fuera_de_rango_con_decision_function_no_genera_brier():
    assert np.isnan(compute_metrics([0, 1, 1, 0], [-2.0, 1.5, 0.7, -0.3], metodo="decision_function")["brier"])


def test_el_metodo_sale_del_modelo_no_del_rango():
    modelos = a.modelos_supervisados()
    assert a.metodo_puntaje(modelos["logreg"]) == "predict_proba"  # el campeón actual (regresión logística)
    ridge = next((m for m in modelos.values() if not hasattr(m, "predict_proba")), None)
    if ridge is not None:
        assert a.metodo_puntaje(ridge) == "decision_function"


# ---------------------------------------------------------------- 5. tasa de retiro: advertencia, no rechazo


def _baja_prevalencia(extracto: pd.DataFrame) -> pd.DataFrame:
    """Todos los activos y un solo retirado (1 de 39 = 2,6 %): fuera del 5 %-50 % heurístico."""
    return pd.concat([extracto[extracto["#RETIROS"] == 0], extracto[extracto["#RETIROS"] == 1].head(1)])


def test_prevalencia_distinta_se_advierte_y_no_se_rechaza(extracto):
    lote = _baja_prevalencia(extracto)
    assert k.validar(lote) == []  # no se rechaza por ser distinta de la sintética
    avisos = k.advertencias(lote)
    assert len(avisos) == 1 and avisos[0].startswith("Nivel 2 · tasa_retiro_plausible:")
    assert "observado: 0.02564" in avisos[0] and "no bloquea" in avisos[0]
    assert k.advertencias(extracto) == []  # 24 %: dentro del rango


def test_modo_real_entrena_con_otra_prevalencia_y_avisa(tmp_path, extracto):
    ruta = tmp_path / "anon.parquet"
    an.anonimizar_extracto(_baja_prevalencia(extracto), salt=SAL).to_parquet(ruta)
    with pytest.warns(k.AdvertenciaContrato, match="tasa_retiro_plausible"):
        df, _ = c.datos_eleccion(ruta, origen="real_anonimizado")
    assert len(df) == 39
    with warnings.catch_warnings():
        warnings.simplefilter("error", k.AdvertenciaContrato)
        c.datos_eleccion(GeneradorAsociados(n=200, seed=1).generar().pipe(_a_parquet, tmp_path), origen="sintetico")


def _a_parquet(df: pd.DataFrame, carpeta) -> object:
    ruta = carpeta / "sinteticos.parquet"
    df.to_parquet(ruta)
    return ruta


def test_severidad_configurable_en_el_yaml(tmp_path, extracto):
    texto = k.CONTRATO_ASOCIADOS.read_text(encoding="utf-8")
    assert texto.count("severidad: advertencia") == 1
    estricto = tmp_path / "estricto.yaml"
    estricto.write_text(texto.replace("severidad: advertencia", "severidad: error"), encoding="utf-8")
    contrato = k.cargar(estricto)  # cuando el fondo confirme una regla de negocio, vuelve a bloquear
    errores = k.validar(_baja_prevalencia(extracto), contrato)
    assert any(e.startswith("Nivel 2 · tasa_retiro_plausible:") for e in errores)
    assert k.advertencias(_baja_prevalencia(extracto), contrato) == []
    malo = tmp_path / "malo.yaml"
    malo.write_text(texto.replace("severidad: advertencia", "severidad: quizas"), encoding="utf-8")
    with pytest.raises(ValueError, match="severidad"):
        k.cargar(malo)


# ---------------------------------------------------------------- auditoría: predicción solo de activos


def test_prediccion_solo_de_asociados_activos(campeon_guardado, extracto):
    """El modelo se entrenó con la foto de ACTIVOS en t0: a quien ya se retiró no hay nada que predecirle."""
    salida = p.predecir(extracto, campeon_guardado, origen="sintetico")
    retirados = extracto["FECHARETIRO"].notna().to_numpy()
    assert len(salida) == int((~retirados).sum()) == 38
    assert not retirados[salida["fila"].to_numpy()].any()  # fila = posición en la entrada, sin retirados
    todos = p.predecir(extracto, campeon_guardado, origen="sintetico", solo_activos=False)
    assert list(todos["fila"]) == list(range(len(extracto)))
    sin_columna = p.predecir(extracto.drop(columns="FECHARETIRO"), campeon_guardado, origen="sintetico")
    assert len(sin_columna) == len(extracto)  # sin FECHARETIRO no se puede saber: se predicen todos


# ---------------------------------------------------------------- auditoría: validación temporal


def test_validacion_temporal_elige_el_umbral_sin_ver_la_foto_reciente(monkeypatch):
    df = GeneradorAsociados(n=1500, seed=5, tasa_retiro=0.3).generar()
    modelo = a.modelos_supervisados()["logreg"]
    vistos = []
    original = c.elegir_umbrales

    def espia(y, puntaje, *args, **kw):
        vistos.append(len(y))
        return original(y, puntaje, *args, **kw)

    monkeypatch.setattr(c, "elegir_umbrales", espia)
    resultado = c.validacion_temporal(modelo, df, horizonte_meses=3)
    assert vistos == [resultado["asociados_entrenamiento"]]  # el umbral sale solo de la foto vieja
    assert resultado["t0_entrenamiento"] < resultado["t0_prueba"]
    assert 0 < resultado["retiros_prueba"] < resultado["asociados_prueba"]
    assert 0 <= resultado["pr_auc"] <= 1 and resultado["base_rate"] == pytest.approx(
        resultado["retiros_prueba"] / resultado["asociados_prueba"]
    )
    with pytest.raises(ValueError, match="menos de"):
        c.validacion_temporal(modelo, df.head(40), horizonte_meses=3)


# ---------------------------------------------------------------- auditoría: scripts exploratorios


def test_scripts_exploratorios_solo_con_sinteticos_y_con_contrato(tmp_path, monkeypatch, extracto):
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    with pytest.raises(c.DatosInvalidos, match="exploratorio"):
        c.datos_exploracion("data/ejemplo/asociados_ejemplo.parquet", "make abandono")
    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    assert len(c.datos_exploracion("data/ejemplo/asociados_ejemplo.parquet", "make abandono")) == 50
    malo = tmp_path / "malo.parquet"
    roto = extracto.copy()
    roto.loc[roto.index[0], "ESTADO"] = "VALOR_RARO"
    roto.to_parquet(malo)
    with pytest.raises(c.DatosInvalidos, match="no cumple el contrato"):
        c.datos_exploracion(malo, "make afinar")
