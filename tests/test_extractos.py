"""Infraestructura para la validación fuera de tiempo con varios extractos (extractos.py y
campeon.validacion_fuera_de_tiempo).

Los "extractos" de estas pruebas se arman recortando un solo archivo generado a distintas fechas: sirven para
probar la interfaz (quién entra, de dónde sale la etiqueta, qué se rechaza), NO para medir nada. Sus conteos no
cambian entre fechas, justo lo contrario de un extracto real.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml import extractos as ex
from proyecto_ml.generador import GeneradorAsociados

FECHAS = ("2026-01-31", "2026-04-30", "2026-07-31")


@pytest.fixture(scope="module")
def base():
    return GeneradorAsociados(n=3000, seed=5, tasa_retiro=0.3).generar()


def recortar(df: pd.DataFrame, fecha: str) -> pd.DataFrame:
    """Cómo se vería el extracto sacado en ``fecha``: sin retiros posteriores, sin quien no había ingresado."""
    fecha = pd.Timestamp(fecha)
    vista = df[pd.to_datetime(df["FECHAINGRESOLABORA"]) <= fecha].copy()
    despues = pd.to_datetime(vista["FECHARETIRO"]) > fecha
    vista.loc[despues, ["FECHARETIRO"]] = pd.NaT
    vista.loc[despues, "#RETIROS"] = 0
    vista.loc[despues, ["ESTADO", "ESTADO_LABORAL", "MOTIVORETIRO"]] = ["ACTIVO", "VIGENTE", ""]
    for columna in ("FECHAULTACTUALIZACION", "ULTIMAACTUALIZACION"):
        vista.loc[despues, columna] = vista.loc[despues, columna].where(
            pd.to_datetime(vista.loc[despues, columna]) <= fecha, fecha
        )
    return vista


@pytest.fixture(scope="module")
def extractos(base):
    return [ex.Extracto.de(recortar(base, f), f) for f in FECHAS]


def test_la_etiqueta_sale_del_extracto_siguiente(base, extractos):
    viejo, nuevo = extractos[0], extractos[1]
    foto = ex.foto_entre_extractos(viejo, nuevo, horizonte_meses=3)
    t0, fin = viejo.fecha, viejo.fecha + pd.DateOffset(months=3)
    retiro = pd.to_datetime(base.set_index("CEDULA").loc[foto["CEDULA"], "FECHARETIRO"]).to_numpy()
    esperado = ((retiro > t0) & (retiro <= fin)).astype(int)
    assert (foto[a.TARGET].to_numpy() == esperado).all() and foto[a.TARGET].sum() > 0
    assert (foto[a.FECHA_OBSERVACION] == t0).all()  # EDAD y ANTIGUEDAD a la fecha del extracto viejo
    assert pd.to_datetime(foto["FECHARETIRO"]).isna().all()  # en el viejo, todos activos: no ven el retiro
    assert len(foto) == len(ex.activos(viejo)) and foto.attrs["sin_seguimiento"] == 0


def test_revisar_par_rechaza_lo_que_no_se_puede_cruzar(extractos):
    viejo, nuevo, _ = extractos
    temprano = ex.Extracto.de(nuevo.datos, "2026-03-15")
    assert any("no cubre el horizonte" in p for p in ex.revisar_par(viejo, temprano, 3))
    otra_sal = ex.Extracto.de(nuevo.datos.assign(CEDULA=nuevo.datos["CEDULA"] + 10**9), nuevo.fecha)
    assert any("ANON_SALT" in p for p in ex.revisar_par(viejo, otra_sal, 3))
    sin_llave = ex.Extracto.de(nuevo.datos.drop(columns="CEDULA"), nuevo.fecha)
    assert any("no comparten ninguna llave" in p for p in ex.revisar_par(viejo, sin_llave, 3))
    futuro = viejo.datos.copy()
    futuro.loc[futuro.index[0], "FECHARETIRO"] = pd.Timestamp("2026-12-31")
    assert any("posteriores a la fecha" in p for p in ex.revisar_par(ex.Extracto.de(futuro, viejo.fecha), nuevo, 3))
    assert ex.revisar_par(viejo, nuevo, 3) == []
    with pytest.raises(ValueError, match="no cubre el horizonte"):
        ex.foto_entre_extractos(viejo, temprano, 3)


def test_fuera_de_tiempo_elige_solo_con_los_pares_viejos(monkeypatch, extractos):
    modelos = a.modelos_supervisados()
    candidatos = {"logreg": modelos["logreg"], "ridge": modelos["ridge"]}
    with pytest.raises(ValueError, match="al menos 3 extractos"):
        c.validacion_fuera_de_tiempo(candidatos, extractos[:2])
    filas = []
    original = c.puntajes_fuera_de_particion

    def espia(modelo, X, y, *args, **kw):
        filas.append(len(X))
        return original(modelo, X, y, *args, **kw)

    monkeypatch.setattr(c, "puntajes_fuera_de_particion", espia)
    resultado = c.validacion_fuera_de_tiempo(candidatos, list(reversed(extractos)), horizonte_meses=3, folds=3)
    assert resultado["t0_entrenamiento"] == [FECHAS[0]] and resultado["t0_prueba"] == FECHAS[1]  # ordena por fecha
    assert filas == [resultado["asociados_entrenamiento"]] * 2  # la elección no ve el par de prueba
    assert resultado["elegido"] in candidatos and 0 <= resultado["pr_auc"] <= 1


def test_script_pide_tres_extractos_y_guarda_solo_agregados(tmp_path, monkeypatch, base):
    spec = importlib.util.spec_from_file_location(
        "fuera_de_tiempo", config.ROOT_DIR / "scripts" / "validacion_fuera_de_tiempo.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    monkeypatch.setattr(script, "METRICS_DIR", tmp_path / "metrics")
    rutas = []
    for fecha in FECHAS:
        ruta = tmp_path / f"extracto_{fecha}.parquet"
        recortar(base, fecha).to_parquet(ruta)
        rutas.append(f"{ruta}:{fecha}")
    with pytest.raises(SystemExit, match="al menos 3"):
        script.main(["--extracto", rutas[0], "--extracto", rutas[1]])
    script.main([arg for r in rutas for arg in ("--extracto", r)] + ["--folds", "3"])
    (salida,) = (tmp_path / "metrics").glob("fuera_de_tiempo_*.json")
    info = json.loads(salida.read_text(encoding="utf-8"))
    assert info["origen_datos"] == "sintetico" and info["t0_prueba"] == FECHAS[1]
    texto = salida.read_text(encoding="utf-8")
    assert not any(str(v) in texto for v in base["CEDULA"].head(50))  # ninguna fila de asociados


# ---------------------------------------------------------------- selección sin fuga entre fotos (4+ extractos)
#
# Con 4 extractos hay dos fotos de entrenamiento y el mismo asociado está en las dos. Estas pruebas usan un modelo
# que REGISTRA con qué filas se entrena y a cuáles puntúa, para comprobar las propiedades, no solo que corra.

FECHAS_4 = ("2025-10-31", "2026-01-31", "2026-04-30", "2026-07-31")
REGISTRO: list[tuple[str, pd.DataFrame]] = []


class Registrador(ClassifierMixin, BaseEstimator):
    """Puntúa con las intenciones de retiro y anota (fit/puntaje, período, identificador) de cada fila."""

    def fit(self, X, y):
        REGISTRO.append(("fit", X[[a.FECHA_OBSERVACION, "CEDULA"]].copy()))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        REGISTRO.append(("puntaje", X[[a.FECHA_OBSERVACION, "CEDULA"]].copy()))
        p = np.clip(X["#INTENCIONRETIROS_ULT3MESES"].fillna(0).to_numpy() / 5, 0, 1)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def extractos_4(base):
    return [ex.Extracto.de(recortar(base, f), f) for f in FECHAS_4]


@pytest.fixture
def registro():
    REGISTRO.clear()
    yield REGISTRO
    REGISTRO.clear()


def test_hacia_adelante_nunca_entrena_con_el_periodo_que_valida_ni_con_uno_posterior(extractos_4, registro):
    fotos = [ex.foto_entre_extractos(v, n, 3) for v, n in zip(extractos_4, extractos_4[1:], strict=False)]
    datos = pd.concat(fotos, ignore_index=True)  # 3 períodos, casi todos los asociados en los 3
    assert datos["CEDULA"].duplicated().sum() > 2000
    X, y = a.separar(datos)
    puntajes, validadas = c.puntajes_hacia_adelante(Registrador(), X, y, datos[a.FECHA_OBSERVACION])
    pares = list(zip(registro[::2], registro[1::2], strict=True))
    assert len(pares) == 2 and all(f == "fit" and p == "puntaje" for (f, _), (p, _) in pares)
    for (_, entreno), (_, valido) in pares:
        assert entreno[a.FECHA_OBSERVACION].max() < valido[a.FECHA_OBSERVACION].min()  # nunca el futuro
        # el mismo asociado puede estar en los dos, pero solo con un estado ANTERIOR: nunca la misma fila
        filas_entreno = set(zip(entreno["CEDULA"], entreno[a.FECHA_OBSERVACION], strict=True))
        assert not filas_entreno & set(zip(valido["CEDULA"], valido[a.FECHA_OBSERVACION], strict=True))
    primero = datos[a.FECHA_OBSERVACION].min()
    assert (validadas == (datos[a.FECHA_OBSERVACION] > primero).to_numpy()).all() and len(puntajes) == validadas.sum()


def test_la_prueba_final_no_participa_en_la_seleccion_ni_en_el_umbral(monkeypatch, extractos_4, registro):
    umbrales_con = []
    original = c.elegir_umbrales

    def espia(y, puntaje, *args, **kw):
        umbrales_con.append(len(y))
        return original(y, puntaje, *args, **kw)

    monkeypatch.setattr(c, "elegir_umbrales", espia)
    resultado = c.validacion_fuera_de_tiempo({"registrador": Registrador()}, extractos_4, horizonte_meses=3, folds=3)
    prueba = pd.Timestamp(resultado["t0_prueba"])
    assert resultado["t0_entrenamiento"] == list(FECHAS_4[:2]) and "hacia adelante" in resultado["seleccion"]
    *seleccion, (ultimo_tipo, ultimo) = registro
    # todo lo de la selección (y el entrenamiento final) es de las fotos viejas; la prueba solo se puntúa al final
    assert all((filas[a.FECHA_OBSERVACION] < prueba).all() for _, filas in seleccion)
    assert ultimo_tipo == "puntaje" and (ultimo[a.FECHA_OBSERVACION] == prueba).all()
    # el umbral sale de las filas validadas hacia adelante (la segunda foto), no de la primera ni de la prueba
    segunda = ex.foto_entre_extractos(extractos_4[1], extractos_4[2], 3)
    assert umbrales_con == [len(segunda)] == [resultado["filas_seleccion"]]


def test_el_identificador_no_es_variable_del_modelo(extractos_4):
    foto = ex.foto_entre_extractos(extractos_4[0], extractos_4[1], 3)
    X, y = a.separar(foto)
    modelo = a.modelos_supervisados()["logreg"].fit(X, y)
    otro = X.assign(CEDULA=X["CEDULA"].sample(frac=1, random_state=0).to_numpy() + 7)  # otros identificadores
    assert np.allclose(a.puntaje(modelo, X), a.puntaje(modelo, otro))
    assert not set(ex.LLAVES) & set(modelo[:-1].get_feature_names_out())


def test_casos_sin_datos_suficientes_fallan_con_un_mensaje_claro(extractos_4):
    fotos = [ex.foto_entre_extractos(v, n, 3) for v, n in zip(extractos_4, extractos_4[1:], strict=False)]
    X, y = a.separar(fotos[0])
    with pytest.raises(ValueError, match="al menos 2 períodos"):
        c.puntajes_hacia_adelante(Registrador(), X, y, fotos[0][a.FECHA_OBSERVACION])
    sin_retiros = pd.concat([fotos[0], fotos[1].assign(**{a.TARGET: 0})], ignore_index=True)
    X, y = a.separar(sin_retiros)
    with pytest.raises(ValueError, match="no tiene las dos clases"):
        c.puntajes_hacia_adelante(Registrador(), X, y, sin_retiros[a.FECHA_OBSERVACION])
    repetida = pd.concat([fotos[0], fotos[0].head(5)], ignore_index=True)  # el mismo asociado dos veces, misma foto
    with pytest.raises(ValueError, match="se repite dentro de una misma foto"):
        c.elegir_y_medir({"registrador": Registrador()}, repetida, fotos[2], folds=3)
    with pytest.raises(ValueError, match="foto de prueba tiene menos de"):
        c.elegir_y_medir({"registrador": Registrador()}, fotos[0], fotos[2].head(3), folds=3)


def test_con_tres_extractos_la_seleccion_sigue_siendo_estratificada(extractos):
    resultado = c.validacion_fuera_de_tiempo(
        {"logreg": a.modelos_supervisados()["logreg"]}, extractos, horizonte_meses=3, folds=3
    )
    assert resultado["seleccion"].startswith("StratifiedKFold")  # una sola foto: cada asociado una vez
    assert resultado["filas_seleccion"] == resultado["asociados_entrenamiento"]


def test_hacia_adelante_con_los_candidatos_del_proyecto(extractos_4):
    primera = ex.foto_entre_extractos(extractos_4[0], extractos_4[1], 3)
    candidatos = c.lista_candidatos(a.separar(primera)[0])
    resultado = c.validacion_fuera_de_tiempo(candidatos, extractos_4, horizonte_meses=3, folds=3)
    assert set(resultado["pr_auc_vieja"]) == set(candidatos) and resultado["elegido"] in candidatos
    assert "hacia adelante" in resultado["seleccion"] and 0 <= resultado["pr_auc"] <= 1
    assert not {str(v) for v in primera["CEDULA"].head(20)} & {str(v) for v in resultado.values()}  # sin ids
