"""Correcciones de la auditoría post-ejecución: simulador con datos del mismo origen (E1), imagen de Docker (E2),
aviso de MLflow (E3) y modelo sin firma, métricas honestas en el simulador (E4), calibración y etiqueta (E5),
matriz de confusión con umbrales iguales (E6), fecha de corte y marca en las figuras."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml import registro as r
from proyecto_ml.figuras import MARCA, marcar_origen
from proyecto_ml.generador import FALTANTES, GeneradorAsociados

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "src" / "proyecto_ml" / "app_abandono.py"


@pytest.fixture(scope="module")
def foto():
    return a.foto(GeneradorAsociados(n=600, seed=11, tasa_retiro=0.25).generar(), 3)


@pytest.fixture(scope="module")
def elegido(foto):
    return c.elegir(foto, folds=3, repeticiones=1, honestos=False)


# ---------------------------------------------------------------- E1: el simulador no mezcla orígenes


@pytest.mark.integration
def test_simulador_se_detiene_si_la_referencia_real_no_es_la_del_campeon(tmp_path, monkeypatch, foto):
    from proyecto_ml.anonymize import anonimizar_extracto

    elegido = c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1)
    elegido.info |= {"origen_datos": config.ORIGEN_REAL, "datos_huella": "otra000huella"}
    c.guardar(elegido, tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    # 1) un dataset anonimizado válido, pero no el del campeón: la huella no coincide
    otro = tmp_path / "otro_anon.parquet"
    anonimizar_extracto(GeneradorAsociados(n=400, seed=21).generar(), salt="clave-de-prueba-123456").to_parquet(otro)
    monkeypatch.setenv("DATOS_ASOCIADOS", str(otro))
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    assert any("huella no coincide" in e.value for e in app.error)
    assert not app.metric  # se detuvo antes de predecir
    # 2) el archivo de ejemplo (con PII) no cumple el contrato para ML: error claro, sin excepción ni valores
    monkeypatch.setenv("DATOS_ASOCIADOS", "data/ejemplo/asociados_ejemplo.parquet")
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception and not app.metric
    assert any("no se pueden usar" in e.value and "contrato" in e.value for e in app.error)


# ---------------------------------------------------------------- E2: la imagen lleva el contrato


def test_el_dockerfile_copia_lo_que_el_codigo_lee():
    dockerfile = (RAIZ / "Dockerfile").read_text(encoding="utf-8")
    for carpeta in ("src", "contratos"):  # contratos/asociados.yaml: validar y huella_contrato
        assert f"COPY {carpeta} ./{carpeta}" in dockerfile
    from proyecto_ml.contrato import CONTRATO_ASOCIADOS

    assert CONTRATO_ASOCIADOS.relative_to(config.ROOT_DIR).parts[0] == "contratos"  # la carpeta que se copia


# ---------------------------------------------------------------- E3 y firma del modelo en MLflow


@pytest.mark.integration
def test_registrar_datos_sin_aviso_y_el_pyfunc_acepta_enteros_y_vacios(tmp_path, monkeypatch, foto):
    """Sin firma a propósito (ver registro.registrar_modelo): el pyfunc acepta el extracto con enteros y con vacíos."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("ORIGEN_DATOS", raising=False)
    X, y = a.separar(foto)
    modelo = c.ModeloZonas(a.modelos_supervisados()["logreg"].fit(X, y), 0.3, 0.7)
    r.configurar("prueba_post")
    with warnings.catch_warnings(record=True) as avisos, r.corrida("x"):
        warnings.simplefilter("always")
        r.registrar_datos(foto, "foto", "eleccion", "prueba")
        info = r.registrar_modelo(modelo)
    assert not [w for w in avisos if "integer column" in str(w.message)]
    pyfunc = mlflow.pyfunc.load_model(info.model_uri)
    con_vacios = a.separar(a.foto(GeneradorAsociados(300, seed=4, faltantes=FALTANTES).generar(), 3))[0]
    for filas in (X.head(50), con_vacios):  # enteros sin vacíos, y conteos con vacíos
        assert set(pd.Series(pyfunc.predict(filas))) <= set(c.ZONAS)


# ---------------------------------------------------------------- E5: calibración y etiqueta del puntaje


def test_brier_skill_y_etiqueta_del_puntaje(foto, elegido):
    tasa = foto[a.TARGET].mean()
    for cand in elegido.candidatos.values():
        m = cand.metricas
        if a.metodo_puntaje(cand.modelo) == "predict_proba":
            assert m["brier_skill"] == pytest.approx(1 - m["brier"] / (tasa * (1 - tasa)))
        else:
            assert np.isnan(m["brier"]) and np.isnan(m["brier_skill"])  # Ridge: no es una probabilidad
    assert c.etiqueta_puntaje("predict_proba", {"brier_skill": 0.79}) == "Probabilidad de retiro"
    assert c.etiqueta_puntaje("predict_proba", {"brier_skill": 0.01}) == "Puntaje de riesgo (no calibrado)"
    assert c.etiqueta_puntaje("predict_proba", {}) == "Probabilidad de retiro"  # campeones guardados antes
    assert c.etiqueta_puntaje("decision_function", {"brier_skill": 0.9}) == "Puntaje del modelo"


# ---------------------------------------------------------------- E6: umbrales iguales, una sola matriz


def test_umbrales_iguales_una_sola_matriz(elegido):
    import copy

    iguales = copy.copy(elegido)
    iguales.umbral_alto = iguales.umbral
    matrices = c.matrices_confusion(iguales)
    assert list(matrices) == ["alerta"]
    assert len(c.figura_confusion(iguales).data) == 1
    distintos = copy.copy(elegido)
    distintos.umbral_alto = elegido.umbral + 0.1
    assert list(c.matrices_confusion(distintos)) == ["alerta", "se retira"]


def test_umbral_alto_igual_al_de_alerta_cuando_f2_ya_cumple_la_precision():
    """Por qué el campeón tiene los dos umbrales en 0,308: el de máximo F2 ya tiene precisión >= 0,8."""
    y = np.r_[np.zeros(950), np.ones(50)].astype(int)
    puntaje = np.r_[np.linspace(0, 0.3, 950), np.linspace(0.6, 1, 50)]  # separación perfecta
    u = c.elegir_umbrales(y, puntaje, beta=2, precision_min=0.8)
    assert u["precision"] >= 0.8 and u["umbral_alto"] == u["umbral"]


# ---------------------------------------------------------------- fecha de corte


def test_fecha_de_corte_configurable(monkeypatch):
    monkeypatch.delenv("FECHA_CORTE", raising=False)
    assert config.fecha_corte() == config.FECHA_CORTE_DEFECTO == str(a.FECHA_CORTE.date())
    monkeypatch.setenv("FECHA_CORTE", "2027-03-31")
    assert config.fecha_corte() == "2027-03-31"


def test_datos_reales_con_fechas_despues_del_corte(tmp_path, monkeypatch):
    archivo = tmp_path / "extracto.parquet"
    GeneradorAsociados(n=80, seed=3).generar().to_parquet(archivo)  # fechas hasta 2026-10
    monkeypatch.setattr(a, "FECHA_CORTE", pd.Timestamp("2026-01-31"))  # un corte anterior a esas fechas
    with pytest.raises(c.DatosInvalidos, match="FECHA_CORTE"):
        c.datos_eleccion(archivo, origen="real_anonimizado")
    df, _ = c.datos_eleccion(archivo, origen="sintetico")  # en modo sintético no aplica
    assert len(df) == 80


# ---------------------------------------------------------------- marca de experimental en las figuras


def test_las_figuras_sinteticas_llevan_la_marca(monkeypatch):
    monkeypatch.delenv("ORIGEN_DATOS", raising=False)
    sintetica = marcar_origen(go.Figure())
    assert [n.text for n in sintetica.layout.annotations] == [MARCA]
    real = marcar_origen(go.Figure(), origen=config.ORIGEN_REAL)
    assert not real.layout.annotations
