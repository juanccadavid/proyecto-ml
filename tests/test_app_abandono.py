"""Prueba del simulador de Streamlit: corre sin errores con modelos de abandono reales (AppTest, sin navegador)."""

from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest

from proyecto_ml import abandono as a
from proyecto_ml import config
from proyecto_ml.generador import GeneradorAsociados

APP = Path(__file__).resolve().parents[1] / "src" / "proyecto_ml" / "app_abandono.py"


@pytest.fixture
def carpeta_modelos(tmp_path, monkeypatch):
    df = GeneradorAsociados(n=200, seed=1).generar()
    X, y = a.preparar(df), df[a.TARGET]
    modelos = a.modelos_supervisados()
    for nombre in ("ridge", "random_forest"):
        joblib.dump(modelos[nombre].fit(X, y), tmp_path / f"abandono_{nombre}.joblib")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return tmp_path


def test_simulador_predice_y_cambia_de_modelo(carpeta_modelos):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.header[0].value.startswith("Riesgo de abandono")
    assert app.metric[0].label == "Puntaje de Ridge"  # el primer modelo guardado, en el orden de TITULOS

    app.radio(key="modelo").set_value("random_forest").run()
    app.slider(key="#INTENCIONRETIROS_ULT3MESES").set_value(4).run()
    assert not app.exception
    assert app.metric[0].label == "Probabilidad de retiro"
    assert app.metric[0].value.endswith("%")


def test_simulador_entrena_si_no_hay_modelos(tmp_path, monkeypatch):
    """Como en Streamlit Community Cloud: sin .joblib, entrena los 4 supervisados al arrancar."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.radio(key="modelo").options == [a.TITULOS[n] for n in a.TITULOS]
    assert "se entrenaron al abrir la app" in app.info[0].value
    assert not list(tmp_path.iterdir())  # en memoria: no escribe en models/
