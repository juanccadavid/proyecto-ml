"""Prueba del simulador de Streamlit con el campeón y sus tres zonas (AppTest, sin navegador)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml.generador import GeneradorAsociados

APP = Path(__file__).resolve().parents[1] / "src" / "proyecto_ml" / "app_abandono.py"


@pytest.fixture
def carpeta_campeon(tmp_path, monkeypatch):
    foto = a.foto(GeneradorAsociados(n=900, seed=1).generar(), 3)
    elegido = c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1)
    c.guardar(elegido, tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return tmp_path


def _mensaje(app) -> str:
    """El mensaje de la zona (el banner del modo sintético también es un aviso, así que se busca por su texto)."""
    textos = [m.value for m in (*app.success, *app.warning, *app.error)]
    return next(t for t in textos if t.startswith(("Se queda", "Zona gris", "Se retira")))


def test_simulador_usa_el_campeon_y_sus_zonas(carpeta_campeon):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.metric[0].label == "Probabilidad de retiro"
    assert app.header[0].value == "Riesgo de abandono: 😀" and _mensaje(app).startswith("Se queda")
    assert any(s.value == "Grupo de comportamiento: estable" for s in app.subheader)
    assert any("MODO SINTÉTICO" in w.value for w in app.warning)  # el simulador dice que los datos son sintéticos

    for columna in ("#INTENCIONRETIROS_ULT3MESES", "#SOLICITUD_CANCELACIONAHORROS", "#SOLICITUD_DISMINUCIONCUOTAS"):
        app.slider(key=columna).set_value(3).run()
    assert not app.exception
    assert app.header[0].value == "Riesgo de abandono: 😟" and _mensaje(app).startswith("Se retira")
    assert any(s.value == "Grupo de comportamiento: riesgo" for s in app.subheader)


@pytest.mark.integration
def test_simulador_elige_el_campeon_si_no_hay(tmp_path, monkeypatch):
    """Como en Streamlit Community Cloud: sin campeón en models/, lo elige al arrancar, en memoria."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    assert "se eligió al abrir la app" in app.info[0].value
    assert not list(tmp_path.iterdir())  # en memoria: no escribe en models/
