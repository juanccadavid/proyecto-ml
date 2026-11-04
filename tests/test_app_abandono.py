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


def test_simulador_rechaza_edad_y_antiguedad_imposibles(carpeta_campeon):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.number_input(key="EDAD").set_value(30.0).run()
    app.number_input(key="ANTIGUEDAD").set_value(20.0).run()  # habría ingresado a los 10 años
    assert not app.exception
    assert any("Combinación imposible" in e.value for e in app.error)
    assert not app.metric  # no puntúa a un asociado que no puede existir
    # 26,9 años e ingreso a los 18 se escribe 26 y 9 (edad entera, antigüedad en medios años): sí existe
    app.number_input(key="EDAD").set_value(26.0).run()
    app.number_input(key="ANTIGUEDAD").set_value(9.0).run()
    assert not any("Combinación imposible" in e.value for e in app.error) and app.metric


def test_simulador_con_un_campeon_corrupto_no_muestra_traceback(tmp_path, monkeypatch):
    for archivo in (c.ARCHIVO_MODELO, c.ARCHIVO_GRUPOS):  # un guardado a medias: los archivos están, pero rotos
        (tmp_path / archivo).write_bytes(b"no es un modelo")
    (tmp_path / c.ARCHIVO_INFO).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert any("No se pudo leer el campeón" in e.value for e in app.error)
    assert all(str(tmp_path) not in e.value for e in app.error)  # sin rutas


def test_mapa_sin_detalle_individual_con_datos_reales():
    from proyecto_ml import segmentacion as sg

    df = a.foto(GeneradorAsociados(n=400, seed=2).generar(), 3)
    segmentacion = sg.Segmentacion().fit(df)
    con = sg.figura_mapa(segmentacion, df)
    sin = sg.figura_mapa(segmentacion, df, detalle_individual=False)
    puntos = [t for t in sin.data if t.mode == "markers"]
    assert puntos and all(t.customdata is None and "se retiró" not in t.hovertemplate for t in puntos)
    assert any(t.customdata is not None for t in con.data if t.mode == "markers")


def test_simulador_con_campeon_real_no_muestra_detalle_individual(tmp_path, monkeypatch):
    """Con un campeón real (y su misma referencia anonimizada), la app pide el mapa sin datos por persona."""
    from proyecto_ml import registro as r
    from proyecto_ml import segmentacion as sg
    from proyecto_ml.anonymize import anonimizar_extracto

    ruta = tmp_path / "anon.parquet"
    anonimizar_extracto(GeneradorAsociados(n=900, seed=1).generar(), salt="clave-de-prueba-123456").to_parquet(ruta)
    monkeypatch.setenv("DATOS_ASOCIADOS", str(ruta))
    foto = a.foto(c.datos_eleccion(ruta, origen=config.ORIGEN_REAL)[0], 3)
    elegido = c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1)
    elegido.info |= {"origen_datos": config.ORIGEN_REAL, "datos_huella": r.huella(foto), "horizonte_meses": 3}
    carpeta = tmp_path / "modelos"
    c.guardar(elegido, carpeta)
    monkeypatch.setattr(config, "MODELS_DIR", carpeta)
    llamadas = []
    original = sg.figura_mapa
    monkeypatch.setattr(sg, "figura_mapa", lambda *args, **kw: llamadas.append(kw) or original(*args, **kw))
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    assert llamadas and llamadas[-1]["detalle_individual"] is False


def _guardar_campeon(carpeta, seed):
    foto = a.foto(GeneradorAsociados(n=900, seed=seed).generar(), 3)
    elegido = c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1)
    c.guardar(elegido, carpeta)
    return elegido


def test_simulador_usa_el_campeon_nuevo_sin_reiniciar(tmp_path, monkeypatch):
    """Antes, ``cargar_campeon`` tenía como llave solo la carpeta: tras ``make campeon`` la app seguía con el viejo."""
    from proyecto_ml import explicar as e

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    construidos = []
    original = e.Explicador.__init__

    def contar(self, modelo, *args, **kw):
        construidos.append(modelo)
        original(self, modelo, *args, **kw)

    monkeypatch.setattr(e.Explicador, "__init__", contar)
    viejo = _guardar_campeon(tmp_path, 1)
    firma_vieja = c.firma(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert f"{viejo.umbral:.1%}" in " ".join(m.value for m in app.markdown)
    AppTest.from_file(str(APP), default_timeout=180).run()  # sin cambios: nada se vuelve a construir
    assert len(construidos) == 1

    nuevo = _guardar_campeon(tmp_path, 2)  # como correr make campeon con la app abierta
    assert c.firma(tmp_path) != firma_vieja
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    textos = " ".join(m.value for m in app.markdown)
    assert not app.exception
    assert f"{nuevo.umbral:.1%}" in textos and f"{viejo.umbral:.1%}" not in textos
    assert len(construidos) == 2 and construidos[-1].umbral == nuevo.umbral  # SHAP con el modelo nuevo


def test_firma_del_campeon(tmp_path):
    assert c.firma(tmp_path) is None  # sin campeón
    _guardar_campeon(tmp_path, 1)
    assert c.firma(tmp_path) == c.firma(tmp_path)  # determinista: no cambia si no se guarda de nuevo
