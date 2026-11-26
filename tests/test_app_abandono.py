"""El simulador de Streamlit: arranca, puntúa, explica y recarga selectivamente lo que cambió (AppTest)."""

import ast
import json
import os
import runpy
from collections import Counter
from pathlib import Path

import joblib
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml.generador import NOMBRE_DESARROLLO, GeneradorAsociados

APP = Path(__file__).resolve().parents[1] / "src" / "proyecto_ml" / "app_abandono.py"


def _secciones_formulario() -> dict[str, list[str]]:
    """Lee la configuración sin ejecutar la aplicación de Streamlit."""
    texto = APP.read_text(encoding="utf-8")
    modulo = ast.parse(texto)
    asignacion = next(
        nodo
        for nodo in modulo.body
        if isinstance(nodo, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "SECCIONES_FORMULARIO" for x in nodo.targets)
    )
    return ast.literal_eval(asignacion.value)


def test_simulador_usa_antiguedades_definitivas():
    texto = APP.read_text(encoding="utf-8")
    assert "ANTIGUEDAD_F_MESES" in texto
    assert "ANTIGUEDAD_LA_MESES" in texto
    assert '"ANTIGUEDAD"' not in texto


def test_simulador_no_referencia_variables_eliminadas():
    texto = APP.read_text(encoding="utf-8")
    for antigua in ("#INTENCIONRETIROS_ULT3MESES", "TIPOPACTO", "PERIODODEDUCE", "TIPOCUENTA"):
        assert antigua not in texto


def test_simulador_expone_todas_las_variables_del_modelo_una_vez():
    campos = [campo for seccion in _secciones_formulario().values() for campo in seccion]
    assert Counter(campos) == Counter(a.NUMERICAS + a.CATEGORICAS)


def test_streamlit_app_ejecuta_el_simulador(monkeypatch):
    llamadas = []
    monkeypatch.setattr(runpy, "run_path", lambda ruta, run_name: llamadas.append((Path(ruta), run_name)))
    entrada = Path("streamlit_app.py").resolve()
    exec(compile(entrada.read_text(encoding="utf-8"), str(entrada), "exec"), {"__file__": str(entrada)})
    assert llamadas == [(entrada.parent / "src" / "proyecto_ml" / "app_abandono.py", "__main__")]


def _guardar_campeon(carpeta: Path, seed: int):
    foto = a.foto(GeneradorAsociados(n=900, seed=seed).generar())
    elegido = c.elegir(foto, candidatos={"logreg": a.modelos_supervisados()["logreg"]}, repeticiones=1, honestos=False)
    c.guardar(elegido, carpeta)
    return elegido


def _escribir_desarrollo(carpeta: Path, seed: int) -> Path:
    """Los datos de referencia del simulador (los de make datos-desarrollo), chicos para que la prueba sea rápida."""
    ruta = carpeta / f"{NOMBRE_DESARROLLO}.parquet"
    GeneradorAsociados(n=600, seed=seed, tasa_retiro=0.15).generar().to_parquet(ruta, index=False)
    return ruta


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """models/ y data/processed/ en carpetas temporales, y las cachés de Streamlit vacías."""
    modelos, procesados = tmp_path / "models", tmp_path / "processed"
    modelos.mkdir()
    procesados.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", modelos)
    monkeypatch.setattr(config, "PROCESSED_DIR", procesados)
    st.cache_data.clear()
    st.cache_resource.clear()
    _escribir_desarrollo(procesados, seed=3)
    return modelos, procesados


def _correr() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=240).run()


def _puntaje(app: AppTest) -> float:
    return float(app.metric[0].value.rstrip("%"))


def test_simulador_arranca_puntua_y_responde_a_las_senales(entorno):
    modelos, _ = entorno
    _guardar_campeon(modelos, seed=1)
    app = _correr()
    assert not app.exception
    assert app.metric and any(h.value.startswith("Riesgo de abandono") for h in app.header)
    assert any("MODO SINTÉTICO" in w.value for w in app.warning)
    base = _puntaje(app)
    for senal in ("#SOLICITUD_CANCELACIONAHORROS", "#SOLICITUD_RETIROPARCIALAHORROS"):
        control = app.slider(key=senal)
        control.set_value(control.max).run()
    app.radio(key="#SOLICITUD_DISMINUCIONCUOTAS").set_value("SI").run()
    assert not app.exception
    assert _puntaje(app) > base  # pedir salir sube el puntaje (la regla del generador)


@pytest.mark.integration
def test_simulador_elige_el_campeon_si_no_hay(entorno):
    """Como en Streamlit Community Cloud: sin campeón en models/, lo elige al arrancar, en memoria."""
    modelos, _ = entorno
    app = _correr()
    assert not app.exception
    assert any("se eligió al abrir la app" in i.value for i in app.info)
    assert not list(modelos.iterdir())  # en memoria: no escribe en models/


def test_simulador_con_un_campeon_corrupto_no_muestra_traceback(entorno):
    modelos, _ = entorno
    for archivo in (c.ARCHIVO_MODELO, c.ARCHIVO_GRUPOS):  # un guardado a medias: los archivos están, pero rotos
        (modelos / archivo).write_bytes(b"no es un modelo")
    (modelos / c.ARCHIVO_INFO).write_text("{}", encoding="utf-8")
    app = _correr()
    assert not app.exception
    assert any("No se pudo leer el campeón" in e.value for e in app.error)
    assert all(str(modelos) not in e.value for e in app.error)  # sin rutas


def test_simulador_con_un_campeon_de_otras_variables_dice_por_que(entorno):
    modelos, _ = entorno
    _guardar_campeon(modelos, seed=1)
    ruta = modelos / c.ARCHIVO_INFO
    info = json.loads(ruta.read_text(encoding="utf-8"))
    ruta.write_text(json.dumps({**info, "variables": [*info["variables"], "CARGO"]}), encoding="utf-8")
    app = _correr()
    assert not app.exception
    assert any("otras variables" in e.value and "CARGO" in e.value for e in app.error)


def test_simulador_usa_el_campeon_nuevo_sin_reiniciar(entorno, monkeypatch):
    """Cambiar el modelo reconstruye SHAP; cambiar solo umbral no relee datos ni reconstruye SHAP."""
    from proyecto_ml import explicar as e

    modelos, _ = entorno
    construidos, lecturas = [], []
    original_shap = e.Explicador.__init__
    original_datos = c.datos_eleccion

    def contar_shap(self, modelo, *args, **kw):
        construidos.append(modelo)
        original_shap(self, modelo, *args, **kw)

    monkeypatch.setattr(e.Explicador, "__init__", contar_shap)
    monkeypatch.setattr(c, "datos_eleccion", lambda *args, **kw: lecturas.append(1) or original_datos(*args, **kw))

    viejo = _guardar_campeon(modelos, 1)
    firma_vieja = c.firma(modelos)
    app = _correr()
    assert f"{viejo.umbral:.1%}" in " ".join(m.value for m in app.markdown)
    lecturas_iniciales = len(lecturas)
    _correr()  # sin cambios: todo sale de caché
    assert len(construidos) == 1 and len(lecturas) == lecturas_iniciales

    # Cambiar solo el umbral en campeon.json (misma ruta): actualiza zona y umbral sin reconstruir SHAP ni releer datos.
    ruta_info = modelos / c.ARCHIVO_INFO
    info = json.loads(ruta_info.read_text(encoding="utf-8"))
    info["umbral"], info["umbral_alto"] = 0.001, 0.002
    ruta_info.write_text(json.dumps(info), encoding="utf-8")
    app_umbral = _correr()
    assert not app_umbral.exception
    assert len(app_umbral.error) == 1 and "Se retira" in app_umbral.error[0].value  # modelo.predict usa el umbral nuevo
    assert len(construidos) == 1 and len(lecturas) == lecturas_iniciales  # invalidación selectiva: SHAP intacto

    # Cambiar el campeón completo (misma ruta): reconstruye SHAP pero sigue sin releer los datos de referencia.
    nuevo = _guardar_campeon(modelos, 2)
    assert c.firma(modelos) != firma_vieja
    app = _correr()
    textos = " ".join(m.value for m in app.markdown)
    assert not app.exception
    assert f"{nuevo.umbral:.1%}" in textos and f"{viejo.umbral:.1%}" not in textos
    assert len(construidos) == 2 and construidos[-1].umbral == nuevo.umbral
    assert len(lecturas) == lecturas_iniciales


def test_simulador_invalida_al_cambiar_artefacto_kmeans_o_variables_predictoras(entorno, monkeypatch):
    """Cambiar KMeans (incluso conservando mtime) o predictores invalida cargar_campeon."""
    modelos, _ = entorno
    _guardar_campeon(modelos, 1)
    assert not _correr().exception

    # 1) Cambiar abandono_kmeans.joblib conservando mtime invalida la caché del campeón.
    ruta_kmeans = modelos / c.ARCHIVO_GRUPOS
    stat_viejo = ruta_kmeans.stat()
    cargas = []
    orig_cargar = c.cargar
    monkeypatch.setattr(c, "cargar", lambda carpeta: cargas.append(1) or orig_cargar(carpeta))
    otro_kmeans = a.AgrupadorAbandono(random_state=77).fit(GeneradorAsociados(n=300, seed=77).generar())
    joblib.dump(otro_kmeans, ruta_kmeans)
    os.utime(ruta_kmeans, ns=(stat_viejo.st_atime_ns, stat_viejo.st_mtime_ns))
    assert not _correr().exception and len(cargas) == 1

    # 2) Si cambian las variables predictoras sin cambiar los archivos de models/, cargar_campeon se invalida y avisa.
    monkeypatch.setattr(a, "NUMERICAS", [col for col in a.NUMERICAS if col != "#RETIROS"])
    app = _correr()
    assert not app.exception
    assert any("otras variables" in err.value and "#RETIROS" in err.value for err in app.error)


def test_simulador_recalcula_si_cambian_datos_o_env_conservando_ruta(entorno, tmp_path, monkeypatch):
    """Cambiar datos o .env en la misma ruta invalida cachés afectadas, sin recargar el campeón."""
    modelos, procesados = entorno
    _guardar_campeon(modelos, 1)
    env_temp = tmp_path / ".env"
    env_temp.write_text("FECHA_CORTE=2026-03-31\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_temp)
    monkeypatch.setattr(config, "_DOTENV_FIRMA", None)
    monkeypatch.delenv("FECHA_CORTE", raising=False)

    lecturas, cargas_campeon = [], []
    orig_datos = c.datos_eleccion
    orig_cargar = c.cargar
    monkeypatch.setattr(c, "datos_eleccion", lambda *args, **kw: lecturas.append(1) or orig_datos(*args, **kw))
    monkeypatch.setattr(c, "cargar", lambda carpeta: cargas_campeon.append(1) or orig_cargar(carpeta))

    assert not _correr().exception
    leidas = len(lecturas)
    assert leidas >= 1 and len(cargas_campeon) == 1
    _correr()  # mismo contenido: todo sale de la caché
    assert len(lecturas) == leidas and len(cargas_campeon) == 1

    # 1) Misma ruta de datos, otro contenido -> recalcula referencia, NO recarga el campeón de models/.
    _escribir_desarrollo(procesados, seed=4)
    assert not _correr().exception
    assert len(lecturas) > leidas and len(cargas_campeon) == 1
    leidas_tras_datos = len(lecturas)

    # 2) Nueva FECHA_CORTE en la misma ruta invalida datos_referencia sin recargar el campeón.
    env_temp.write_text("FECHA_CORTE=2026-06-30\n", encoding="utf-8")
    assert config.fecha_corte() == "2026-06-30"
    assert not _correr().exception
    assert len(lecturas) > leidas_tras_datos and len(cargas_campeon) == 1


def test_marca_de_datos_cambia_con_el_contenido_y_no_con_la_ruta(tmp_path):
    from proyecto_ml.datos_campeon import marca_datos

    ruta = _escribir_desarrollo(tmp_path, seed=1)
    primera = marca_datos(ruta)
    assert marca_datos(ruta) == primera  # mismo contenido, misma marca
    _escribir_desarrollo(tmp_path, seed=2)
    assert marca_datos(ruta) != primera
    assert marca_datos(tmp_path / "no_existe.parquet").startswith("ausente:")


def test_marca_datos_en_memoria_incluye_toda_la_configuracion_del_generador(monkeypatch):
    from proyecto_ml import datos_campeon as dc
    from proyecto_ml import generador as g

    monkeypatch.setattr(dc, "archivo_eleccion", lambda ruta=None: None)
    primera = dc.marca_datos()
    monkeypatch.setattr(g, "EMPRESAS", g.EMPRESAS + 1)
    assert dc.marca_datos() != primera


def test_mapa_sin_detalle_individual_con_datos_reales():
    from proyecto_ml import segmentacion as sg

    df = a.foto(GeneradorAsociados(n=400, seed=2).generar())
    segmentacion = sg.Segmentacion().fit(df)
    con = sg.figura_mapa(segmentacion, df)
    sin = sg.figura_mapa(segmentacion, df, detalle_individual=False)
    puntos = [t for t in sin.data if t.mode == "markers"]
    assert puntos and all(t.customdata is None and "se retiró" not in t.hovertemplate for t in puntos)
    assert any(t.customdata is not None for t in con.data if t.mode == "markers")
