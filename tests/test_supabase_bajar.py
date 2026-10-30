"""Descarga de Supabase a data/raw (make supabase-bajar), con la API simulada: no hay conexión real."""

import importlib.util
import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.generador import GeneradorAsociados

# scripts/ no es un paquete: el CLI se carga desde su ruta para probar sus funciones.
_spec = importlib.util.spec_from_file_location("contrato_datos", ROOT_DIR / "scripts" / "contrato_datos.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _filas_bd(n: int, seed: int = 8) -> list[dict]:
    """Filas como las devuelve Supabase: nombres de la BD + id, creado_en y origen."""
    filas = k.a_registros_db(GeneradorAsociados(n=n, seed=seed).generar(), origen="formulario")
    for i, fila in enumerate(filas, start=1):
        fila.update(id=i, creado_en="2026-10-30T22:00:00+00:00")
    return filas


class _ApiFalsa:
    """Responde GET /rest/v1/<tabla>?limit=&offset= como PostgREST y anota cada URL pedida."""

    def __init__(self, filas: list[dict]):
        self.filas, self.urls = filas, []

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        qs = parse_qs(urlparse(req.full_url).query)
        inicio, limite = int(qs["offset"][0]), int(qs["limit"][0])
        cuerpo = json.dumps(self.filas[inicio : inicio + limite]).encode()

        class _Respuesta:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return cuerpo

        return _Respuesta()


@pytest.fixture
def entorno(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://proyecto.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_de_prueba")

    def preparar(filas: list[dict]) -> _ApiFalsa:
        api = _ApiFalsa(filas)
        monkeypatch.setattr(cli.urllib.request, "urlopen", api)
        return api

    preparar.salida = tmp_path / "raw" / "supabase_asociados.parquet"
    return preparar


def test_ida_y_vuelta_bd_contrato():
    original = GeneradorAsociados(n=60, seed=8).generar()
    vuelta = k.desde_registros_db(_filas_bd(60))
    assert list(vuelta.columns) == k.cargar().nombres  # sin id, creado_en ni origen
    assert k.validar(vuelta) == []
    a_ns = lambda d: d.apply(lambda s: s.astype("datetime64[ns]") if s.dtype.kind == "M" else s)  # noqa: E731
    pd.testing.assert_frame_equal(a_ns(vuelta), a_ns(original), check_dtype=False)


def test_lee_la_tabla_completa_por_paginas(entorno):
    api = entorno(_filas_bd(2500))
    filas = cli.leer_tabla("https://proyecto.supabase.co", "sb_secret_x", "asociados", pagina=1000)
    assert len(filas) == 2500
    assert [parse_qs(urlparse(u).query)["offset"][0] for u in api.urls] == ["0", "1000", "2000"]


def test_baja_valida_y_guarda_ultima_y_fechada(entorno, capsys):
    entorno(_filas_bd(60))
    cli._bajar(entorno.salida, k.cargar())
    guardados = {p.name for p in entorno.salida.parent.iterdir()}
    fechadas = {g for g in guardados if g.startswith("supabase_asociados_2")}
    assert "supabase_asociados.parquet" in guardados and len(fechadas) == 1  # última + copia fechada
    assert len(pd.read_parquet(entorno.salida)) == 60
    assert "formulario: 60" in capsys.readouterr().out


def test_no_guarda_un_lote_que_no_cumple_el_contrato(entorno):
    entorno(_filas_bd(10))  # 10 filas: falla volumen_minimo (nivel 2), una ingesta cortada
    with pytest.raises(SystemExit, match="no cumple el contrato"):
        cli._bajar(entorno.salida, k.cargar())
    assert not entorno.salida.exists()
    cli._bajar(entorno.salida, k.cargar(), forzar=True)  # --forzar: guarda bajo tu responsabilidad
    assert entorno.salida.exists()


def test_rechaza_la_clave_publica(entorno, monkeypatch):
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_publishable_abc")
    with pytest.raises(SystemExit, match="clave pública"):
        cli._bajar(entorno.salida, k.cargar())


def test_sin_clave_o_tabla_vacia(entorno, monkeypatch):
    entorno([])
    with pytest.raises(SystemExit, match="vacía"):
        cli._bajar(entorno.salida, k.cargar())
    monkeypatch.delenv("SUPABASE_SECRET_KEY")
    with pytest.raises(SystemExit, match="SUPABASE_SECRET_KEY"):
        cli._bajar(entorno.salida, k.cargar())
