"""Formulario solo para el equipo: SQL de la tabla de entrada y revisión (make supabase-revisar), sin conexión real."""

import importlib.util
import json
from urllib.parse import parse_qs, urlparse

import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.generador import GeneradorAsociados

_spec = importlib.util.spec_from_file_location("contrato_datos", ROOT_DIR / "scripts" / "contrato_datos.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _entradas(n: int, seed: int = 8, desde: int = 1) -> list[dict]:
    """Filas como las guarda asociados_entrada: columnas del contrato + id, enviado_por y revision."""
    filas = k.a_registros_db(GeneradorAsociados(n=n, seed=seed).generar())
    for i, fila in enumerate(filas, start=desde):
        fila.pop("origen")
        fila.update(id=i, enviado_por="equipo@prueba.co", revision="pendiente")
    return filas


# ---------------------------------------------------------------- SQL


def test_sql_cierra_la_tabla_y_abre_la_entrada_solo_al_equipo():
    sql = k.sql()
    assert "create policy anon_solo_inserta" not in sql and "grant insert on public.asociados to anon" not in sql
    assert "revoke insert on public.asociados from anon;" in sql
    assert "create table if not exists public.asociados_entrada" in sql
    assert "grant insert on public.asociados_entrada to authenticated;" in sql
    assert "public.es_del_equipo() and revision = 'pendiente'" in sql
    assert " to anon" not in sql.replace("from anon", "")  # ningún permiso nuevo para la clave pública
    entrada = sql[sql.index("public.asociados_entrada (") : sql.index("comment on table public.asociados_entrada")]
    assert "cedula bigint not null check" in entrada and " unique" not in entrada  # repetidos: al revisar


def test_columnas_de_la_entrada_no_chocan_con_el_contrato():
    assert not set(k.COLUMNAS_ENTRADA) & {c.db for c in k.cargar().columnas}  # ESTADO ya es "estado"


def test_migracion_002_al_dia():
    """Si cambias el contrato o el SQL del formulario, regenera la migración (contrato.migracion_formulario)."""
    archivo = ROOT_DIR / "supabase" / "migraciones" / "002_formulario_equipo.sql"
    assert archivo.read_text(encoding="utf-8") == k.migracion_formulario()


# ---------------------------------------------------------------- revisar_entradas


def test_aprueba_las_validas_aunque_sean_pocas():
    """Con 3 filas el contrato de lote (mínimo 30, tasa de retiro) no aplica: se revisa fila por fila."""
    decisiones = k.revisar_entradas(_entradas(3), existentes=[])
    assert [d["aprobado"] for d in decisiones] == [True, True, True]
    assert all(d["registro"]["origen"] == "formulario" and "revision" not in d["registro"] for d in decisiones)


def test_rechaza_repetidas_e_invalidas_con_su_motivo():
    entradas = _entradas(4)
    existentes = [{"cedula": entradas[0]["cedula"], "idasociados": -1}]  # ya está en la tabla
    entradas[2]["cedula"] = entradas[1]["cedula"]  # repetida dentro del lote: gana la primera
    entradas[3]["estrato"] = 9  # fuera de 1-6
    decisiones = {d["id"]: d for d in k.revisar_entradas(entradas, existentes)}
    assert not decisiones[1]["aprobado"] and "CEDULA" in decisiones[1]["motivo"]
    assert decisiones[2]["aprobado"]
    assert not decisiones[3]["aprobado"] and "ya existe" in decisiones[3]["motivo"]
    assert not decisiones[4]["aprobado"] and "ESTRATO" in decisiones[4]["motivo"]


# ---------------------------------------------------------------- revisar (CLI) con la API simulada


class _ApiFalsa:
    def __init__(self, tablas: dict[str, list[dict]]):
        self.tablas, self.escrituras = tablas, []

    def __call__(self, req, timeout=None):
        url = urlparse(req.full_url)
        tabla, qs = url.path.rsplit("/", 1)[-1], parse_qs(url.query)
        if req.get_method() == "GET":
            filas = self.tablas[tabla]
            if "revision" in qs:
                filas = [f for f in filas if f["revision"] == qs["revision"][0].removeprefix("eq.")]
            inicio, limite = int(qs["offset"][0]), int(qs["limit"][0])
            cuerpo = json.dumps(filas[inicio : inicio + limite]).encode()
        else:
            self.escrituras.append((req.get_method(), tabla, qs, json.loads(req.data)))
            cuerpo = b""

        class _Respuesta:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return cuerpo

        return _Respuesta()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proyecto.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_de_prueba")
    entradas = _entradas(3)
    entradas[2]["estrato"] = 9
    falsa = _ApiFalsa({"asociados_entrada": entradas, "asociados": []})
    monkeypatch.setattr(cli.urllib.request, "urlopen", falsa)
    return falsa


def test_revisar_sin_aplicar_no_escribe(api, capsys):
    decisiones = cli._revisar(k.cargar())
    assert [d["aprobado"] for d in decisiones] == [True, True, False]
    assert api.escrituras == []
    assert "Nada se escribió" in capsys.readouterr().out


def test_revisar_aplicar_inserta_las_aprobadas_y_marca_todas(api):
    cli._revisar(k.cargar(), aplicar=True)
    inserciones = [e for e in api.escrituras if e[0] == "POST"]
    marcas = [e for e in api.escrituras if e[0] == "PATCH"]
    assert [e[1] for e in inserciones] == ["asociados", "asociados"]
    assert all(e[3][0]["origen"] == "formulario" for e in inserciones)
    assert [(e[2]["id"][0], e[3]["revision"]) for e in marcas] == [
        ("eq.1", "aprobada"),
        ("eq.2", "aprobada"),
        ("eq.3", "rechazada"),
    ]
    assert "ESTRATO" in marcas[2][3]["motivo_revision"]


def test_revisar_rechaza_la_clave_publica(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proyecto.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_publishable_x")
    with pytest.raises(SystemExit, match="clave pública"):
        cli._revisar(k.cargar())
