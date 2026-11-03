"""Formulario solo para el equipo: SQL de la política en la tabla y migración 003, sin conexión real."""

import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import ROOT_DIR


def test_sql_solo_el_equipo_inserta_directo_en_la_tabla():
    sql = k.sql()
    assert "create policy anon_solo_inserta" not in sql and "grant insert on public.asociados to anon" not in sql
    assert "revoke insert on public.asociados from anon;" in sql
    assert " to anon" not in sql.replace("from anon", "")  # ningún permiso para la clave pública sin sesión
    assert "grant insert on public.asociados to authenticated;" in sql
    assert "create policy equipo_inserta on public.asociados for insert to authenticated" in sql
    assert "public.es_del_equipo() and origen = 'formulario'" in sql
    assert "grant select" not in sql and "for select" not in sql  # el equipo inserta, no lee
    assert "_entrada" not in sql


def test_la_tabla_mantiene_las_columnas_unicas():
    tabla = k.sql().split("comment on table")[0]
    assert "cedula bigint not null unique check" in tabla and "idasociados bigint not null unique check" in tabla


def test_contrato_web_sin_tabla_de_entrada():
    assert "tabla_entrada" not in k.js()


def test_migracion_003_al_dia():
    """Si cambias el contrato o el SQL del formulario, regenera la migración (contrato.migracion_formulario)."""
    archivo = ROOT_DIR / "supabase" / "migraciones" / "003_formulario_directo.sql"
    assert archivo.read_text(encoding="utf-8") == k.migracion_formulario()
    assert "drop table if exists public.asociados_entrada;" in archivo.read_text(encoding="utf-8")


def test_migraciones_validas_para_postgres():
    pglast = pytest.importorskip("pglast")
    for archivo in sorted((ROOT_DIR / "supabase" / "migraciones").glob("*.sql")):
        pglast.parse_sql(archivo.read_text(encoding="utf-8"))
    pglast.parse_sql(k.sql())
