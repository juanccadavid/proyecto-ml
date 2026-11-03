"""Crea las tablas de Supabase corriendo supabase/asociados.sql por la conexión directa a Postgres.

Para cuando el SQL Editor no es opción (o se borró la tabla). Es idempotente: lo que ya existe (la tabla
equipo y sus correos) no se toca. Al final quita la tabla de entrada vieja y avisa a la API de la tabla nueva.

    make supabase-crear

Necesita SUPABASE_DB_URL en el .env: Supabase > Connect > Connection string > Session pooler, con la
contraseña de la base en lugar de [YOUR-PASSWORD]. Solo en el .env: nunca en Git, en la web ni en un chat.
"""

from __future__ import annotations

import os
import sys

import psycopg

from proyecto_ml.config import ROOT_DIR  # carga el .env

RESTAURAR = """
drop table if exists public.asociados_entrada;
notify pgrst, 'reload schema';
"""


def main() -> None:
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        sys.exit("Falta SUPABASE_DB_URL en el .env (Supabase > Connect > Session pooler; ver .env.example)")
    if "[YOUR-PASSWORD]" in url:
        sys.exit("SUPABASE_DB_URL todavía dice [YOUR-PASSWORD]: cámbialo por la contraseña de la base")
    sql = (ROOT_DIR / "supabase" / "asociados.sql").read_text(encoding="utf-8") + RESTAURAR
    with psycopg.connect(url, connect_timeout=20) as conn:  # una transacción: si algo falla, no queda nada a medias
        conn.execute(sql)
        columnas = conn.execute(
            "select count(*) from information_schema.columns where table_schema = 'public' and table_name = 'asociados'"
        ).fetchone()[0]
        filas = conn.execute("select count(*) from public.asociados").fetchone()[0]
        equipo = conn.execute("select count(*) from public.equipo").fetchone()[0]
        politicas = [p for (p,) in conn.execute("select policyname from pg_policies where tablename = 'asociados'")]
    print(f"public.asociados: {columnas} columnas, {filas} filas; políticas: {', '.join(politicas) or 'ninguna'}")
    print(f"public.equipo: {equipo} correos")


if __name__ == "__main__":
    main()
