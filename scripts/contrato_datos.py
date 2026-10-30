"""Herramientas del contrato de datos (contratos/asociados.yaml).

Uso:
    uv run python scripts/contrato_datos.py validar data/ejemplo/asociados_ejemplo.parquet
    uv run python scripts/contrato_datos.py sql       # -> supabase/asociados.sql
    uv run python scripts/contrato_datos.py web       # -> web/contrato.js
    uv run python scripts/contrato_datos.py subir data/ejemplo/asociados_ejemplo.parquet
    uv run python scripts/contrato_datos.py bajar     # -> data/raw/supabase_asociados.parquet

'subir' usa SUPABASE_URL y SUPABASE_KEY (clave pública anon/publishable) del .env, valida
contra el contrato antes de enviar y omite los asociados que ya estén en la tabla.

'bajar' usa SUPABASE_SECRET_KEY (clave SECRETA, solo en tu .env: la pública no puede leer por RLS),
valida el lote con el contrato y guarda la última copia + una copia fechada en data/raw/ (no va a Git).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from proyecto_ml import contrato as k
from proyecto_ml.config import DATA_DIR, ROOT_DIR
from proyecto_ml.data import load_dataset

SUPABASE_BAJADA = DATA_DIR / "raw" / "supabase_asociados.parquet"


def _validar(path: Path, contrato: k.Contrato) -> bool:
    errores = k.validar(load_dataset(path), contrato)
    for e in errores:
        print(f"  x {e}")
    print(f"{path}: {'cumple' if not errores else f'{len(errores)} incumplimiento(s)'} el contrato v{contrato.version}")
    return not errores


def _escribir(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8", newline="\n")
    print(f"-> {path.relative_to(ROOT_DIR)}")


def _subir(path: Path, contrato: k.Contrato, lote: int = 500) -> None:
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        sys.exit("Faltan SUPABASE_URL y/o SUPABASE_KEY en el .env (ver .env.example)")
    if not _validar(path, contrato):
        sys.exit("No se sube nada: corrige los incumplimientos primero.")

    registros = k.a_registros_db(load_dataset(path), contrato, origen="script")
    endpoint = f"{url}/rest/v1/{contrato.tabla_db}"
    headers = {"apikey": key, "Content-Type": "application/json", "Prefer": "return=minimal"}

    def insertar(filas: list[dict]) -> int:
        """Devuelve el código HTTP. No se usa upsert/on_conflict: exige permiso de lectura, y la clave
        pública solo puede insertar (RLS)."""
        req = urllib.request.Request(endpoint, data=json.dumps(filas).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if e.code == 409:  # cédula o IDASOCIADOS ya existen
                return 409
            sys.exit(f"Supabase respondió {e.code}: {e.read().decode('utf-8', 'replace')}")

    nuevos = repetidos = 0
    for i in range(0, len(registros), lote):
        bloque = registros[i : i + lote]
        if insertar(bloque) != 409:
            nuevos += len(bloque)
            continue
        # El lote completo se rechaza si una fila choca: se reintenta fila por fila.
        for fila in bloque:
            ok = insertar([fila]) != 409
            nuevos, repetidos = nuevos + ok, repetidos + (not ok)
    print(f"{nuevos} registros nuevos en {contrato.tabla_db}; {repetidos} omitidos por cédula/IDASOCIADOS repetidos")


def leer_tabla(url: str, key: str, tabla: str, pagina: int = 1000) -> list[dict]:
    """Lee la tabla completa por la API REST, de a `pagina` filas (Supabase entrega máximo 1000 por consulta)."""
    filas: list[dict] = []
    while True:
        endpoint = f"{url}/rest/v1/{tabla}?select=*&order=id&limit={pagina}&offset={len(filas)}"
        req = urllib.request.Request(endpoint, headers={"apikey": key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                bloque = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            sys.exit(f"Supabase respondió {e.code}: {e.read().decode('utf-8', 'replace')}")
        filas += bloque
        if len(bloque) < pagina:
            return filas


def _bajar(salida: Path, contrato: k.Contrato, forzar: bool = False) -> None:
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        sys.exit("Faltan SUPABASE_URL y/o SUPABASE_SECRET_KEY en el .env (ver .env.example)")
    if key.startswith("sb_publishable_"):
        sys.exit(
            "SUPABASE_SECRET_KEY es la clave pública: con RLS no puede leer (vería la tabla vacía). Usa la secreta."
        )

    registros = leer_tabla(url, key, contrato.tabla_db)
    if not registros:
        sys.exit(f"La tabla {contrato.tabla_db} está vacía: no se guarda nada.")
    origenes = Counter(r.get("origen") for r in registros)
    df = k.desde_registros_db(registros, contrato)
    print(f"{len(df)} asociados descargados de Supabase ({', '.join(f'{o}: {n}' for o, n in origenes.items())})")

    # Frontera del pipeline: lo que entra a data/ debe cumplir el contrato (incluido el nivel 2, que la BD no ve).
    errores = k.validar(df, contrato)
    for e in errores:
        print(f"  x {e}")
    if errores and not forzar:
        sys.exit("No se guarda: el lote no cumple el contrato. Revísalo (o usa --forzar bajo tu responsabilidad).")

    salida.parent.mkdir(parents=True, exist_ok=True)
    fechada = salida.with_name(f"{salida.stem}_{datetime.now():%Y%m%d_%H%M%S}{salida.suffix}")
    df.to_parquet(fechada, index=False)  # copia fija: permite repetir un entrenamiento con los mismos datos
    df.to_parquet(salida, index=False)  # la última, la que apunta DATOS_ASOCIADOS en el .env
    ruta = salida.resolve().relative_to(ROOT_DIR) if salida.resolve().is_relative_to(ROOT_DIR) else salida
    print(f"-> {ruta} (última) y {fechada.name} (copia fechada)")
    print(f"Para entrenar con ella: en el .env DATOS_ASOCIADOS={ruta.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("accion", choices=["validar", "sql", "web", "subir", "bajar"])
    parser.add_argument("archivo", type=Path, nargs="?", help="Dataset (para validar y subir)")
    parser.add_argument("--contrato", type=Path, default=k.CONTRATO_ASOCIADOS)
    parser.add_argument("--salida", type=Path, default=SUPABASE_BAJADA, help="Destino de 'bajar'")
    parser.add_argument("--forzar", action="store_true", help="'bajar' guarda aunque el lote no cumpla el contrato")
    args = parser.parse_args()
    if args.accion == "bajar":
        _bajar(args.salida, k.cargar(args.contrato), args.forzar)
        return
    contrato = k.cargar(args.contrato)

    if args.accion in ("validar", "subir") and args.archivo is None:
        parser.error(f"'{args.accion}' necesita la ruta del dataset")
    if args.accion == "validar":
        sys.exit(0 if _validar(args.archivo, contrato) else 1)
    if args.accion == "sql":
        _escribir(ROOT_DIR / "supabase" / f"{contrato.tabla_db}.sql", k.sql(contrato))
    elif args.accion == "web":
        _escribir(ROOT_DIR / "web" / "contrato.js", k.js(contrato))
    else:
        _subir(args.archivo, contrato)


if __name__ == "__main__":
    main()
