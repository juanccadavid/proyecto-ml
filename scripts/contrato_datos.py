"""Herramientas del contrato de datos (contratos/asociados.yaml).

Uso:
    uv run python scripts/contrato_datos.py validar data/ejemplo/asociados_ejemplo.parquet
    uv run python scripts/contrato_datos.py sql       # -> supabase/asociados.sql
    uv run python scripts/contrato_datos.py web       # -> web/contrato.js
    uv run python scripts/contrato_datos.py subir data/ejemplo/asociados_ejemplo.parquet

'subir' usa SUPABASE_URL y SUPABASE_KEY (clave pública anon/publishable) del .env, valida
contra el contrato antes de enviar y omite las cédulas que ya estén en la tabla.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from proyecto_ml import contrato as k
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.data import load_dataset


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
    endpoint = f"{url}/rest/v1/{contrato.tabla_db}?on_conflict=cedula"
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal,resolution=ignore-duplicates",
    }
    for i in range(0, len(registros), lote):
        cuerpo = json.dumps(registros[i : i + lote]).encode("utf-8")
        req = urllib.request.Request(endpoint, data=cuerpo, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except urllib.error.HTTPError as e:
            sys.exit(f"Supabase respondió {e.code}: {e.read().decode('utf-8', 'replace')}")
    print(f"{len(registros)} registros enviados a {contrato.tabla_db} (las cédulas repetidas se omiten)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("accion", choices=["validar", "sql", "web", "subir"])
    parser.add_argument("archivo", type=Path, nargs="?", help="Dataset (para validar y subir)")
    parser.add_argument("--contrato", type=Path, default=k.CONTRATO_ASOCIADOS)
    args = parser.parse_args()
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
