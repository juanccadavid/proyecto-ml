"""Ejecuta los notebooks de sesiones/ y comprueba que las cifras escritas en su texto salgan de sus resultados.

Un notebook puede correr sin errores y aun así contar algo que ya no es cierto: si cambia el generador o el diseño, las
salidas cambian y las conclusiones escritas quedan viejas. Eso pasó con el notebook 03: tras cambiar t0 al corte, su
experimento de fuga corría pero daba el azar, y el texto seguía citando las cifras anteriores. Este script lo detecta.

Cada cifra decimal del texto (0,542 · 6,4 %) debe coincidir, redondeada a sus decimales, con algún número de las
salidas (también como porcentaje: 0,064 → 6,4). Es una comprobación necesaria, no suficiente: un número puede coincidir
por azar con otro de la misma salida.

Uso:
    uv run python scripts/verificar_notebooks.py                       # los notebooks de sesiones/
    uv run python scripts/verificar_notebooks.py --kernel venv-proyecto sesiones/03_*.ipynb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat

CIFRA = re.compile(r"(?<![\d.,])\d+,\d{1,3}(?![\d,])")  # 0,542 · 6,4 (coma decimal, como en el texto)
NUMERO = re.compile(r"-?\d+\.\d+")  # 0.542 · -0.003 (punto decimal, como en las salidas)
RAIZ = Path(__file__).resolve().parents[1]


def numeros_de_salidas(nb) -> set[float]:
    """Todos los números decimales de las salidas: texto, tablas (HTML) y figuras (JSON de Plotly)."""
    partes = []
    for celda in nb.cells:
        for salida in celda.get("outputs", []):
            if "text" in salida:
                partes.append(str(salida["text"]))
            for valor in salida.get("data", {}).values():
                partes.append(valor if isinstance(valor, str) else json.dumps(valor))
    return {float(x) for x in NUMERO.findall(" ".join(partes))}


def cifras_sin_respaldo(nb, numeros: set[float]) -> list[tuple[int, str]]:
    """(celda, cifra) de las cifras del texto que no coinciden con ningún número de las salidas."""
    faltan = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "markdown":
            continue
        for cifra in CIFRA.findall(celda.source.replace("−", "-")):
            decimales, valor = len(cifra.split(",")[1]), float(cifra.replace(",", "."))
            # En valor absoluto: el texto dice "resta 0,052" donde la tabla muestra -0.052.
            if not any(round(abs(n), decimales) == valor or round(abs(100 * n), decimales) == valor for n in numeros):
                faltan.append((i, cifra))
    return faltan


def main(argv: list[str] | None = None) -> int:
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Por defecto, sesiones/*.ipynb")
    parser.add_argument("--kernel", default="python3", help="Kernel de Jupyter (el del entorno del proyecto)")
    parser.add_argument("--timeout", type=int, default=1200, help="Segundos máximos por celda")
    args = parser.parse_args(argv)

    rutas = args.notebooks or sorted((RAIZ / "sesiones").glob("*.ipynb"))
    fallas = 0
    for ruta in rutas:
        nb = nbformat.read(ruta, as_version=4)
        en_la_raiz = {"metadata": {"path": str(RAIZ)}}  # la carpeta desde la que corre el kernel
        NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel, resources=en_la_raiz).execute()
        faltan = cifras_sin_respaldo(nb, numeros_de_salidas(nb))
        if faltan:
            fallas += 1
            detalle = ", ".join(f"celda {i}: {cifra}" for i, cifra in faltan)
            print(f"{ruta.name}: {len(faltan)} cifras del texto no salen de sus resultados ({detalle})")
        else:
            print(f"{ruta.name}: corre y sus cifras escritas salen de sus resultados")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
