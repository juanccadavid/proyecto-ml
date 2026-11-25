"""Los notebooks de sesiones/ corren de principio a fin con el código actual.

El CI ya revisa que se suban sin salidas (nbstripout), pero eso no detecta que un cambio de esquema los rompa: así
pasó con el contrato v2.2 (`ANTIGUEDAD` pasó a `ANTIGUEDAD_F_MESES`). Usan solo datos sintéticos de desarrollo.
El 01 es una plantilla que necesita un dataset propio, por eso queda fuera.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["02_juan_eda_foto_asociado.ipynb", "03_sebastian_feature_engineering.ipynb"]


@pytest.mark.notebook
@pytest.mark.parametrize("nombre", NOTEBOOKS)
def test_notebook_corre_completo(nombre):
    nb = nbformat.read(RAIZ / "sesiones" / nombre, as_version=4)
    # Desde la raíz del repo, como se abren en VS Code con el proyecto; 30 min por celda como máximo.
    nbclient.NotebookClient(
        nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}
    ).execute()
    errores = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errores
