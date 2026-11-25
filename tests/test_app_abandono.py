"""El simulador expone solo variables vigentes."""

import ast
import runpy
from collections import Counter
from pathlib import Path

from proyecto_ml import abandono as a


def _secciones_formulario() -> dict[str, list[str]]:
    """Lee la configuración sin ejecutar la aplicación de Streamlit."""
    texto = Path("src/proyecto_ml/app_abandono.py").read_text(encoding="utf-8")
    modulo = ast.parse(texto)
    asignacion = next(
        nodo
        for nodo in modulo.body
        if isinstance(nodo, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "SECCIONES_FORMULARIO" for x in nodo.targets)
    )
    return ast.literal_eval(asignacion.value)


def test_simulador_usa_antiguedades_definitivas():
    texto = Path("src/proyecto_ml/app_abandono.py").read_text(encoding="utf-8")
    assert "ANTIGUEDAD_F_MESES" in texto
    assert "ANTIGUEDAD_LA_MESES" in texto
    assert '"ANTIGUEDAD"' not in texto


def test_simulador_no_referencia_variables_eliminadas():
    texto = Path("src/proyecto_ml/app_abandono.py").read_text(encoding="utf-8")
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
