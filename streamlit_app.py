"""Punto de entrada para Streamlit Community Cloud: abre el simulador de abandono.

Cloud ejecuta ``streamlit run streamlit_app.py`` desde la raíz del repositorio e instala las dependencias
del uv.lock. Por si no instala el paquete (está en src/), se agrega src/ a la ruta de importación.
En local se usa ``make simulador``.
"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "proyecto_ml" / "app_abandono.py"), run_name="__main__")
