"""El simulador expone solo variables vigentes."""

from pathlib import Path


def test_simulador_usa_antiguedades_definitivas():
    texto = Path("src/proyecto_ml/app_abandono.py").read_text(encoding="utf-8")
    assert "ANTIGUEDAD_F_MESES" in texto
    assert "ANTIGUEDAD_LA_MESES" in texto
    assert '"ANTIGUEDAD"' not in texto


def test_simulador_no_referencia_variables_eliminadas():
    texto = Path("src/proyecto_ml/app_abandono.py").read_text(encoding="utf-8")
    for antigua in ("#INTENCIONRETIROS_ULT3MESES", "TIPOPACTO", "PERIODODEDUCE", "TIPOCUENTA"):
        assert antigua not in texto
