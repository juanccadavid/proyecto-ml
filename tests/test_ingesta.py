"""Ingesta con Polars: mismo resultado que la versión anterior en pandas."""

import importlib.util
import sys

import pandas as pd
import pytest

from proyecto_ml.config import ROOT_DIR
from proyecto_ml.generador import FALTANTES, GeneradorAsociados

# scripts/ no es un paquete: el CLI se carga desde su ruta para probar sus funciones.
_spec = importlib.util.spec_from_file_location("ingesta", ROOT_DIR / "scripts" / "ingesta.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

TARGET = "#RETIROS"


@pytest.fixture
def crudo(tmp_path) -> tuple[pd.DataFrame, object]:
    """Asociados con faltantes y 3 filas repetidas, guardados como parquet."""
    df = GeneradorAsociados(n=200, seed=5, faltantes=FALTANTES).generar()
    df = pd.concat([df, df.head(3)], ignore_index=True)
    path = tmp_path / "crudo.parquet"
    df.to_parquet(path, index=False)
    return df, path


def test_perfil_igual_que_pandas(crudo):
    df, path = crudo
    info = cli.profile(cli.leer(path), TARGET)
    assert info["filas"] == len(df)
    assert info["columnas"] == df.shape[1]
    assert info["duplicados"] == df.duplicated().sum() == 3
    esperado = {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0}
    assert info["nulos_pct"] == esperado
    assert info["distribucion_target"] == {str(k): int(v) for k, v in df[TARGET].value_counts().items()}


def test_salida_se_lee_con_pandas(crudo, tmp_path, monkeypatch):
    df, path = crudo
    salida = tmp_path / "dataset.parquet"
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(sys, "argv", ["ingesta", "--input", str(path), "--target", TARGET, "--output", str(salida)])
    cli.main()
    procesado = pd.read_parquet(salida)  # el resto del proyecto sigue leyendo con pandas
    esperado = df.drop_duplicates().reset_index(drop=True)
    pd.testing.assert_frame_equal(procesado, esperado, check_dtype=False)
    assert (tmp_path / "reports" / "perfil_datos.json").exists()
