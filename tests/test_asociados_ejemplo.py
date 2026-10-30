"""Pruebas del generador de asociados ficticios (proyecto_ml.generador)."""

import pandas as pd
import pytest

from proyecto_ml import generador as gen
from proyecto_ml.config import DATA_DIR
from proyecto_ml.data import load_dataset, split_features_target

TARGET = "#RETIROS"


@pytest.fixture(scope="module")
def asociados() -> pd.DataFrame:
    return gen.GeneradorAsociados(n=50, seed=42).generar()


def test_esquema_completo_y_en_orden(asociados):
    esperadas = [c for cols in gen.GRUPOS.values() for c in cols]
    assert list(asociados.columns) == esperadas
    assert asociados.shape == (50, 74)
    assert asociados.columns.is_unique


def test_diccionario_cubre_todas_las_columnas(asociados):
    dic = gen.diccionario()
    assert list(dic["COLUMNA"]) == list(asociados.columns)
    assert dic["GRUPO"].value_counts().to_dict() == {"SOCIODEMOGRAFICO": 32, "LABORAL": 31, "PRODUCTOS": 8, "RETIRO": 3}


def test_reproducible_con_la_misma_semilla(asociados):
    pd.testing.assert_frame_equal(asociados, gen.GeneradorAsociados(n=50, seed=42).generar())
    assert not asociados.equals(gen.GeneradorAsociados(n=50, seed=7).generar())


def test_guardar_escribe_parquet_y_excel(tmp_path):
    parquet, xlsx = gen.GeneradorAsociados(n=10, seed=3).guardar(tmp_path, "prueba")
    assert len(pd.read_parquet(parquet)) == 10
    assert pd.ExcelFile(xlsx).sheet_names == ["DATOS", "DICCIONARIO"]


def test_objetivo_binario_con_ambas_clases(asociados):
    assert set(asociados[TARGET].unique()) == {0, 1}


def test_retirados_coherentes(asociados):
    ret = asociados[asociados[TARGET] == 1]
    act = asociados[asociados[TARGET] == 0]
    assert ret["FECHARETIRO"].notna().all() and act["FECHARETIRO"].isna().all()
    assert (ret["ESTADO"] == "RETIRADO").all() and (act["ESTADO"] == "ACTIVO").all()
    assert (ret["MOTIVORETIRO"] != "").all() and (act["MOTIVORETIRO"] == "").all()
    assert (ret["ESTADO_LABORAL"] == "RETIRADO").all()


def test_contrato_fijo_que_termina_con_el_retiro(asociados):
    fijo_ret = asociados[(asociados["TIPOCONTRATO"] == "FIJO") & (asociados[TARGET] == 1)]
    assert (fijo_ret["FECHATERMINACONTRATO"] == fijo_ret["FECHARETIRO"]).all()
    assert (fijo_ret["MOTIVORETIRO"] == "TERMINACION CONTRATO LABORAL").all()


def test_datos_evidentemente_ficticios(asociados):
    assert asociados["CEDULA"].between(900_000_001, 900_000_050).all()
    assert asociados["EMAIL"].str.endswith("@example.com").all()
    assert asociados["PRIMERAPELLIDO"].isin(gen.APELLIDOS).all()
    assert asociados["SEGUNDOAPELLIDO"].isin(gen.APELLIDOS).all()


def test_archivo_versionado_coincide_con_el_generador(asociados):
    """Si alguien edita el generador sin regenerar data/ejemplo (o al revés), esta prueba lo detecta."""
    path = DATA_DIR / "ejemplo" / "asociados_ejemplo.parquet"
    if not path.exists():
        pytest.skip("data/ejemplo/asociados_ejemplo.parquet no existe")
    pd.testing.assert_frame_equal(load_dataset(path), asociados, check_dtype=False)


def test_compatible_con_ingesta(tmp_path, asociados):
    path = tmp_path / "asociados.parquet"
    asociados.to_parquet(path, index=False)
    X, y = split_features_target(load_dataset(path), TARGET)
    assert TARGET not in X.columns
    assert len(X) == len(y) == 50
