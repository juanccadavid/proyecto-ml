"""Pruebas del contrato de datos (contratos/asociados.yaml) y de lo que se genera a partir de él."""

import importlib.util

import pandas as pd
import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import DATA_DIR, ROOT_DIR


@pytest.fixture(scope="module")
def contrato() -> k.Contrato:
    return k.cargar()


@pytest.fixture(scope="module")
def asociados() -> pd.DataFrame:
    # scripts/ no es un paquete: se carga el generador desde su ruta.
    spec = importlib.util.spec_from_file_location("gen", ROOT_DIR / "scripts" / "generar_asociados_ejemplo.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen.make_asociados(n=200, seed=1)


def test_contrato_cubre_el_esquema_del_generador(contrato, asociados):
    assert contrato.nombres == list(asociados.columns)
    assert len(contrato.columnas) == 74
    assert contrato.objetivo in contrato.nombres


def test_cada_regla_tiene_implementacion_en_python(contrato):
    assert {r["id"] for r in contrato.reglas} == set(k.REGLAS)


def test_nombres_en_la_bd(contrato):
    db = {c.nombre: c.db for c in contrato.columnas}
    assert db["#RETIROS"] == "num_retiros"
    assert db["CANCELACIONCREDITOS_ULT3MESES"] == "cancelacioncreditos_ult3meses"
    assert len(set(db.values())) == len(db)


def test_datos_generados_cumplen(contrato, asociados):
    assert k.validar(asociados, contrato) == []


def test_archivo_de_ejemplo_cumple(contrato):
    path = DATA_DIR / "ejemplo" / "asociados_ejemplo.parquet"
    if not path.exists():
        pytest.skip("data/ejemplo/asociados_ejemplo.parquet no existe")
    assert k.validar(pd.read_parquet(path), contrato) == []


@pytest.mark.parametrize(
    ("columna", "valor", "esperado"),
    [
        ("SEXO", "X", "SEXO: 1 fila(s) fuera de"),
        ("NOMBRES", "  ", "NOMBRES: 1 fila(s) vacías en columna requerida"),
        ("TELEFONO1", "12ab", "TELEFONO1: 1 fila(s) no cumplen el patrón"),
        ("ESTRATO", 7, "ESTRATO: 1 fila(s) mayores que 6"),
        ("SALARIO", "mucho", "SALARIO: 1 fila(s) no son de tipo entero"),
        ("FECHANACIMIENTO", "1900-05-01", "FECHANACIMIENTO: 1 fila(s) menores que 1920-01-01"),
    ],
)
def test_detecta_incumplimientos_por_columna(contrato, asociados, columna, valor, esperado):
    df = asociados.astype({columna: object})
    df.loc[0, columna] = valor
    assert any(e.startswith(esperado) for e in k.validar(df, contrato))


def test_detecta_duplicados_y_columnas(contrato, asociados):
    df = asociados.copy()
    df.loc[1, "CEDULA"] = df.loc[0, "CEDULA"]
    df["EXTRA"] = 1
    errores = k.validar(df.drop(columns="CARGO"), contrato)
    assert any(e.startswith("CEDULA: 2 fila(s) duplicadas") for e in errores)
    assert "Faltan columnas del contrato: ['CARGO']" in errores
    assert "Columnas fuera del contrato: ['EXTRA']" in errores


def test_detecta_reglas_entre_columnas(contrato, asociados):
    activo = asociados.index[asociados["#RETIROS"] == 0][0]
    df = asociados.copy()
    df.loc[activo, "#RETIROS"] = 1  # retirado sin fecha ni motivo de retiro
    assert any(e.startswith("Regla retiro_coherente: 1 fila(s)") for e in k.validar(df, contrato))


def test_registros_para_la_bd(contrato, asociados):
    reg = k.a_registros_db(asociados.head(20), contrato)
    activo = next(r for r in reg if r["num_retiros"] == 0)
    assert activo["origen"] == "script"
    assert activo["fecharetiro"] is None and activo["motivoretiro"] is None
    assert isinstance(activo["cedula"], int)
    assert len(activo["fechanacimiento"]) == 10  # AAAA-MM-DD


def test_sql_incluye_restricciones_y_seguridad(contrato):
    sql = k.sql(contrato)
    assert "cedula bigint not null unique" in sql
    assert "constraint retiro_coherente check" in sql
    assert "enable row level security" in sql
    assert "grant select" not in sql  # la clave pública no puede leer


@pytest.mark.parametrize(
    ("archivo", "generar"),
    [("supabase/asociados.sql", k.sql), ("web/contrato.js", k.js)],
)
def test_archivos_generados_al_dia(archivo, generar):
    """Si cambias el contrato, regenera con make contrato-sql / make contrato-web."""
    assert (ROOT_DIR / archivo).read_text(encoding="utf-8") == generar()
