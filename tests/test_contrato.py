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
    assert {d["id"] for d in contrato.distribucion} == set(k.DISTRIBUCION)


def test_esquema_pandera_con_los_tres_niveles(contrato):
    esquema = k.esquema(contrato)
    assert len(esquema.columns) == 74  # nivel 1
    nombres = {c.name for c in esquema.checks}
    assert {"volumen_minimo", "salario_en_pesos"} <= nombres  # nivel 2
    assert {"retiro_coherente", "nombreintegrado_coherente"} <= nombres  # nivel 3


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
        ("SEXO", "X", "Nivel 1 · SEXO: 1 fila(s) no cumplen isin"),
        ("NOMBRES", "  ", "Nivel 1 · NOMBRES: 1 fila(s) vacías en columna requerida"),
        ("TELEFONO1", "12ab", "Nivel 1 · TELEFONO1: 1 fila(s) no cumplen str_matches"),
        ("ESTRATO", 7, "Nivel 1 · ESTRATO: 1 fila(s) no cumplen less_than_or_equal_to(6)"),
        ("SALARIO", "mucho", "Nivel 1 · SALARIO: 1 fila(s) no son de tipo entero"),
        ("FECHANACIMIENTO", "1900-05-01", "Nivel 1 · FECHANACIMIENTO: 1 fila(s) no cumplen greater_than_or_equal_to"),
    ],
)
def test_nivel_1_por_fila(contrato, asociados, columna, valor, esperado):
    df = asociados.astype({columna: object})
    df.loc[0, columna] = valor
    errores = k.validar(df, contrato)
    assert any(e.startswith(esperado) for e in errores)
    # Un valor que no es del tipo esperado se reporta una vez, sin fallas en cadena de sus otros checks.
    assert len([e for e in errores if e.startswith(f"Nivel 1 · {columna}:")]) == 1


def test_nivel_1_duplicados_y_columnas(contrato, asociados):
    df = asociados.copy()
    df.loc[1, "CEDULA"] = df.loc[0, "CEDULA"]
    df["EXTRA"] = 1
    errores = k.validar(df.drop(columns="CARGO"), contrato)
    assert any(e.startswith("Nivel 1 · CEDULA: 2 fila(s) duplicadas") for e in errores)
    assert "Nivel 1 · Faltan columnas del contrato: ['CARGO']" in errores
    assert "Nivel 1 · Columnas fuera del contrato: ['EXTRA']" in errores


def _nivel_2(errores: list[str]) -> list[str]:
    return [e.split(":")[0] for e in errores if e.startswith("Nivel 2")]


def test_nivel_2_cambio_de_unidades(contrato, asociados):
    en_miles = asociados.assign(SALARIO=asociados["SALARIO"] // 1000)
    assert _nivel_2(k.validar(en_miles, contrato)) == ["Nivel 2 · salario_en_pesos"]


def test_nivel_2_ingesta_cortada(contrato, asociados):
    assert _nivel_2(k.validar(asociados.head(10), contrato)) == ["Nivel 2 · volumen_minimo"]


def test_nivel_2_atipicos_marginales(contrato, asociados):
    df = asociados.copy()
    df.loc[df.index[:30], "SALARIO"] *= 10  # 15 % de salarios multiplicados por error
    assert "Nivel 2 · salarios_atipicos_marginales" in _nivel_2(k.validar(df, contrato))


def test_nivel_2_tasa_de_retiro(contrato, asociados):
    solo_activos = asociados[asociados["#RETIROS"] == 0]  # p. ej. un filtro mal puesto en la extracción
    assert _nivel_2(k.validar(solo_activos, contrato)) == ["Nivel 2 · tasa_retiro_plausible"]


def test_nivel_3_entre_columnas(contrato, asociados):
    activo = asociados.index[asociados["#RETIROS"] == 0][0]
    df = asociados.copy()
    df.loc[activo, "#RETIROS"] = 1  # retirado sin fecha ni motivo de retiro
    df.loc[0, "NOMBREINTEGRADO"] = "OTRO NOMBRE"
    errores = k.validar(df, contrato)
    assert any(e.startswith("Nivel 3 · retiro_coherente: 1 fila(s)") for e in errores)
    assert any(e.startswith("Nivel 3 · nombreintegrado_coherente: 1 fila(s)") for e in errores)


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
