"""El contrato es la regla; este test es la prueba de que la regla está encendida.

- El CONTRATO (contratos/asociados.yaml + src/proyecto_ml/contrato.py) describe el dato válido y se
  ejecuta en la frontera del pipeline: ingesta (--contrato), subida a Supabase y CHECK de la BD.
- Este TEST verifica cómo se comporta la validación ante el dato inválido. Corre en el CI, nunca
  en producción.

Los límites están ESCRITOS A MANO a propósito, no se leen del YAML: si alguien relaja un rango del
contrato para desbloquear un pipeline (un viernes por la tarde), se rompe aquí un test con nombre
propio y la relajación queda a la vista en el PR. Si cambias una regla a conciencia, cambia también
su caso aquí, en el mismo PR.
"""

from datetime import timedelta

import pandas as pd
import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import DATA_DIR


@pytest.fixture(scope="module")
def contrato() -> k.Contrato:
    return k.cargar()


@pytest.fixture
def lote() -> pd.DataFrame:
    """Lote válido de partida: los 50 asociados ficticios versionados (24 % de retiros)."""
    return pd.read_parquet(DATA_DIR / "ejemplo" / "asociados_ejemplo.parquet")


def _errores(df: pd.DataFrame, contrato: k.Contrato, prefijo: str) -> list[str]:
    return [e for e in k.validar(df, contrato) if e.startswith(prefijo)]


def test_el_lote_de_partida_cumple(lote, contrato):
    assert k.validar(lote, contrato) == []


# ---------------------------------------------------------------- nivel 1: por fila

# (columna, valor, ¿cumple?). Los pares valor-límite (1 sí / 0 no) fijan cada borde del rango.
POR_FILA = [
    ("CEDULA", 1, True),
    ("CEDULA", 0, False),
    ("ESTRATO", 1, True),
    ("ESTRATO", 6, True),
    ("ESTRATO", 0, False),
    ("ESTRATO", 7, False),
    ("AGENCIA", 99, True),
    ("AGENCIA", 100, False),
    ("SEXO", "F", True),
    ("SEXO", "X", False),
    ("TIPOIDENTIFICACION", "NIT", False),
    ("PERIODODEDUCE", "SEMANAL", False),
    ("TIPOCONTRATO", "OBRA LABOR", False),
    ("#RETIROS", 2, False),
    ("SALARIO", 0, True),
    ("SALARIO", -1, False),
    ("SALARIO", "mucho", False),
    ("#SOLICITUD_CANCELACIONAHORROS", -1, False),
    ("TELEFONO1", "1234567", True),
    ("TELEFONO1", "123456", False),
    ("TELEFONO1", "12345678901", False),
    ("TELEFONO1", "12ab567", False),
    ("TELEFONO2", "", True),  # opcional: vacío es válido
    ("EMAIL", "sin-arroba.com", False),
    ("NUMEROCUENTA", "12345", False),
    ("CARGO", "", False),  # requerido: vacío no es válido
    ("FECHANACIMIENTO", "1920-01-01", True),
    ("FECHANACIMIENTO", "1919-12-31", False),
    ("FECHANACIMIENTO", "no-es-fecha", False),
]


@pytest.mark.parametrize(("columna", "valor", "cumple"), POR_FILA, ids=[f"{c}={v!r}" for c, v, _ in POR_FILA])
def test_nivel_1_por_fila(lote, contrato, columna, valor, cumple):
    lote = lote.astype({columna: object})
    lote.loc[0, columna] = valor
    errores = _errores(lote, contrato, f"Nivel 1 · {columna}:")
    assert (errores == []) == cumple, errores


@pytest.mark.parametrize("columna", ["CEDULA", "IDASOCIADOS"])
def test_nivel_1_identificadores_unicos(lote, contrato, columna):
    lote.loc[1, columna] = lote.loc[0, columna]
    assert _errores(lote, contrato, f"Nivel 1 · {columna}: 2 fila(s) duplicadas")


def test_nivel_1_columnas_exactas(lote, contrato):
    errores = k.validar(lote.drop(columns="CARGO").assign(EXTRA=1), contrato)
    assert "Nivel 1 · Faltan columnas del contrato: ['CARGO']" in errores
    assert "Nivel 1 · Columnas fuera del contrato: ['EXTRA']" in errores


# ---------------------------------------------------------------- nivel 2: distribución


def _activos_y_retirados(lote: pd.DataFrame, retirados: int) -> pd.DataFrame:
    return pd.concat([lote[lote["#RETIROS"] == 0], lote[lote["#RETIROS"] == 1].head(retirados)])


def _con_mediana_de_edad(lote: pd.DataFrame, anios: int) -> pd.DataFrame:
    return lote.assign(FECHANACIMIENTO=pd.Timestamp("2026-10-31") - pd.DateOffset(years=anios, days=1))


def _con_atipicos(lote: pd.DataFrame, n: int) -> pd.DataFrame:
    lote = lote.copy()
    lote.loc[lote.index[:n], "SALARIO"] = lote["SALARIO"].median() * 6
    return lote


DISTRIBUCION = [
    # (check, lote que cumple, lote que no cumple)
    ("volumen_minimo", lambda d: d.head(30), lambda d: d.head(29)),
    (
        "tasa_retiro_plausible",
        lambda d: _activos_y_retirados(d, 2),
        lambda d: _activos_y_retirados(d, 1),
    ),  # 5 % / 2,6 %
    ("salario_en_pesos", lambda d: d.assign(SALARIO=1_000_000), lambda d: d.assign(SALARIO=999_999)),
    ("salarios_atipicos_marginales", lambda d: _con_atipicos(d, 2), lambda d: _con_atipicos(d, 3)),  # 4 % / 6 %
    ("edad_mediana_plausible", lambda d: _con_mediana_de_edad(d, 22), lambda d: _con_mediana_de_edad(d, 21)),
]


@pytest.mark.parametrize(("check", "cumple", "no_cumple"), DISTRIBUCION, ids=[c for c, *_ in DISTRIBUCION])
def test_nivel_2_distribucion(lote, contrato, check, cumple, no_cumple):
    prefijo = f"Nivel 2 · {check}:"
    assert not _errores(cumple(lote), contrato, prefijo)
    assert _errores(no_cumple(lote), contrato, prefijo)


# ---------------------------------------------------------------- nivel 3: entre columnas


def _romper(lote: pd.DataFrame, filtro, columna: str, valor) -> pd.DataFrame:
    """Cambia `columna` en la primera fila que cumple `filtro` (valor puede depender de la fila)."""
    i = lote.index[filtro(lote)][0]
    nuevo = valor(lote.loc[i]) if callable(valor) else valor
    lote = lote.astype({columna: object})
    lote.loc[i, columna] = nuevo
    return lote


activo = lambda d: d["#RETIROS"] == 0  # noqa: E731
retirado = lambda d: d["#RETIROS"] == 1  # noqa: E731
fijo = lambda d: d["TIPOCONTRATO"] == "FIJO"  # noqa: E731
indefinido = lambda d: d["TIPOCONTRATO"] == "INDEFINIDO"  # noqa: E731
un_dia = timedelta(days=1)

ENTRE_COLUMNAS = [
    ("retiro_coherente", activo, "#RETIROS", 1),
    ("estado_laboral_coherente", retirado, "ESTADO_LABORAL", "VIGENTE"),
    ("contrato_fijo_con_fecha_fin", indefinido, "FECHATERMINACONTRATO", pd.Timestamp("2027-01-01")),
    ("fin_contrato_posterior_ingreso", fijo, "FECHATERMINACONTRATO", lambda f: f["FECHAINGRESOLABORA"] - un_dia),
    ("retiro_posterior_ingreso", retirado, "FECHARETIRO", lambda f: f["FECHAINGRESOLABORA"] - un_dia),
    ("nombreintegrado_coherente", activo, "NOMBREINTEGRADO", "OTRO NOMBRE"),
    (
        "mayor_de_edad_al_ingreso",
        activo,
        "FECHAINGRESOLABORA",
        lambda f: f["FECHANACIMIENTO"] + pd.DateOffset(years=17),
    ),
    ("pacto_integral_coherente", activo, "TIPOPACTO", "INTEGRAL"),
]


@pytest.mark.parametrize(("regla", "filtro", "columna", "valor"), ENTRE_COLUMNAS, ids=[r for r, *_ in ENTRE_COLUMNAS])
def test_nivel_3_entre_columnas(lote, contrato, regla, filtro, columna, valor):
    roto = _romper(lote, filtro, columna, valor)
    assert _errores(roto, contrato, f"Nivel 3 · {regla}: 1 fila(s)")


# ---------------------------------------------------------------- guardia


def test_toda_regla_del_contrato_tiene_su_test(contrato):
    """Una regla nueva en el YAML sin caso aquí sería una regla que nadie comprueba que funcione."""
    assert {d["id"] for d in contrato.distribucion} == {c for c, *_ in DISTRIBUCION}
    assert {r["id"] for r in contrato.reglas} == {r for r, *_ in ENTRE_COLUMNAS}
