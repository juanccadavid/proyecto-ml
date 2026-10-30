"""Pruebas de la HERRAMIENTA del contrato: carga del YAML, esquema pandera, SQL, JS y registros para la BD.

Las pruebas de las REGLAS (qué dato es válido y cuál no) están en tests/data/test_contrato_datos.py.
"""

import pandas as pd
import pytest

from proyecto_ml import contrato as k
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def contrato() -> k.Contrato:
    return k.cargar()


@pytest.fixture(scope="module")
def asociados() -> pd.DataFrame:
    return GeneradorAsociados(n=200, seed=1).generar()


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


def test_un_tipo_invalido_se_reporta_una_sola_vez(contrato, asociados):
    """Formato de los mensajes: 'mucho' en SALARIO no debe arrastrar fallas en cadena de ge/le."""
    df = asociados.astype({"SALARIO": object})
    df.loc[0, "SALARIO"] = "mucho"
    errores = [e for e in k.validar(df, contrato) if e.startswith("Nivel 1 · SALARIO:")]
    assert errores == ["Nivel 1 · SALARIO: 1 fila(s) no son de tipo entero (ej. fila 0: 'mucho')"]


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
