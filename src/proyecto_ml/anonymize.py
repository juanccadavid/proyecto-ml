"""Anonimización ANTES de que los datos salgan de la organización origen (la usa scripts/anonimizar.py).

- Datos personales: el contrato de origen marca cada columna con ``pii: eliminar`` o ``pii: seudonimizar``
  (contratos/asociados.yaml). Las primeras se eliminan; las segundas salen como ``<COLUMNA>_SEUDO``.
- Seudónimo: HMAC-SHA256 con una clave secreta (``ANON_SALT``), los 64 caracteres hexadecimales completos (texto,
  nunca un número: un entero recortaría el hash y podría chocar). El mismo identificador da siempre el mismo
  seudónimo, así que los registros se pueden unir entre extractos. Sin la clave no se recalcula por fuerza
  bruta sobre cédulas.
- Cuasi-identificadores (edad, salario) se pueden agrupar en rangos (``bins``).

**Seudonimizar no es anonimizar de forma irreversible:** quien tenga la clave puede recalcular el seudónimo de
una cédula conocida y reconocer su fila. La clave la guarda solo quien custodia los datos; nunca va al
repositorio, a MLflow ni a los logs. Una tabla de correspondencia (seudónimo -> cédula), si se necesita para
contactar asociados, vive fuera del pipeline de ML, con el custodio.

``anonimizar_extracto`` no devuelve nada si algo queda mal (``AnonimizacionInsegura``): columnas sin decisión,
columnas personales que siguen ahí, seudónimos inválidos o valores personales copiados en otra columna. Sus
mensajes dicen columnas y cantidades, nunca valores.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from collections.abc import Iterable

import pandas as pd

from proyecto_ml import contrato as k

LARGO_MIN_SAL = 16
LARGO_MIN_VALOR = 6  # los valores personales más cortos (p. ej. una extensión) chocarían con códigos legítimos


class AnonimizacionInsegura(ValueError):
    """La salida conservaría datos personales o no se pudo seudonimizar: no se escribe."""


def _texto_id(value: object) -> str:
    """El identificador como texto estable: 1037000000, 1037000000.0 y ' 1037000000 ' dan lo mismo."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        raise ValueError("identificador vacío")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    texto = str(value).strip()
    if not texto:
        raise ValueError("identificador vacío")
    return texto


def _clave(salt: str | None) -> str:
    salt = salt or os.getenv("ANON_SALT", "")
    if len(salt) < LARGO_MIN_SAL:
        raise AnonimizacionInsegura(f"ANON_SALT debe tener al menos {LARGO_MIN_SAL} caracteres (make salt)")
    return salt


def hash_value(value: object, salt: str) -> str:
    """Seudónimo de un identificador: HMAC-SHA256 completo, 64 caracteres hexadecimales."""
    if not salt or len(salt) < LARGO_MIN_SAL:
        raise ValueError(f"ANON_SALT debe tener al menos {LARGO_MIN_SAL} caracteres")
    return hmac.new(salt.encode(), _texto_id(value).encode(), hashlib.sha256).hexdigest()


def seudonimizar(serie: pd.Series, salt: str) -> pd.Series:
    """Seudónimo de cada fila. Un identificador vacío no se puede seudonimizar: error (sin mostrar valores)."""
    vacios = serie.isna() | serie.astype("string").str.strip().eq("")
    if vacios.any():
        raise AnonimizacionInsegura(f"{serie.name}: {int(vacios.sum())} identificador(es) vacío(s)")
    return serie.map(lambda v: hash_value(v, salt)).astype("string")


def anonymize(
    df: pd.DataFrame,
    *,
    id_columns: Iterable[str] = (),
    drop_columns: Iterable[str] = (),
    bins: dict[str, list[float]] | None = None,
    salt: str | None = None,
) -> pd.DataFrame:
    """Versión genérica (sin contrato): seudonimiza ``id_columns`` en su lugar y elimina ``drop_columns``.

    Para el extracto de asociados usa ``anonimizar_extracto``, que además verifica la salida.
    """
    salt = _clave(salt)
    out = df.copy()
    for col in id_columns:
        out[col] = seudonimizar(out[col], salt)
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    for col, edges in (bins or {}).items():
        out[col] = pd.cut(out[col], bins=edges, include_lowest=True).astype(str)
    return out


def _valores(serie: pd.Series) -> set[str]:
    textos = serie.dropna().map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip())
    return {t for t in textos if len(t) >= LARGO_MIN_VALOR}


def verificar(
    original: pd.DataFrame, salida: pd.DataFrame, eliminadas: set[str], seudonimizadas: set[str], salt: str
) -> list[str]:
    """Problemas de la salida (lista vacía si está bien). Solo nombres de columnas y cantidades, nunca valores."""
    problemas = []
    if len(salida) != len(original):
        problemas.append(f"la salida tiene {len(salida)} filas y el extracto {len(original)}")
    if quedan := sorted((eliminadas | seudonimizadas) & set(salida.columns)):
        problemas.append(f"columnas personales que siguen en la salida: {quedan}")
    patron = re.compile(k.PATRON_SEUDONIMO)
    for c in sorted(seudonimizadas & set(original.columns)):
        destino = c + k.SUFIJO_SEUDONIMO
        if destino not in salida:
            problemas.append(f"falta {destino} (seudónimo de {c})")
        elif not salida[destino].astype(str).map(lambda v: bool(patron.fullmatch(v))).all():
            problemas.append(f"{destino}: hay valores que no son un seudónimo de 64 caracteres hexadecimales")
    personales = set().union(*(_valores(original[c]) for c in (eliminadas | seudonimizadas) & set(original.columns)))
    for col in salida.columns:  # un dato personal copiado en otra columna (p. ej. la cédula dentro de un texto)
        textos = salida[col].dropna().map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
        if (n := int(textos.str.strip().isin(personales).sum())) > 0:
            problemas.append(f"{col}: {n} fila(s) con valores de columnas personales del extracto")
        if (n := int(textos.str.contains(salt, regex=False).sum())) > 0:
            problemas.append(f"{col}: {n} fila(s) contienen la clave ANON_SALT")
    return problemas


def anonimizar_extracto(
    df: pd.DataFrame,
    contrato: k.Contrato | None = None,
    *,
    salt: str | None = None,
    conservar: Iterable[str] = (),
    eliminar: Iterable[str] = (),
    seudonimizar_columnas: Iterable[str] = (),
    bins: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """El extracto listo para compartir, según el contrato de origen, o ``AnonimizacionInsegura``.

    Toda columna del extracto necesita una decisión: estar en el contrato (y entonces su ``pii`` dice qué hacer),
    o venir en ``conservar``, ``eliminar`` o ``seudonimizar_columnas``. Una columna desconocida no pasa en silencio:
    podría ser un dato personal que el contrato no conoce.
    """
    contrato = contrato or k.cargar()
    salt = _clave(salt)
    conservar, eliminar, a_seudo = set(conservar), set(eliminar), set(seudonimizar_columnas)
    a_eliminar = set(k.columnas_pii(contrato, "eliminar")) | eliminar
    a_seudo |= set(k.columnas_pii(contrato, "seudonimizar"))
    if choque := sorted(conservar & (a_eliminar | a_seudo)):
        raise AnonimizacionInsegura(f"Columnas personales marcadas para conservar: {choque}")
    decididas = set(contrato.nombres) | conservar | a_eliminar | a_seudo
    if sin_decision := [c for c in df.columns if c not in decididas]:
        raise AnonimizacionInsegura(
            f"Columnas que el contrato no conoce y no tienen decisión: {sin_decision}. Indica para cada una "
            "--conservar, --eliminar o --seudonimizar."
        )
    salida = df.drop(columns=[c for c in df.columns if c in a_eliminar])
    for c in [c for c in df.columns if c in a_seudo]:  # el seudónimo queda en la posición de la columna original
        posicion = salida.columns.get_loc(c)
        seudonimos = seudonimizar(df[c], salt)
        salida = salida.drop(columns=c)
        salida.insert(posicion, c + k.SUFIJO_SEUDONIMO, seudonimos)
    for col, edges in (bins or {}).items():
        salida[col] = pd.cut(salida[col], bins=edges, include_lowest=True).astype(str)
    if problemas := verificar(df, salida, a_eliminar & set(df.columns), a_seudo & set(df.columns), salt):
        raise AnonimizacionInsegura("La salida no es segura:\n  - " + "\n  - ".join(problemas))
    return salida
