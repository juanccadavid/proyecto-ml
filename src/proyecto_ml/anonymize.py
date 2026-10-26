"""Utilidades de anonimización ANTES de que los datos salgan de la organización origen.

- Identificadores -> HMAC-SHA256 con una sal secreta (ANON_SALT). Sin la sal no se revierte
  ni se puede recalcular por fuerza bruta sobre cédulas.
- Columnas de PII directa se eliminan.
- Variables cuasi-identificadoras (edad, salario) se agrupan en rangos.
"""

from __future__ import annotations

import hashlib
import hmac
import os

import pandas as pd


def hash_value(value: object, salt: str) -> str:
    if not salt or len(salt) < 16:
        raise ValueError("ANON_SALT debe tener al menos 16 caracteres")
    digest = hmac.new(salt.encode(), str(value).strip().encode(), hashlib.sha256)
    return digest.hexdigest()[:16]


def anonymize(
    df: pd.DataFrame,
    *,
    id_columns: list[str] = (),
    drop_columns: list[str] = (),
    bins: dict[str, list[float]] | None = None,
    salt: str | None = None,
) -> pd.DataFrame:
    salt = salt or os.getenv("ANON_SALT", "")
    out = df.copy()
    for col in id_columns:
        out[col] = out[col].map(lambda v: hash_value(v, salt))
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    for col, edges in (bins or {}).items():
        out[col] = pd.cut(out[col], bins=edges, include_lowest=True).astype(str)
    return out
