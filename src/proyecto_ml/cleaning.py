"""Limpieza y transformación general de datos.

Hay dos tipos de herramientas y la diferencia importa para evitar fuga de datos:

1. **Funciones sin estado** (``normalize_column_names``, ``standardize_missing``, ``coerce_numeric``...):
   aplican reglas fijas fila a fila y no aprenden nada del dataset. Se pueden usar sobre el dataset
   completo ANTES de partir en train/test. ``clean_dataset`` las encadena todas.
2. **Transformadores con estado** (``ColumnDropper``, ``OutlierClipper``): aprenden umbrales de los
   datos (porcentaje de nulos, cuantiles). Van DENTRO del ``Pipeline`` para que se ajusten solo con train.

Uso típico:

    df = clean_dataset(load_dataset(ruta), date_columns=["periodo"])
    X, y = split_features_target(df, "target")
    X_train, X_test, y_train, y_test = split_data(X, y)
    pre = build_preprocessor(numeric, categorical, clip_outliers="iqr")
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Textos que en la práctica significan "dato faltante" (se comparan en minúsculas y sin espacios).
DEFAULT_NA_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "sin dato", "sin información", "no aplica", "nd"}
)


# ---------------------------------------------------------------------------
# 1. Funciones sin estado (seguras antes del split)
# ---------------------------------------------------------------------------


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de texto (object o string)."""
    return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]


def _slugify(name: str) -> str:
    """'Fecha de Afiliación (dd/mm)' -> 'fecha_de_afiliacion_dd_mm'."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name or "columna"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nombres en snake_case, sin tildes ni símbolos. Los repetidos reciben sufijo _2, _3..."""
    seen: dict[str, int] = {}
    new_names = []
    for col in df.columns:
        base = _slugify(col)
        seen[base] = seen.get(base, 0) + 1
        new_names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return df.set_axis(new_names, axis=1)


def strip_strings(df: pd.DataFrame, *, lowercase: bool = False) -> pd.DataFrame:
    """Quita espacios al inicio/fin y colapsa espacios internos en las columnas de texto.

    ``lowercase=True`` unifica mayúsculas ('Medellín' y 'MEDELLÍN' pasan a ser la misma categoría).
    """
    df = df.copy()
    for col in _text_columns(df):
        s = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = s.str.lower() if lowercase else s
    return df


def standardize_missing(df: pd.DataFrame, tokens: Iterable[str] = DEFAULT_NA_TOKENS) -> pd.DataFrame:
    """Convierte textos como 'N/A', '-', '?' o '' en nulos reales (``pd.NA``)."""
    tokens = {t.strip().lower() for t in tokens}
    df = df.copy()
    for col in _text_columns(df):
        s = df[col].astype("string")
        df[col] = s.mask(s.str.strip().str.lower().isin(tokens))
    return df


def coerce_numeric(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    decimal: str = ".",
    thousands: str | None = ",",
    min_valid_ratio: float = 0.95,
) -> pd.DataFrame:
    """Convierte a número columnas de texto como '$1,234.50', '12 %' o ' 3.5 '.

    - ``columns=None``: revisa todas las columnas de texto y convierte solo las que logran
      interpretarse en al menos ``min_valid_ratio`` de sus valores no nulos (así no se
      destruyen columnas categóricas). Si se pasan ``columns``, se convierten siempre.
    - Formato colombiano/europeo ('1.234,56'): ``decimal=","``, ``thousands="."``.
    - Lo que no se pueda convertir queda como NaN.
    """
    df = df.copy()
    explicit = columns is not None
    for col in columns if explicit else _text_columns(df):
        raw = df[col].astype("string")
        s = raw.str.replace(r"[\s$%€]", "", regex=True)
        if thousands:
            s = s.str.replace(thousands, "", regex=False)
        if decimal != ".":
            s = s.str.replace(decimal, ".", regex=False)
        converted = pd.to_numeric(s, errors="coerce")

        non_null = raw.notna().sum()
        ratio = converted.notna().sum() / non_null if non_null else 0.0
        if explicit or ratio >= min_valid_ratio:
            df[col] = converted
    return df


def coerce_datetime(
    df: pd.DataFrame, columns: Sequence[str], *, dayfirst: bool = True, format: str | None = None
) -> pd.DataFrame:
    """Convierte columnas a fecha. Por defecto día/mes/año (formato usual en Colombia).

    Las fechas inválidas quedan como ``NaT``.
    """
    df = df.copy()
    for col in columns:
        df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, format=format, errors="coerce")
    return df


def add_datetime_features(
    df: pd.DataFrame,
    column: str,
    *,
    parts: Sequence[str] = ("year", "month", "dayofweek"),
    drop: bool = False,
) -> pd.DataFrame:
    """Crea columnas ``<columna>_<parte>`` (año, mes, día de la semana, ...) a partir de una fecha.

    ``parts`` acepta cualquier atributo de ``Series.dt`` (``"quarter"``, ``"day"``, ``"hour"``...).
    """
    fechas = pd.to_datetime(df[column], errors="coerce")
    df = df.assign(**{f"{column}_{p}": getattr(fechas.dt, p) for p in parts})
    return df.drop(columns=column) if drop else df


def drop_duplicate_rows(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    """Elimina filas duplicadas (por todas las columnas o por ``subset``, ej. el id)."""
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def clean_dataset(
    df: pd.DataFrame,
    *,
    normalize_names: bool = True,
    lowercase_text: bool = False,
    na_tokens: Iterable[str] = DEFAULT_NA_TOKENS,
    numeric_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] = (),
    decimal: str = ".",
    thousands: str | None = ",",
    duplicate_subset: Sequence[str] | None = None,
    drop_empty: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Limpieza general sin estado: aplica, en orden, todas las funciones de este bloque.

    1. nombres de columnas en snake_case        5. fechas (``date_columns``)
    2. espacios sobrantes en textos             6. filas duplicadas
    3. textos tipo 'N/A' -> nulo                7. filas y columnas 100 % vacías
    4. textos numéricos -> número

    Ojo: si ``normalize_names=True``, los nombres de ``numeric_columns``, ``date_columns`` y
    ``duplicate_subset`` deben escribirse ya normalizados (ej. ``"fecha_afiliacion"``).
    """
    steps = []
    if normalize_names:
        steps.append(("nombres de columnas", normalize_column_names, {}))
    steps += [
        ("espacios en textos", strip_strings, {"lowercase": lowercase_text}),
        ("nulos disfrazados", standardize_missing, {"tokens": na_tokens}),
        (
            "texto -> número",
            coerce_numeric,
            {"columns": numeric_columns, "decimal": decimal, "thousands": thousands},
        ),
    ]
    if date_columns:
        steps.append(("texto -> fecha", coerce_datetime, {"columns": date_columns}))
    steps.append(("filas duplicadas", drop_duplicate_rows, {"subset": duplicate_subset}))
    if drop_empty:
        steps.append(
            (
                "filas/columnas vacías",
                lambda d: d.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True),
                {},
            )
        )

    for name, func, kwargs in steps:
        before = df.shape
        nulls_before = int(df.isna().sum().sum())
        df = df.pipe(func, **kwargs)
        if verbose:
            nulls_after = int(df.isna().sum().sum())
            print(
                f"[limpieza] {name:<22} filas {before[0]:>7} -> {df.shape[0]:<7} "
                f"columnas {before[1]:>3} -> {df.shape[1]:<3} nulos {nulls_before} -> {nulls_after}"
            )
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por columna: tipo, nº y % de nulos y nº de valores únicos (útil en el EDA)."""
    return (
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "nulos": df.isna().sum(),
                "pct_nulos": (df.isna().mean() * 100).round(2),
                "unicos": df.nunique(dropna=True),
            }
        )
        .sort_values("pct_nulos", ascending=False)
        .rename_axis("columna")
    )


# ---------------------------------------------------------------------------
# 2. Transformadores con estado (van dentro del Pipeline; se ajustan solo con train)
# ---------------------------------------------------------------------------


class ColumnDropper(TransformerMixin, BaseEstimator):
    """Elimina columnas con demasiados nulos o con un único valor, decidido SOLO con train.

    Parámetros
    ----------
    max_missing : fracción máxima de nulos permitida (0.5 = se elimina si >50 % es nulo).
    drop_constant : elimina columnas con un solo valor distinto (no aportan información).
    columns : columnas a eliminar siempre (ej. identificadores que se colaron).
    """

    def __init__(self, max_missing: float = 0.5, drop_constant: bool = True, columns: Sequence[str] = ()):
        self.max_missing = max_missing
        self.drop_constant = drop_constant
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnDropper necesita un DataFrame (trabaja con nombres de columnas)")
        to_drop = set(self.columns) & set(X.columns)
        to_drop |= set(X.columns[X.isna().mean() > self.max_missing])
        if self.drop_constant:
            to_drop |= set(X.columns[X.nunique(dropna=True) <= 1])  # noqa: PD101 (es un DataFrame, no una Serie)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Se conserva el orden original de las columnas para que el resultado sea reproducible.
        self.columns_to_drop_ = [c for c in X.columns if c in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_to_drop_")
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "columns_to_drop_")
        return np.asarray([c for c in self.feature_names_in_ if c not in self.columns_to_drop_], dtype=object)


class OutlierClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Recorta (winsoriza) valores extremos de columnas numéricas con límites aprendidos en train.

    No elimina filas: los valores fuera de rango se reemplazan por el límite, así el número de
    filas de X e y no cambia y se puede usar dentro de un Pipeline.

    Parámetros
    ----------
    method : ``"iqr"`` -> límites Q1 - factor·IQR y Q3 + factor·IQR.
             ``"quantile"`` -> límites en los cuantiles ``quantiles`` (ej. 1 % y 99 %).
    factor : multiplicador del IQR (1.5 clásico; 3 solo recorta extremos muy marcados).
    quantiles : par (inferior, superior) para ``method="quantile"``.
    """

    def __init__(self, method: str = "iqr", factor: float = 1.5, quantiles: tuple[float, float] = (0.01, 0.99)):
        self.method = method
        self.factor = factor
        self.quantiles = quantiles

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if self.method == "iqr":
            q1, q3 = np.nanquantile(values, [0.25, 0.75], axis=0)
            iqr = q3 - q1
            self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        elif self.method == "quantile":
            self.lower_, self.upper_ = np.nanquantile(values, self.quantiles, axis=0)
        else:
            raise ValueError(f"method debe ser 'iqr' o 'quantile', no '{self.method}'")
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        if isinstance(X, pd.DataFrame):
            return X.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
