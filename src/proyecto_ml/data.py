"""Carga de datos, separación X/y y estrategias de partición train/test."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Los datos NO están en Git: pídelos por el canal privado "
            "del equipo y ubícalos en data/raw o data/processed."
        )
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Formato no soportado: {path.suffix}")
    return reader(path)


def split_features_target(
    df: pd.DataFrame, target: str, drop: Iterable[str | None] = ()
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"La columna objetivo '{target}' no está en el dataset")
    to_drop = [c for c in {target, *drop} if c and c in df.columns]
    return df.drop(columns=to_drop), df[target]


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas). Booleanos se tratan como categóricos."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    time: pd.Series | None = None,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición train/test.

    - Con ``time``: partición temporal (lo más reciente va a test). Evita fuga de futuro.
    - Con ``groups``: ninguna entidad (ej. asociado) queda en train y test a la vez.
    - Sin ninguno: partición aleatoria estratificada.
    """
    if time is not None:
        cutoff = time.quantile(1 - test_size)
        train_mask = time < cutoff
        if groups is not None:
            test_groups = set(groups[~train_mask])
            train_mask &= ~groups.isin(test_groups)
        test_mask = time >= cutoff
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
