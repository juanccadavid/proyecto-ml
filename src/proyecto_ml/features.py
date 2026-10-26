"""Preprocesamiento con Pipeline + ColumnTransformer (imputación, escalado, OneHot, PCA)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool = True,
    max_categories: int = 20,
    pca_components: int | float | None = None,
    random_state: int = 42,
):
    """Construye el preprocesador.

    ``pca_components`` activa reducción de dimensionalidad después del ColumnTransformer
    (entero = nº de componentes, float en (0, 1) = varianza explicada a conservar).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=max_categories,
                sparse_output=False,
            ),
        ),
    ]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(categorical_steps), categorical))
    if not transformers:
        raise ValueError("No hay columnas para preprocesar")

    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    if pca_components is None:
        return column_transformer
    return Pipeline(
        [
            ("columns", column_transformer),
            ("pca", PCA(n_components=pca_components, random_state=random_state)),
        ]
    )
