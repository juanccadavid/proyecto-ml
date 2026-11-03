"""Preprocesamiento con Pipeline + ColumnTransformer (imputación, recorte, escalado, OneHot/ordinal, PCA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from proyecto_ml.cleaning import ColumnDropper, OutlierClipper


def _na_a_nan(X):
    """pd.NA -> np.nan. clean_dataset deja el texto como dtype 'string' (vacío = pd.NA) y SimpleImputer no lo admite."""
    X = pd.DataFrame(X)
    return X.astype(object).where(X.notna(), np.nan)


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool = True,
    max_categories: int = 20,
    numeric_strategy: str = "median",
    encoding: str = "onehot",
    pca_components: int | float | None = None,
    clip_outliers: str | None = None,
    clip_columns: list[str] | None = None,
    drop_columns: ColumnDropper | None = None,
    random_state: int = 42,
):
    """Construye el preprocesador.

    ``numeric_strategy`` rellena los vacíos numéricos: ``"median"`` (robusta a atípicos) o ``"mean"``.
    ``encoding`` codifica las categóricas: ``"onehot"`` (una columna 0/1 por categoría) u ``"ordinal"``
    (un número por categoría, como PyCaret; las categorías que train no vio valen -1).

    ``pca_components`` activa reducción de dimensionalidad después del ColumnTransformer
    (entero = nº de componentes, float en (0, 1) = varianza explicada a conservar).

    ``clip_outliers`` (``"iqr"`` o ``"quantile"``) recorta atípicos de las numéricas tras imputar.
    Con ``clip_columns`` solo se recortan esas columnas, en una rama aparte (imputar -> recortar ->
    escalar): en conteos con casi todo en 0 el IQR vale [0, 0] y recortar borraría la señal.
    Sin ``clip_outliers``, ``clip_columns`` no hace nada (así se puede apagar el recorte en una búsqueda).
    ``drop_columns`` (ej. ``ColumnDropper(max_missing=0.5)``) descarta columnas antes del
    ColumnTransformer; las columnas descartadas se ignoran sin error.
    Todos aprenden sus valores solo con los datos de ``fit`` (train): sin fuga.
    """
    if clip_outliers and clip_columns:
        sobran = sorted(set(clip_columns) - set(numeric))
        if sobran:
            raise ValueError(f"clip_columns deben ser numéricas: {sobran}")
        recortar = [c for c in numeric if c in clip_columns]
        numeric = [c for c in numeric if c not in clip_columns]
    else:
        recortar = []

    def numeric_steps(clip: str | None) -> list:
        steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
        if clip:
            steps.append(("outliers", OutlierClipper(method=clip)))
        if scale:
            steps.append(("scaler", StandardScaler()))
        return steps

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=max_categories, sparse_output=False
        )
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"encoding debe ser 'onehot' u 'ordinal', no '{encoding}'")
    categorical_steps = [
        ("na", FunctionTransformer(_na_a_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps(None if recortar else clip_outliers)), numeric))
    if recortar:
        transformers.append(("clip", Pipeline(numeric_steps(clip_outliers)), recortar))
    if categorical:
        transformers.append(("cat", Pipeline(categorical_steps), categorical))
    if not transformers:
        raise ValueError("No hay columnas para preprocesar")

    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    if drop_columns is None and pca_components is None:
        return column_transformer

    steps = []
    if drop_columns is not None:
        steps.append(("drop_columns", drop_columns))
        # Las columnas que ColumnDropper elimine no existirán al llegar al ColumnTransformer:
        # se seleccionan con una función que filtra las que sigan presentes.
        column_transformer.transformers = [
            (name, pipe, _present(cols)) for name, pipe, cols in column_transformer.transformers
        ]
    steps.append(("columns", column_transformer))
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))
    return Pipeline(steps)


class _present:
    """Selector de columnas para ColumnTransformer: devuelve solo las que existen en X.

    Es una clase (no una lambda) para que el pipeline se pueda serializar con joblib/skops.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def __call__(self, X):
        return [c for c in self.columns if c in X.columns]


class Preprocesador(TransformerMixin, BaseEstimator):
    """``build_preprocessor`` como estimador con parámetros simples, para buscarlos con GridSearchCV.

    Ej. en un Pipeline con el paso ``"preprocess"``::

        {"preprocess__numeric_strategy": ["median", "mean"], "preprocess__encoding": ["onehot", "ordinal"],
         "preprocess__clip_outliers": [None, "iqr"]}

    ``max_nulos`` activa el ``ColumnDropper`` (descarta columnas con más de esa fracción de vacíos).

    El preprocesador real se arma y se ajusta en ``fit`` (``preprocessor_``). Para guardarlo o verlo en
    el diagrama, ``aplanar`` lo cambia por ese preprocesador ya ajustado.
    """

    def __init__(
        self,
        numeric: list[str] = (),
        categorical: list[str] = (),
        numeric_strategy: str = "median",
        encoding: str = "onehot",
        clip_outliers: str | None = None,
        clip_columns: list[str] | None = None,
        max_nulos: float | None = None,
    ):
        self.numeric = numeric
        self.categorical = categorical
        self.numeric_strategy = numeric_strategy
        self.encoding = encoding
        self.clip_outliers = clip_outliers
        self.clip_columns = clip_columns
        self.max_nulos = max_nulos

    def fit(self, X, y=None):
        self.preprocessor_ = build_preprocessor(
            list(self.numeric),
            list(self.categorical),
            numeric_strategy=self.numeric_strategy,
            encoding=self.encoding,
            clip_outliers=self.clip_outliers,
            clip_columns=self.clip_columns,
            drop_columns=ColumnDropper(max_missing=self.max_nulos) if self.max_nulos is not None else None,
        ).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.get_feature_names_out()


def aplanar(pipeline: Pipeline) -> Pipeline:
    """Cambia cada ``Preprocesador`` ajustado por el preprocesador real que armó (también ajustado).

    Predice exactamente igual, pero el diagrama muestra cada rama y paso, y al guardarlo no depende
    de ``Preprocesador``.
    """
    return Pipeline(
        [(nombre, paso.preprocessor_ if isinstance(paso, Preprocesador) else paso) for nombre, paso in pipeline.steps]
    )


def pagina_pipelines(modelos: dict, titulo: str = "Pipelines") -> str:
    """Página HTML con el diagrama interactivo de cada pipeline: el mismo que muestra Jupyter.

    Cada caja (ColumnTransformer, cada rama numérica/categórica, cada paso) se despliega con un clic
    para ver sus parámetros y columnas; ``?`` abre la documentación de scikit-learn. En un notebook
    no hace falta: basta con dejar el pipeline al final de una celda.
    """
    from html import escape

    from sklearn.utils import estimator_html_repr

    secciones = "\n".join(f"<h2>{escape(nombre)}</h2>\n{estimator_html_repr(m)}" for nombre, m in modelos.items())
    estilo = "body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}h2{margin-top:32px}"
    return (
        f'<!doctype html>\n<html lang="es">\n<head><meta charset="utf-8"><title>{escape(titulo)}</title>'
        f"<style>{estilo}</style></head>\n<body>\n<h1>{escape(titulo)}</h1>\n{secciones}\n</body>\n</html>\n"
    )
