"""Preprocesamiento con Pipeline + ColumnTransformer.

Imputación (mediana/media, KNN, iterativa, + indicador de vacíos), recorte de atípicos, escalado (estándar,
min-max, max-abs, robusto, normalizer), codificación (OneHot, ordinal, target), binning, PCA y RFE.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (habilita IterativeImputer)
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import check_is_fitted

from proyecto_ml.cleaning import ColumnDropper, OutlierClipper

IMPUTADORES = ("median", "mean", "knn", "iterative")
ESCALADORES = {
    "standard": StandardScaler,  # media 0, desviación 1
    "minmax": MinMaxScaler,  # a [0, 1]: sensible a atípicos
    "maxabs": MaxAbsScaler,  # a [-1, 1] sin centrar (conserva los ceros)
    "robust": RobustScaler,  # mediana y rango intercuartílico: robusto a atípicos
    "normalizer": Normalizer,  # cada FILA a norma 1 (no cada columna)
}
CODIFICACIONES = ("onehot", "ordinal", "target")


def _na_a_nan(X):
    """pd.NA -> np.nan. clean_dataset deja el texto como dtype 'string' (vacío = pd.NA) y SimpleImputer no lo admite."""
    X = pd.DataFrame(X)
    return X.astype(object).where(X.notna(), np.nan)


def _imputador(estrategia: str, indicador: bool, random_state: int):
    """Imputador numérico. ``indicador`` agrega una columna 0/1 por variable con vacíos (MissingIndicator)."""
    if estrategia in ("median", "mean"):
        return SimpleImputer(strategy=estrategia, add_indicator=indicador)
    if estrategia == "knn":  # promedio de los 5 vecinos más parecidos (en las demás columnas)
        return KNNImputer(n_neighbors=5, add_indicator=indicador)
    if estrategia == "iterative":  # predice cada columna con las demás, en rondas (tipo MICE)
        return IterativeImputer(max_iter=10, random_state=random_state, add_indicator=indicador)
    raise ValueError(f"numeric_strategy debe ser uno de {IMPUTADORES}, no '{estrategia}'")


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
    scaler: str = "standard",
    missing_indicator: bool = False,
    bins: int | None = None,
    bin_columns: list[str] | None = None,
    rfe_features: int | None = None,
):
    """Construye el preprocesador.

    ``numeric_strategy`` rellena los vacíos numéricos: ``"median"`` (robusta a atípicos), ``"mean"``,
    ``"knn"`` (KNNImputer) o ``"iterative"`` (IterativeImputer). ``missing_indicator`` agrega además una
    columna 0/1 por variable que tenía vacíos en train (MissingIndicator): el vacío también puede ser señal.
    ``scaler`` (con ``scale=True``): ``"standard"``, ``"minmax"``, ``"maxabs"``, ``"robust"`` o ``"normalizer"``.
    ``encoding`` codifica las categóricas: ``"onehot"`` (una columna 0/1 por categoría), ``"ordinal"``
    (un número por categoría, como PyCaret; las categorías que train no vio valen -1) o ``"target"``
    (TargetEncoder: la tasa del objetivo por categoría, con validación cruzada interna para no filtrarlo).

    ``pca_components`` activa reducción de dimensionalidad después del ColumnTransformer
    (entero = nº de componentes, float en (0, 1) = varianza explicada a conservar).

    ``clip_outliers`` (``"iqr"`` o ``"quantile"``) recorta atípicos de las numéricas tras imputar.
    Con ``clip_columns`` solo se recortan esas columnas, en una rama aparte (imputar -> recortar ->
    escalar): en conteos con casi todo en 0 el IQR vale [0, 0] y recortar borraría la señal.
    Sin ``clip_outliers``, ``clip_columns`` no hace nada (así se puede apagar el recorte en una búsqueda).
    ``drop_columns`` (ej. ``ColumnDropper(max_missing=0.5)``) descarta columnas antes del
    ColumnTransformer; las columnas descartadas se ignoran sin error.

    ``bins`` + ``bin_columns`` (binning): esas numéricas se cambian por ``bins`` rangos de igual frecuencia
    (KBinsDiscretizer, una columna 0/1 por rango; ya no se recortan). ``rfe_features`` (RFE) se queda con
    las mejores ``rfe_features`` columnas ya transformadas, quitando las menos útiles para una logística.
    Todos aprenden sus valores solo con los datos de ``fit`` (train): sin fuga.
    """
    if scale and scaler not in ESCALADORES:
        raise ValueError(f"scaler debe ser uno de {sorted(ESCALADORES)}, no '{scaler}'")
    binear = [c for c in numeric if bins and c in (bin_columns or [])]
    numeric = [c for c in numeric if c not in binear]
    recortar = []
    # Con clip_columns solo se recortan esas (aunque el binning las tome todas: los conteos nunca se recortan).
    solo_algunas = bool(clip_outliers and clip_columns)
    if clip_outliers and clip_columns:
        sobran = sorted(set(clip_columns) - set(numeric) - set(binear))
        if sobran:
            raise ValueError(f"clip_columns deben ser numéricas: {sobran}")
        recortar = [c for c in numeric if c in clip_columns]
        numeric = [c for c in numeric if c not in clip_columns]

    def numeric_steps(clip: str | None) -> list:
        steps = [("imputer", _imputador(numeric_strategy, missing_indicator, random_state))]
        if clip:
            steps.append(("outliers", OutlierClipper(method=clip)))
        if scale:
            steps.append(("scaler", ESCALADORES[scaler]()))
        return steps

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=max_categories, sparse_output=False
        )
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif encoding == "target":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)  # validación cruzada interna
        encoder = TargetEncoder(target_type="binary", cv=cv)
    else:
        raise ValueError(f"encoding debe ser uno de {CODIFICACIONES}, no '{encoding}'")
    categorical_steps = [
        ("na", FunctionTransformer(_na_a_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps(None if solo_algunas else clip_outliers)), numeric))
    if recortar:
        transformers.append(("clip", Pipeline(numeric_steps(clip_outliers)), recortar))
    if binear:
        binning = [
            ("imputer", SimpleImputer(strategy="median")),
            ("bins", KBinsDiscretizer(n_bins=bins, encode="onehot-dense", strategy="quantile")),
        ]
        transformers.append(("bins", Pipeline(binning), binear))
    if categorical:
        transformers.append(("cat", Pipeline(categorical_steps), categorical))
    if not transformers:
        raise ValueError("No hay columnas para preprocesar")

    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    if drop_columns is None and pca_components is None and rfe_features is None:
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
    if rfe_features is not None:
        estimador = LogisticRegression(max_iter=2000, class_weight="balanced")
        steps.append(("rfe", RFE(estimador, n_features_to_select=rfe_features, step=0.1)))
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
    ``scaler``, ``missing_indicator``, ``bins``/``bin_columns`` y ``rfe_features``: como en
    ``build_preprocessor``; ``pca`` es su ``pca_components``.

    El preprocesador real se arma y se ajusta en ``fit`` (``preprocessor_``). Para guardarlo o verlo en
    el diagrama, ``aplanar`` lo cambia por ese preprocesador ya ajustado.
    """

    def __init__(
        self,
        numeric: Sequence[str] = (),
        categorical: Sequence[str] = (),
        numeric_strategy: str = "median",
        encoding: str = "onehot",
        clip_outliers: str | None = None,
        clip_columns: list[str] | None = None,
        max_nulos: float | None = None,
        scaler: str = "standard",
        missing_indicator: bool = False,
        bins: int | None = None,
        bin_columns: list[str] | None = None,
        pca: int | float | None = None,
        rfe_features: int | None = None,
        random_state: int = 42,
    ):
        self.numeric = numeric
        self.categorical = categorical
        self.numeric_strategy = numeric_strategy
        self.encoding = encoding
        self.clip_outliers = clip_outliers
        self.clip_columns = clip_columns
        self.max_nulos = max_nulos
        self.scaler = scaler
        self.missing_indicator = missing_indicator
        self.bins = bins
        self.bin_columns = bin_columns
        self.pca = pca
        self.rfe_features = rfe_features
        self.random_state = random_state

    def fit(self, X, y=None):
        self.preprocessor_ = build_preprocessor(
            list(self.numeric),
            list(self.categorical),
            numeric_strategy=self.numeric_strategy,
            encoding=self.encoding,
            clip_outliers=self.clip_outliers,
            clip_columns=self.clip_columns,
            drop_columns=ColumnDropper(max_missing=self.max_nulos) if self.max_nulos is not None else None,
            scaler=self.scaler,
            missing_indicator=self.missing_indicator,
            bins=self.bins,
            bin_columns=self.bin_columns,
            pca_components=self.pca,
            rfe_features=self.rfe_features,
            random_state=self.random_state,
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
