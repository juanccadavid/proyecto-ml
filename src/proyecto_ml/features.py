"""Preprocesamiento con Pipeline + ColumnTransformer (imputación, escalado, OneHot, PCA)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from proyecto_ml.cleaning import ColumnDropper, OutlierClipper


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool = True,
    max_categories: int = 20,
    pca_components: int | float | None = None,
    clip_outliers: str | None = None,
    drop_columns: ColumnDropper | None = None,
    random_state: int = 42,
):
    """Construye el preprocesador.

    ``pca_components`` activa reducción de dimensionalidad después del ColumnTransformer
    (entero = nº de componentes, float en (0, 1) = varianza explicada a conservar).

    ``clip_outliers`` (``"iqr"`` o ``"quantile"``) recorta atípicos de las numéricas tras imputar.
    ``drop_columns`` (ej. ``ColumnDropper(max_missing=0.5)``) descarta columnas antes del
    ColumnTransformer; las columnas descartadas se ignoran sin error.
    Ambos aprenden sus límites solo con los datos de ``fit`` (train): sin fuga.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if clip_outliers:
        numeric_steps.append(("outliers", OutlierClipper(method=clip_outliers)))
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
