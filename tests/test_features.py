import numpy as np
import pandas as pd
import pytest

from proyecto_ml.data import infer_column_types, split_features_target
from proyecto_ml.features import build_preprocessor, pagina_pipelines


def _xy(df):
    return split_features_target(df, "target", drop=["id_entidad", "periodo"])


def test_preprocessor_sin_nulos_y_categoria_nueva(synthetic_df):
    X, _ = _xy(synthetic_df)
    numeric, categorical = infer_column_types(X)
    pre = build_preprocessor(numeric, categorical)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()

    nuevo = X.head(1).copy()
    nuevo["sede"] = "sede_inexistente"
    assert pre.transform(nuevo).shape[1] == out.shape[1]


def test_pca_reduce_dimensiones(synthetic_df):
    X, _ = _xy(synthetic_df)
    numeric, categorical = infer_column_types(X)
    out = build_preprocessor(numeric, categorical, pca_components=3).fit_transform(X)
    assert out.shape[1] == 3


def test_sin_columnas_falla():
    with pytest.raises(ValueError):
        build_preprocessor([], [])


def test_infer_column_types():
    X = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
    numeric, categorical = infer_column_types(X)
    assert numeric == ["a"]
    assert set(categorical) == {"b", "c"}


def test_pipeline_con_pca_y_smote_entrena(synthetic_df):
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression

    from proyecto_ml.models import build_pipeline

    X, y = _xy(synthetic_df)
    numeric, categorical = infer_column_types(X)
    pre = build_preprocessor(numeric, categorical, pca_components=0.9)
    pipe = build_pipeline(pre, LogisticRegression(max_iter=500), SMOTE(random_state=0))
    proba = pipe.fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)


def test_pagina_pipelines_con_el_diagrama_de_cada_uno():
    from proyecto_ml import abandono as a
    from proyecto_ml.generador import GeneradorAsociados

    df = GeneradorAsociados(n=200, seed=1).generar()
    ridge = a.modelo_ridge().fit(a.preparar(df), df[a.TARGET])
    html = pagina_pipelines({"Ridge": ridge, "Ridge sin entrenar": a.modelo_ridge()}, "Pipelines <abandono>")

    assert html.startswith("<!doctype html>") and '<meta charset="utf-8">' in html
    assert "<h1>Pipelines &lt;abandono&gt;</h1>" in html  # el título se escapa
    assert html.count("<h2>") == 2
    for paso in ("ColumnTransformer", "SimpleImputer", "StandardScaler", "OneHotEncoder", "RidgeClassifier"):
        assert paso in html
    assert "#SOLICITUD_DISMINUCIONCUOTAS" in html and "TIPOCONTRATO" in html  # columnas de cada rama


def test_variantes_de_preprocesamiento_y_rama_de_recorte():
    from proyecto_ml import abandono as a
    from proyecto_ml.generador import FALTANTES, GeneradorAsociados

    X = a.preparar(GeneradorAsociados(n=300, seed=1, faltantes=FALTANTES).generar())
    pre = build_preprocessor(
        a.NUMERICAS,
        a.CATEGORICAS,
        numeric_strategy="mean",
        encoding="ordinal",
        clip_outliers="iqr",
        clip_columns=a.COLUMNAS_RECORTE,
    ).fit(X)

    assert [nombre for nombre, *_ in pre.transformers_ if nombre != "remainder"] == ["num", "clip", "cat"]
    assert list(pre.named_transformers_["clip"].feature_names_in_) == a.COLUMNAS_RECORTE
    assert "outliers" not in pre.named_transformers_["num"].named_steps  # los conteos no se recortan
    assert pre.named_transformers_["num"].named_steps["imputer"].strategy == "mean"
    assert pre.transform(X).shape[1] == len(a.NUMERICAS) + len(a.CATEGORICAS)  # ordinal: una columna por categórica

    # Recortar TODAS las numéricas aplasta los conteos (IQR [0, 0]); con clip_columns se conservan.
    def valores_distintos(**kwargs) -> int:
        p = build_preprocessor(a.NUMERICAS, [], scale=False, **kwargs).fit(X)
        salida = pd.DataFrame(p.transform(X), columns=p.get_feature_names_out())
        return salida["CANCELACIONCREDITOS_ULT3MESES"].nunique()

    assert valores_distintos(clip_outliers="iqr") == 1
    assert valores_distintos(clip_outliers="iqr", clip_columns=a.COLUMNAS_RECORTE) > 1

    with pytest.raises(ValueError, match="numéricas"):
        build_preprocessor(a.NUMERICAS, a.CATEGORICAS, clip_outliers="iqr", clip_columns=["SEXO"])
    with pytest.raises(ValueError, match="encoding"):
        build_preprocessor(a.NUMERICAS, a.CATEGORICAS, encoding="target")


def test_preprocesador_para_gridsearch_y_aplanar():
    from sklearn.pipeline import Pipeline

    from proyecto_ml import abandono as a
    from proyecto_ml.features import Preprocesador, aplanar
    from proyecto_ml.generador import GeneradorAsociados

    df = GeneradorAsociados(n=300, seed=2).generar()
    X, y = a.preparar(df), df[a.TARGET]
    opciones = {"numeric_strategy": "mean", "encoding": "ordinal", "clip_outliers": "iqr"}
    envuelto = Preprocesador(a.NUMERICAS, a.CATEGORICAS, clip_columns=a.COLUMNAS_RECORTE, **opciones).fit(X)
    directo = build_preprocessor(a.NUMERICAS, a.CATEGORICAS, clip_columns=a.COLUMNAS_RECORTE, **opciones).fit(X)
    assert np.array_equal(envuelto.transform(X), directo.transform(X))
    assert list(envuelto.get_feature_names_out()) == list(directo.get_feature_names_out())

    modelo = Pipeline([("preprocess", Preprocesador(a.NUMERICAS, a.CATEGORICAS)), ("model", a.modelo_ridge()[-1])])
    modelo.set_params(preprocess__encoding="ordinal").fit(X, y)  # como lo cambia GridSearchCV
    plano = aplanar(modelo)
    assert type(plano.named_steps["preprocess"]).__name__ == "ColumnTransformer"  # ramas visibles en el diagrama
    assert np.array_equal(plano.predict(X), modelo.predict(X))


def test_clean_dataset_y_luego_el_preprocesador():
    """El uso típico de cleaning.py: clean_dataset deja el texto con pd.NA y el preprocesador debe aceptarlo."""
    from proyecto_ml.cleaning import clean_dataset

    crudo = pd.DataFrame(
        {"salario": ["1,200", "N/A", "3,400", "2,100"], "sede": ["Norte", "  Sur ", "-", "Norte"], "y": [0, 1, 0, 1]}
    )
    df = clean_dataset(crudo, normalize_names=False, verbose=False)
    assert pd.api.types.is_string_dtype(df["sede"]) and df["sede"].isna().sum() == 1  # pd.NA

    pre = build_preprocessor(["salario"], ["sede"]).fit(df)
    out = pre.transform(df)
    assert not np.isnan(out).any()
    assert list(pre.get_feature_names_out()) == ["salario", "sede_Norte", "sede_Sur"]
