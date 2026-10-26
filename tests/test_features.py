import numpy as np
import pandas as pd
import pytest

from proyecto_ml.data import infer_column_types, split_features_target
from proyecto_ml.features import build_preprocessor


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
