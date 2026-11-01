import numpy as np
import pandas as pd
import pytest

from proyecto_ml.cleaning import (
    ColumnDropper,
    OutlierClipper,
    add_datetime_features,
    clean_dataset,
    coerce_numeric,
    missing_report,
    normalize_column_names,
    standardize_missing,
)
from proyecto_ml.data import infer_column_types, split_features_target
from proyecto_ml.features import build_preprocessor


@pytest.fixture
def sucio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Fecha de Afiliación ": ["01/02/2023", "15/03/2023", "15/03/2023", "fecha mala", None],
            "Salario ($)": ["$1,200.50", "900", "900", "N/A", "1,000"],
            "Ciudad": ["  Medellín ", "Bogotá", "Bogotá", "-", "Cali  "],
            "Vacía": [None, None, None, None, None],
        }
    )


def test_normalize_column_names_snake_case_y_repetidos():
    df = pd.DataFrame(columns=["Fecha de Afiliación", "fecha-de-afiliacion", "%"])
    assert normalize_column_names(df).columns.tolist() == [
        "fecha_de_afiliacion",
        "fecha_de_afiliacion_2",
        "columna",
    ]


def test_standardize_missing_convierte_tokens_en_nulos():
    df = pd.DataFrame({"a": ["x", " N/A ", "-", "", "?", "y"]})
    assert standardize_missing(df)["a"].isna().sum() == 4


def test_coerce_numeric_respeta_categoricas_y_formato_colombiano():
    df = pd.DataFrame({"monto": ["1.234,5", "10,0", None], "sede": ["norte", "sur", "centro"]})
    out = coerce_numeric(df, decimal=",", thousands=".")
    assert out["monto"].tolist()[:2] == [1234.5, 10.0]
    assert out["sede"].tolist() == ["norte", "sur", "centro"]


def test_clean_dataset_completo(sucio):
    df = clean_dataset(sucio, date_columns=["fecha_de_afiliacion"], verbose=False)
    assert df.columns.tolist() == ["fecha_de_afiliacion", "salario", "ciudad"]  # 'vacía' se elimina
    assert len(df) == 3  # una fila duplicada y otra que queda vacía tras limpiar
    assert pd.api.types.is_float_dtype(df["salario"])
    assert df["salario"].iloc[0] == pytest.approx(1200.5)
    assert pd.api.types.is_datetime64_any_dtype(df["fecha_de_afiliacion"])
    assert df["fecha_de_afiliacion"].iloc[0] == pd.Timestamp("2023-02-01")  # día primero
    assert df["ciudad"].tolist() == ["Medellín", "Bogotá", "Cali"]


def test_add_datetime_features():
    df = pd.DataFrame({"f": pd.to_datetime(["2024-01-15", "2024-06-01"])})
    out = add_datetime_features(df, "f", parts=("year", "month"), drop=True)
    assert out.columns.tolist() == ["f_year", "f_month"]
    assert out["f_month"].tolist() == [1, 6]


def test_missing_report_ordenado():
    rep = missing_report(pd.DataFrame({"a": [1, None], "b": [1, 2]}))
    assert rep.index[0] == "a"
    assert rep.loc["a", "pct_nulos"] == 50.0


def test_column_dropper_aprende_solo_de_train():
    train = pd.DataFrame({"ok": [1, 2, 3, 4], "nulos": [1, None, None, None], "cte": [7, 7, 7, 7]})
    test = pd.DataFrame({"ok": [5], "nulos": [9], "cte": [8]})
    dropper = ColumnDropper(max_missing=0.5).fit(train)
    assert dropper.columns_to_drop_ == ["nulos", "cte"]
    assert dropper.transform(test).columns.tolist() == ["ok"]
    assert dropper.get_feature_names_out().tolist() == ["ok"]


@pytest.mark.parametrize("method", ["iqr", "quantile"])
def test_outlier_clipper_recorta_sin_eliminar_filas(method):
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    clipper = OutlierClipper(method=method).fit(train)
    out = clipper.transform(pd.DataFrame({"x": [-1000.0, 50.0, 1000.0]}))
    assert len(out) == 3
    assert out["x"].iloc[1] == 50.0
    assert out["x"].iloc[0] >= clipper.lower_[0] and out["x"].iloc[2] <= clipper.upper_[0]
    # también funciona con arrays (como llega dentro del ColumnTransformer)
    assert clipper.transform(np.array([[1000.0]]))[0, 0] == clipper.upper_[0]


def test_outlier_clipper_metodo_invalido():
    with pytest.raises(ValueError):
        OutlierClipper(method="zscore").fit(np.ones((3, 1)))


def test_preprocessor_con_limpieza_en_pipeline(synthetic_df):
    X, _ = split_features_target(synthetic_df, "target", drop=["id_entidad", "periodo"])
    X = X.assign(casi_vacia=np.where(np.arange(len(X)) % 10 == 0, 1.0, np.nan), constante="a")
    numeric, categorical = infer_column_types(X)
    pre = build_preprocessor(
        numeric, categorical, clip_outliers="iqr", drop_columns=ColumnDropper(max_missing=0.5), pca_components=None
    )
    out = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    assert not np.isnan(out).any()
    assert "casi_vacia" not in names
    assert not any(n.startswith("constante") for n in names)
