import pandas as pd
import pytest

from proyecto_ml.data import load_dataset, split_data, split_features_target


def test_split_estratificado_conserva_tasa(synthetic_df):
    X, y = split_features_target(synthetic_df, "target")
    _, _, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_temporal_no_mezcla_futuro(synthetic_df):
    X, y = split_features_target(synthetic_df, "target")
    X_train, X_test, _, _ = split_data(X, y, time=synthetic_df["periodo"])
    assert X_train["periodo"].max() < X_test["periodo"].min()


def test_split_por_grupos_sin_solapamiento():
    df = pd.DataFrame({"g": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "x": range(10), "target": [0, 1] * 5})
    X, y = split_features_target(df, "target")
    X_train, X_test, _, _ = split_data(X, y, groups=df["g"], test_size=0.4)
    assert set(X_train["g"]).isdisjoint(X_test["g"])


def test_target_inexistente(synthetic_df):
    with pytest.raises(KeyError):
        split_features_target(synthetic_df, "no_existe")


def test_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nada.csv")
