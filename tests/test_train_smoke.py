"""Prueba de integración: entrenamiento completo sobre datos sintéticos."""

import json

import pytest

from proyecto_ml import train


@pytest.mark.integration
def test_train_end_to_end(tmp_path, monkeypatch, synthetic_df):
    data_path = tmp_path / "sintetico.parquet"
    synthetic_df.to_parquet(data_path, index=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("ID_COLUMN", "id_entidad")
    monkeypatch.setenv("TIME_COLUMN", "periodo")
    monkeypatch.delenv("GROUP_COLUMN", raising=False)

    results = train.run(
        [
            "--data",
            str(data_path),
            "--target",
            "target",
            "--models",
            "logreg",
            "--no-log-model",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert results["logreg"]["roc_auc"] > 0.6
    meta = json.loads((tmp_path / "best_model.json").read_text(encoding="utf-8"))
    assert "id_entidad" not in meta["numeric"] + meta["categorical"]
    assert (tmp_path / "best_model.joblib").exists()


@pytest.mark.integration
def test_train_con_limpieza_registra_las_opciones(tmp_path, monkeypatch, synthetic_df):
    import joblib
    import mlflow

    data_path = tmp_path / "sintetico.parquet"
    synthetic_df.to_parquet(data_path, index=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("ID_COLUMN", "id_entidad")
    monkeypatch.setenv("TIME_COLUMN", "periodo")
    monkeypatch.delenv("GROUP_COLUMN", raising=False)

    opciones = ["--clip-outliers", "iqr", "--clip-columns", "salario,saldo_ahorro", "--drop-columns", "0.5"]
    args = ["--data", str(data_path), "--target", "target", "--models", "logreg", "--no-log-model"]
    results = train.run([*args, "--output-dir", str(tmp_path), *opciones])
    assert results["logreg"]["roc_auc"] > 0.6

    # El recorte va en su propia rama, solo con las columnas pedidas.
    columnas = joblib.load(tmp_path / "best_model.joblib").named_steps["columns"]
    assert columnas.named_transformers_["clip"].named_steps["outliers"].method == "iqr"
    assert list(columnas.named_transformers_["clip"].feature_names_in_) == ["salario", "saldo_ahorro"]

    mlflow.set_tracking_uri(uri)
    params = mlflow.search_runs(search_all_experiments=True).iloc[0]
    assert params["params.clip_outliers"] == "iqr"
    assert params["params.clip_columns"] == "salario,saldo_ahorro"
    assert params["params.drop_columns"] == "0.5"
