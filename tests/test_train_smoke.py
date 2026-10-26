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
