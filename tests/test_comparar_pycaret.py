"""PyCaret (extra opcional): se salta si no está instalado, como en el CI (uv sync --extra pycaret)."""

import importlib.util

import pytest

from proyecto_ml import abandono as a
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.generador import GeneradorAsociados

pytest.importorskip("pycaret")

# scripts/ no es un paquete: el CLI se carga desde su ruta para probar sus funciones.
_spec = importlib.util.spec_from_file_location("comparar_pycaret", ROOT_DIR / "scripts" / "comparar_pycaret.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.mark.integration
def test_pycaret_compara_y_devuelve_pipelines_que_generalizan():
    train = GeneradorAsociados(n=400, seed=1).generar()
    nuevos = GeneradorAsociados(n=400, seed=2).generar()
    tabla, modelos = cli.experimento(train, folds=3)

    assert sorted(tabla.index) == sorted(cli.MODELOS)
    assert list(modelos) == list(tabla.index)  # todos, en el orden de PyCaret
    clases = {"lr": "LogisticRegression", "rf": "RandomForest", "et": "ExtraTrees", "ada": "AdaBoost"}
    clases |= {"gbc": "GradientBoosting", "lightgbm": "LGBM"}
    for nombre, pipeline in modelos.items():
        assert type(pipeline.steps[-1][1]).__name__.startswith(clases[nombre])  # el nombre es el modelo correcto
        assert not set(pipeline.feature_names_in_) & set(a.FUGA)
        puntaje = pipeline.predict_proba(a.preparar(nuevos))[:, 1]
        assert cli.compute_metrics(nuevos[a.TARGET], puntaje)["roc_auc"] > 0.8
