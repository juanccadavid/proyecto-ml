"""Afinamiento de Ridge y pycaret_lr (extra opcional): se salta si PyCaret no está instalado, como en el CI."""

import importlib.util

import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from proyecto_ml import abandono as a
from proyecto_ml.config import ROOT_DIR
from proyecto_ml.evaluate import afinar
from proyecto_ml.generador import GeneradorAsociados

pytest.importorskip("pycaret")

# scripts/ no es un paquete: el CLI se carga desde su ruta para probar sus funciones.
_spec = importlib.util.spec_from_file_location("afinar_modelos", ROOT_DIR / "scripts" / "afinar_modelos.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.mark.integration
def test_grillas_validas_y_el_afinado_generaliza():
    train = GeneradorAsociados(n=300, seed=1).generar()
    nuevos = GeneradorAsociados(n=400, seed=2).generar()
    base = cli.modelos_base(train, folds=3)
    assert list(base) == list(cli.GRILLAS) == list(cli.EJE_X)
    assert type(base["pycaret_lr"].steps[-1][1]).__name__ == "LogisticRegression"

    for nombre, modelo in base.items():
        for params in ParameterGrid(cli.GRILLAS[nombre]):  # cada combinación existe en su Pipeline
            clone(modelo).set_params(**params)

    # Grilla reducida (L1 y L2) para que la prueba sea rápida.
    grilla = {**cli.GRILLAS["pycaret_lr"], "lr__C": [0.01, 1.0]}
    busqueda = afinar(base["pycaret_lr"], grilla, a.preparar(train), train[a.TARGET], folds=3)
    assert a.metricas_supervisado(busqueda.best_estimator_, nuevos)["roc_auc"] > 0.8
