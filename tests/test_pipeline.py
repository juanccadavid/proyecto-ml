"""El Pipeline aprende SOLO de train y aplica esas mismas reglas a test.

1. Separar train/test.
2. pipeline.fit(X_train, y_train): aprende medianas, escalado y categorías del one-hot SOLO con train.
3. pipeline.predict(X_test): aplica a test lo aprendido; no recalcula nada con test.
Si el preprocesamiento se ajustara con todo el dataset, test "filtraría" información a train y la
evaluación saldría más optimista de lo que será en producción.
"""

import numpy as np
import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml.data import split_data
from proyecto_ml.features import build_preprocessor
from proyecto_ml.generador import GeneradorAsociados


def test_rellena_test_con_la_mediana_de_train():
    train = pd.DataFrame({"salario": [1.0, 2.0, 3.0, np.nan], "cargo": ["A", "B", "A", "B"]})
    test = pd.DataFrame({"salario": [np.nan, 100.0, 200.0], "cargo": ["A", "A", "NUEVO"]})

    pre = build_preprocessor(["salario"], ["cargo"], scale=False).fit(train)
    out = pre.transform(test)

    assert out[0, 0] == 2.0  # mediana de train (1, 2, 3), no la de test (150)
    assert out.shape[1] == 3  # salario + one-hot de las categorías vistas en train (A, B)
    assert out[2, 1:].tolist() == [0.0, 0.0]  # categoría que train nunca vio: sin columna propia


def test_pipeline_de_abandono_solo_aprende_de_train():
    df = GeneradorAsociados(n=300, seed=5).generar()
    X, y = a.preparar(df), df[a.TARGET]
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.25, random_state=0)

    modelo = a.modelo_ridge().fit(X_train, y_train)

    escalador = modelo.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    assert np.allclose(escalador.mean_, X_train[a.NUMERICAS].mean())  # medias de train...
    assert not np.allclose(escalador.mean_, X[a.NUMERICAS].mean())  # ...no de todo el dataset
    assert len(modelo.predict(X_test)) == len(X_test)
