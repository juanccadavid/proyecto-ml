"""El pipeline del abandono (abandono.armar_pipeline): recibe la fila cruda y la prepara él mismo."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from proyecto_ml import abandono as a
from proyecto_ml import registro as r
from proyecto_ml.features import Preprocesador
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def datos():
    entrenamiento = GeneradorAsociados(n=300, seed=4, tasa_retiro=0.25).generar()
    nuevos = GeneradorAsociados(n=80, seed=5, tasa_retiro=0.25).generar()
    return entrenamiento, nuevos


def test_predice_igual_con_la_fila_cruda_o_preparada(datos):
    entrenamiento, nuevos = datos
    modelo = a.armar_pipeline(LogisticRegression(max_iter=2000)).fit(*a.separar(entrenamiento))
    crudo = modelo.predict_proba(nuevos.drop(columns=a.TARGET))[:, 1]
    assert np.allclose(crudo, modelo.predict_proba(a.preparar(nuevos))[:, 1])
    # y es lo mismo que entrenar con preparar() a mano
    a_mano = a.armar_pipeline(LogisticRegression(max_iter=2000)).fit(a.preparar(entrenamiento), entrenamiento[a.TARGET])
    assert np.allclose(crudo, a_mano.predict_proba(nuevos)[:, 1])


def test_preparar_variables_es_el_primer_paso_y_no_aprende():
    paso = a.PrepararVariables()
    assert paso.fit(None) is paso
    assert list(paso.get_feature_names_out()) == a.NUMERICAS + a.CATEGORICAS


def test_pasos_segun_las_opciones():
    def pasos(**kw):
        return [n for n, _ in a.armar_pipeline(LogisticRegression(), **kw).steps]

    assert pasos() == ["preparar", "preprocess", "model"]
    assert pasos(crear_variables=True) == ["preparar", "crear", "preprocess", "model"]
    con_smote = a.armar_pipeline(LogisticRegression(), muestreo="smote")
    assert [n for n, _ in con_smote.steps] == ["preparar", "preprocess", "sampler", "model"]
    assert type(con_smote).__module__.startswith("imblearn")
    assert isinstance(con_smote.named_steps["preprocess"], Preprocesador)  # imblearn: sin Pipeline anidado
    buscable = a.armar_pipeline(LogisticRegression(), buscable=True).set_params(preprocess__scaler="robust")
    assert buscable.named_steps["preprocess"].scaler == "robust"
    with pytest.raises(ValueError, match="muestreo"):
        a.armar_pipeline(LogisticRegression(), muestreo="tomek")


def test_opciones_del_preprocesador_y_variables_nuevas(datos):
    entrenamiento, nuevos = datos
    modelo = a.armar_pipeline(LogisticRegression(max_iter=2000), crear_variables=True, pca=5)
    modelo.fit(*a.separar(entrenamiento))
    pasos = modelo.named_steps
    con_nuevas = pasos["crear"].transform(pasos["preparar"].transform(nuevos))
    assert set(a.VARIABLES_NUEVAS) <= set(con_nuevas.columns)
    assert pasos["preprocess"].transform(con_nuevas).shape[1] == 5  # PCA a 5 componentes


def test_supervisados_y_campeon_con_el_mismo_esqueleto():
    for modelo in a.modelos_supervisados().values():
        assert [n for n, _ in modelo.steps] == ["preparar", "preprocess", "model"]


def test_skops_acepta_los_pasos_nuevos(datos):
    pytest.importorskip("skops")
    entrenamiento, _ = datos
    modelo = a.armar_pipeline(LogisticRegression(max_iter=2000), crear_variables=True).fit(*a.separar(entrenamiento))
    tipos = r._tipos_confiables(modelo)
    assert tipos is not None  # todos del repo/sklearn/numpy: se guarda con skops, no con cloudpickle
    assert "proyecto_ml.abandono.PrepararVariables" in tipos
