"""Auditoría metodológica: origen de los datos, contrato obligatorio, fuga en el preprocesamiento y el muestreo,
métricas de clases desbalanceadas, trazabilidad en MLflow y reproducibilidad."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml import contrato as k
from proyecto_ml import registro as r
from proyecto_ml.evaluate import _column_transformer, compute_metrics, precision_at_k
from proyecto_ml.generador import FALTANTES, GeneradorAsociados
from proyecto_ml.models import get_models


@pytest.fixture
def sin_origen(monkeypatch):
    monkeypatch.delenv("ORIGEN_DATOS", raising=False)


@pytest.fixture(scope="module")
def foto():
    return a.foto(GeneradorAsociados(n=600, seed=11, tasa_retiro=0.25).generar(), 3)


# ---------------------------------------------------------------- origen de los datos y contrato


def test_origen_por_defecto_sintetico_y_valores_validos(sin_origen, monkeypatch):
    assert config.origen_datos() == config.ORIGEN_SINTETICO and config.es_experimental()
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    assert config.origen_datos() == config.ORIGEN_REAL and not config.es_experimental()
    assert "SINTÉTICO" in config.aviso_origen("sintetico") and "REALES" in config.aviso_origen("real_anonimizado")
    with pytest.raises(ValueError, match="ORIGEN_DATOS"):
        config.origen_datos("produccion")


def test_modo_real_no_cae_en_los_sinteticos(sin_origen):
    with pytest.raises(c.DatosInvalidos, match="--datos"):
        c.datos_eleccion(origen="real_anonimizado")
    df, fuente = c.datos_eleccion(origen="sintetico")  # el modo sintético sigue funcionando
    assert len(df) and "desarrollo" in fuente


def test_el_contrato_es_obligatorio_y_con_datos_reales_no_muestra_valores(tmp_path, sin_origen):
    sucio = GeneradorAsociados(n=80, seed=3, faltantes=FALTANTES).generar()  # vacíos en columnas requeridas
    archivo = tmp_path / "extracto.parquet"
    sucio.to_parquet(archivo)
    with pytest.raises(c.DatosInvalidos, match="no cumple el contrato") as sintetico:
        c.datos_eleccion(archivo, origen="sintetico")
    assert "ej. fila" in str(sintetico.value)  # en modo sintético se ve el ejemplo, para depurar
    with pytest.raises(c.DatosInvalidos) as real:
        c.datos_eleccion(archivo, origen="real_anonimizado")
    assert "ej." not in str(real.value) and "fila(s)" in str(real.value)  # cuántas y dónde, sin valores

    limpio = tmp_path / "limpio.parquet"
    GeneradorAsociados(n=80, seed=3).generar().to_parquet(limpio)
    df, fuente = c.datos_eleccion(limpio, origen="real_anonimizado")
    assert fuente == "limpio.parquet" and len(df) == 80


def test_sin_valores_quita_ejemplos_y_filas():
    errores = [
        "Nivel 1 · EMAIL: 4 fila(s) vacías en columna requerida (ej. fila 15: 'ana@x.com')",
        "Nivel 3 · retiro_coherente: 2 fila(s) no cumplen (FECHARETIRO si y solo si retiro); ej. [3, 9]",
        "Nivel 2 · tasa_retiro_plausible: tasa (observado: 0.5; límites: {'max': 0.4})",
    ]
    limpios = k.sin_valores(errores)
    assert limpios[0] == "Nivel 1 · EMAIL: 4 fila(s) vacías en columna requerida"
    assert "[3, 9]" not in limpios[1] and "retiro_coherente" in limpios[1]
    assert limpios[2] == errores[2]  # el nivel 2 solo trae agregados


# ---------------------------------------------------------------- fuga en el preprocesamiento y el muestreo


def test_el_preprocesamiento_aprende_solo_del_entrenamiento(foto):
    X, y = a.separar(foto)
    entrena, prueba = X.iloc[:400], X.iloc[400:].copy()
    prueba["SALARIO"] = 1e12  # si la prueba entrara al ajuste, la mediana se movería
    modelo = a.modelos_supervisados()["logreg"].fit(entrena, y.iloc[:400])
    imputer = _column_transformer(modelo).named_transformers_["num"].named_steps["imputer"]
    columnas = list(_column_transformer(modelo).named_transformers_["num"].feature_names_in_)
    medianas = a.preparar(entrena)[columnas].median()
    assert np.allclose(imputer.statistics_, medianas.to_numpy())
    antes = modelo.predict_proba(X.iloc[400:450])
    modelo.predict_proba(prueba)  # transformar la prueba no reajusta nada
    assert np.allclose(modelo.predict_proba(X.iloc[400:450]), antes)


def test_identificadores_y_fuga_no_son_predictores(foto):
    X, y = a.separar(foto)
    modelo = a.modelos_supervisados()["logreg"].fit(X, y)
    entradas = set(_column_transformer(modelo).feature_names_in_)
    prohibidas = {"CEDULA", "IDASOCIADOS", "NOMBRES", "EMAIL", "TELEFONO1", "NUMEROCUENTA", *a.FUGA}
    assert not entradas & prohibidas and entradas == set(a.NUMERICAS + a.CATEGORICAS)


class _SmoteEspia(SMOTE):
    """SMOTE que anota cuántas filas re-muestrea cada vez que se llama."""

    llamadas: list[int] = []

    def fit_resample(self, X, y):
        _SmoteEspia.llamadas.append(len(X))
        return super().fit_resample(X, y)


def test_smote_solo_en_el_entrenamiento_de_cada_particion(foto):
    X, y = a.separar(foto)
    modelo = a.armar_pipeline(get_models()["logreg"].set_params(class_weight=None), muestreo="smote")
    modelo.set_params(sampler=_SmoteEspia(random_state=0))
    _SmoteEspia.llamadas = []
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    cross_validate(modelo, X, y, cv=cv, scoring="average_precision")
    tamanos_entrenamiento = [len(e) for e, _ in cv.split(X, y)]
    assert sorted(_SmoteEspia.llamadas) == sorted(tamanos_entrenamiento)  # una vez por partición, solo con train
    modelo.fit(X, y)
    _SmoteEspia.llamadas = []
    assert len(modelo.predict_proba(X)) == len(X) and _SmoteEspia.llamadas == []  # predecir no re-muestrea


# ---------------------------------------------------------------- métricas


def test_metricas_de_clases_desbalanceadas():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.1, 0.2, 0.9, 0.8, 0.4, 0.7])
    m = compute_metrics(y, p, threshold=0.5)  # alerta: 0.6 (FP), 0.9, 0.8, 0.7 (TP); 0.4 (FN)
    assert m["specificity"] == pytest.approx(5 / 6) and m["recall"] == pytest.approx(3 / 4)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 3 / 4) / 2)
    assert m["f2"] == pytest.approx(5 * 0.75 * 0.75 / (4 * 0.75 + 0.75))
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert precision_at_k(y, p, 0.3) == 1.0  # los 3 de mayor puntaje se retiran
    assert np.isnan(compute_metrics(y, p * 4 - 2)["brier"])  # una distancia (Ridge) no es una probabilidad


def test_calibracion_y_metricas_del_campeon(foto):
    elegido = c.elegir(foto, folds=3, repeticiones=1, honestos=False)
    m = elegido.candidatos[elegido.nombre].metricas
    assert {"f2", "specificity", "balanced_accuracy", "precision_at_10pct", "brier"} <= m.keys()
    figura = c.figura_calibracion(elegido)
    if elegido.metodo == "predict_proba":
        assert figura is not None and "Brier" in figura.data[1].name
    ridge = c.elegir(foto, candidatos={"ridge": a.modelo_ridge()}, folds=3, repeticiones=1, honestos=False)
    assert c.figura_calibracion(ridge) is None and np.isnan(ridge.candidatos["ridge"].metricas["brier"])


def test_reproducible(foto):
    uno, dos = (c.elegir(foto, folds=3, repeticiones=1, honestos=False) for _ in range(2))
    assert uno.nombre == dos.nombre
    tabla_uno, tabla_dos = uno.tabla(), dos.tabla()
    pd.testing.assert_frame_equal(tabla_uno, tabla_dos)


# ---------------------------------------------------------------- MLflow: orígenes separados y trazabilidad


def test_experimentos_y_modelo_separados_por_origen(sin_origen, monkeypatch):
    assert r.con_origen("abandono") == "abandono" and r.modelo_registrado() == "abandono"
    monkeypatch.setenv("ORIGEN_DATOS", "real_anonimizado")
    assert r.con_origen("abandono") == "abandono_real" and r.modelo_registrado() == "abandono_real"
    assert r.con_origen("salario") == "salario_real"
    contexto = r.contexto_reproducible()
    assert contexto["origen_datos"] == "real_anonimizado" and contexto["experimental"] == "false"
    assert {"version.python", "version.scikit-learn", "version.pandas", "version.mlflow"} <= contexto.keys()


@pytest.mark.integration
def test_corrida_con_contexto_reproducible(tmp_path, monkeypatch, sin_origen):
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    r.configurar("prueba_auditoria")
    with r.corrida("padre") as padre, r.corrida("hija", anidada=True) as hija:
        pass
    tags_padre, tags_hija = mlflow.get_run(padre).data.tags, mlflow.get_run(hija).data.tags
    assert tags_padre["origen_datos"] == "sintetico" and tags_padre["experimental"] == "true"
    assert "version.scikit-learn" in tags_padre and "version.scikit-learn" not in tags_hija
    assert mlflow.get_experiment(mlflow.get_run(padre).info.experiment_id).name == "prueba_auditoria"
