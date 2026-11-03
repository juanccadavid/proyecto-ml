"""Evaluación del campeón (ROC, matriz de confusión, MCC/Kappa), estrategias de CV, regresión y SHAP."""

import numpy as np
import optuna
import pytest
from sklearn.metrics import average_precision_score, cohen_kappa_score, matthews_corrcoef

from proyecto_ml import abandono as a
from proyecto_ml import busqueda_optuna as bo
from proyecto_ml import campeon as c
from proyecto_ml import explicar as e
from proyecto_ml import regresion as g
from proyecto_ml import validacion as v
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import FALTANTES, GeneradorAsociados


@pytest.fixture(scope="module")
def datos():
    entrenamiento = GeneradorAsociados(n=300, seed=6, tasa_retiro=0.25).generar()
    nuevos = GeneradorAsociados(n=60, seed=7, tasa_retiro=0.25, faltantes=FALTANTES).generar()
    return entrenamiento, nuevos


@pytest.fixture(scope="module")
def campeon(datos):
    return c.elegir(datos[0], folds=3, repeticiones=1)


# ---------------------------------------------------------------- campeón: MCC, Kappa, confusión y ROC


def test_mcc_y_kappa_al_umbral_de_alerta(campeon):
    for candidato in campeon.candidatos.values():
        alerta = (candidato.promedio >= candidato.umbrales["umbral"]).astype(int)
        assert candidato.metricas["mcc"] == pytest.approx(matthews_corrcoef(campeon.y, alerta))
        assert candidato.metricas["kappa"] == pytest.approx(cohen_kappa_score(campeon.y, alerta))
    m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["mcc"] == pytest.approx(1.0) and m["kappa"] == pytest.approx(1.0)


def test_matriz_de_confusion_cuadra_con_precision_y_recall(campeon):
    matrices = c.matrices_confusion(campeon)
    (tn, fp), (fn, tp) = matrices["alerta"]
    metricas = campeon.candidatos[campeon.nombre].metricas
    assert tn + fp + fn + tp == len(campeon.y) and tp + fn == campeon.y.sum()
    assert tp / (tp + fp) == pytest.approx(metricas["precision"])
    assert tp / (tp + fn) == pytest.approx(metricas["recall"])
    if campeon.umbral_alto is not None:  # el umbral alto alerta a menos: nunca más alertas que el de alerta
        assert matrices["se retira"][:, 1].sum() <= matrices["alerta"][:, 1].sum()
    assert c.resumen(campeon)["confusion_cv"]["alerta"] == {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def test_figuras_roc_y_confusion(campeon):
    roc = c.figura_roc(campeon)
    nombres = [t.name for t in roc.data]
    assert sum("AUC" in n for n in nombres) == len(campeon.candidatos) + 1  # cada candidato + el azar
    assert any("campeón" in n for n in nombres) and "umbral alerta (zona gris)" in nombres
    confusion = c.figura_confusion(campeon)
    assert len(confusion.data) == len(c.matrices_confusion(campeon))


# ---------------------------------------------------------------- KFold vs StratifiedKFold vs TimeSeriesSplit


def test_comparar_validaciones(datos):
    df = a.foto(GeneradorAsociados(n=600, seed=3, tasa_retiro=0.2).generar(), 3)
    tabla = v.comparar_validaciones(a.modelos_supervisados()["logreg"], df, folds=4)
    assert tabla.groupby("estrategia").size().to_dict() == {"KFold": 4, "StratifiedKFold": 4, "TimeSeriesSplit": 4}
    estratificada = tabla[tabla["estrategia"] == "StratifiedKFold"]["retirados_prueba"]
    assert estratificada.max() - estratificada.min() <= 1  # misma tasa en cada partición
    temporal = tabla[tabla["estrategia"] == "TimeSeriesSplit"]["filas_entrenamiento"]
    assert temporal.is_monotonic_increasing and temporal.iloc[0] < temporal.iloc[-1]  # el pasado crece
    resumen = v.resumen_validaciones(tabla)
    assert list(resumen.index) == ["KFold", "StratifiedKFold", "TimeSeriesSplit"]
    assert len(v.figura_validaciones(tabla, "prueba").data) == 6  # puntos + promedio por estrategia


def test_time_series_split_prueba_con_los_mas_nuevos():
    df = GeneradorAsociados(n=200, seed=4).generar().sort_values(v.ORDEN_TEMPORAL).reset_index(drop=True)
    fechas = df[v.ORDEN_TEMPORAL]
    for entrena, prueba in v.estrategias_cv(4)["TimeSeriesSplit"].split(df):
        assert fechas.iloc[entrena].max() <= fechas.iloc[prueba].min()


# ---------------------------------------------------------------- regresión: MAE, MSE, RMSE, R²


def test_metricas_de_regresion():
    m = g.metricas([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    assert m["MAE"] == pytest.approx(0.5) and m["MSE"] == pytest.approx(1.0) and m["RMSE"] == pytest.approx(1.0)
    assert m["R2"] == pytest.approx(1 - 4 / 5)
    assert g.metricas([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])["R2"] == pytest.approx(0.0)  # el promedio: R² = 0


def test_regresion_sin_fuga_y_referencia(datos):
    _, nuevos = datos
    X, y = g.separar(nuevos)
    assert y.notna().all() and len(X) < len(nuevos)  # quien no reporta salario no entra a entrenar
    modelo = g.modelos()["ridge"].fit(X, y)
    columnas = modelo.named_steps["preprocess"].get_feature_names_out()
    assert not any(n.startswith(("SALARIO", "TIPOPACTO")) for n in columnas)  # ni el objetivo ni su fuga
    tabla = g.comparar(g.modelos(), X, y, folds=3)
    assert set(tabla.index) == set(g.TITULOS) and tabla.loc["media", "R2"] <= 0.01
    assert (tabla[g.ERRORES] > 0).all().all()  # errores en positivo (sklearn los da negativos)


# ---------------------------------------------------------------- Optuna: hiperparámetros comunes del bosque


def test_espacio_random_forest_con_los_hiperparametros_comunes():
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    espacio = bo.ESPACIOS["random_forest"](trial)
    comunes = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"]
    assert {f"model__{h}" for h in comunes} <= espacio.keys()
    assert trial.distributions["max_features"].choices == ("sqrt", "log2", None)
    assert None in trial.distributions["max_depth"].choices


# ---------------------------------------------------------------- SHAP


@pytest.fixture(scope="module")
def explicador(datos):
    entrenamiento, _ = datos
    modelo = c.ModeloZonas(a.modelos_supervisados()["logreg"].fit(*a.separar(entrenamiento)), 0.3, 0.7)
    return e.Explicador(modelo, a.separar(entrenamiento)[0], n_fondo=40)


def test_shap_suma_el_puntaje(datos, explicador):
    _, nuevos = datos
    filas = a.separar(nuevos)[0].head(4)
    contribuciones = explicador.contribuciones(filas)
    assert list(contribuciones.columns) == e.VARIABLES
    puntaje = a.puntaje(explicador.modelo, filas)
    assert np.allclose(explicador.base + contribuciones.sum(axis=1), puntaje)
    assert np.allclose(contribuciones, explicador.contribuciones(a.preparar(filas)))  # cruda o preparada


def test_shap_con_vacios_y_categorias_nuevas(datos, explicador):
    fila = a.preparar(datos[1]).head(1).copy()
    fila["CODCIUDAD"], fila["SEXO"], fila["SALARIO"] = "99999", np.nan, np.nan
    tabla = explicador.explicar(fila)
    assert explicador.base + tabla["contribucion"].sum() == pytest.approx(a.puntaje(explicador.modelo, fila)[0])
    assert tabla["contribucion"].abs().is_monotonic_decreasing
    assert tabla.set_index("variable").loc["CODCIUDAD", "valor"] == "99999"


def test_shap_ridge_y_figuras(datos):
    entrenamiento, nuevos = datos
    ridge = a.modelo_ridge().fit(*a.separar(entrenamiento))
    explicador = e.Explicador(ridge, entrenamiento, n_fondo=30)
    assert explicador.metodo == "decision_function"
    fila = a.separar(nuevos)[0].head(1)
    tabla = explicador.explicar(fila)
    assert explicador.base + tabla["contribucion"].sum() == pytest.approx(ridge.decision_function(fila)[0])

    cascada = e.figura_explicacion(tabla, explicador.base, explicador.metodo, top=5)
    assert len(cascada.data[0].y) == 5 + 3  # base + 5 variables + "otras" + este asociado
    importancia = e.importancia(explicador.contribuciones(a.separar(nuevos)[0].head(5)))
    assert importancia.is_monotonic_decreasing and set(importancia.index) == set(e.VARIABLES)
    assert len(e.figura_importancia(importancia, "decision_function", "prueba", top=15).data[0].y) == 15


# ---------------------------------------------------------------- incertidumbre del campeón


def test_comparar_particiones_con_la_correccion_de_nadeau_bengio():
    rng = np.random.default_rng(0)
    base = rng.normal(0.85, 0.05, (3, 5))
    iguales = c.comparar_particiones(base, base + rng.normal(0, 0.01, (3, 5)), folds=5)
    assert iguales["ic_inf"] < 0 < iguales["ic_sup"]  # ruido: no concluyente
    claro = c.comparar_particiones(base + 0.2 + rng.normal(0, 0.01, (3, 5)), base, folds=5)
    assert claro["ic_inf"] > 0 and claro["p_valor"] < 0.01 and claro["gana_en"] == 1
    # la corrección ensancha el intervalo frente a la t común (que supone particiones independientes)
    d = (base + rng.normal(0, 0.02, (3, 5)) - base).ravel()
    t_comun = 2.145 * d.std(ddof=1) / np.sqrt(d.size)  # t(0,975; 14 gl)
    corregido = c.comparar_particiones(base + d.reshape(3, 5), base, folds=5)
    assert corregido["ic_sup"] - corregido["diferencia"] > t_comun


def test_bootstrap_y_variacion_entre_particiones(campeon):
    cand = campeon.candidatos[campeon.nombre]
    m = cand.metricas
    assert m["pr_auc_ic_inf"] <= average_precision_score(campeon.y, cand.promedio) <= m["pr_auc_ic_sup"]
    assert cand.pr_auc_particion.shape == (1, 3)  # repeticiones x folds del fixture
    assert m["pr_auc_std"] == pytest.approx(np.std(cand.pr_auc_particion, ddof=1))
    bajo, alto = c.intervalo_bootstrap(campeon.y, cand.promedio, n=200, nivel=0.5)
    assert m["pr_auc_ic_inf"] <= bajo <= alto <= m["pr_auc_ic_sup"]  # el del 50 % queda dentro del del 95 %


def test_comparacion_empates_y_umbrales_honestos(campeon):
    tabla = campeon.comparacion()
    assert campeon.nombre not in tabla.index and set(tabla.index) == set(campeon.candidatos) - {campeon.nombre}
    propio = campeon.candidatos[campeon.nombre].pr_auc_particion
    for nombre, fila in tabla.iterrows():  # diferencia = promedio de (campeón − candidato) partición a partición
        assert fila["diferencia"] == pytest.approx((propio - campeon.candidatos[nombre].pr_auc_particion).mean())
    assert campeon.empatados() == list(tabla.index[~tabla["concluyente"].astype(bool)])
    hon = campeon.honestos
    assert {"precision", "recall", "precision_alto", "recall_alto", "alertas_alto"} <= hon.keys()
    assert 0 <= hon["recall_alto"] <= hon["recall"] <= 1  # el umbral alto alerta a menos
    info = c.resumen(campeon)
    assert info["umbrales_honestos"] == hon and info["empatados"] == campeon.empatados()
    assert set(info["comparacion"]) == set(tabla.index)


def test_lista_de_candidatos_abierta(datos):
    entrenamiento, nuevos = datos
    sin_vacios = c.lista_candidatos(a.separar(entrenamiento)[0])
    assert {"ridge", "logreg", "random_forest", "hist_gb", "logreg_sin_balancear", "logreg_target"} == set(sin_vacios)
    assert "logreg_indicador" in c.lista_candidatos(a.separar(nuevos)[0])  # nuevos trae vacíos (FALTANTES)
    assert set(c.lista_candidatos()) <= set(c.TITULOS)
    assert sin_vacios["logreg_sin_balancear"][-1].class_weight is None
    assert sin_vacios["logreg"][-1].class_weight == "balanced"


def test_resumen_del_ci(campeon):
    import importlib.util
    from pathlib import Path

    ruta = Path(__file__).parents[1] / "scripts" / "resumen_campeon.py"
    spec = importlib.util.spec_from_file_location("resumen_campeon", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    texto = modulo.resumen(c.resumen(campeon))
    assert texto.startswith(f"## Campeón: {c.TITULOS[campeon.nombre]}")
    assert "entre particiones" in texto and "IC 95 %" in texto and "validación anidada" in texto
    assert "ROC" not in texto  # el titular es la PR-AUC, no el ROC-AUC
