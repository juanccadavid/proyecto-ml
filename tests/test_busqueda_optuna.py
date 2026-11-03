"""Búsqueda con Optuna de Random Forest y HistGB: espacios válidos, poda, reproducibilidad y el mejor modelo."""

import optuna
import pytest
from sklearn.base import clone

from proyecto_ml import abandono as a
from proyecto_ml import busqueda_optuna as bo
from proyecto_ml.generador import GeneradorAsociados

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(scope="module")
def datos():
    df = GeneradorAsociados(n=150, seed=1).generar()
    return a.preparar(df), df[a.TARGET]


@pytest.mark.parametrize("nombre", bo.MODELOS)
def test_cada_espacio_existe_en_su_pipeline(nombre):
    """Cada hiperparámetro sugerido es un parámetro real del Pipeline (set_params falla si no)."""
    for semilla in range(5):
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=semilla)).ask()
        clone(bo.modelo_base(nombre)).set_params(**bo.ESPACIOS[nombre](trial))


def test_modelo_base_igual_al_de_abandono(datos):
    """Sin afinar, el modelo base predice lo mismo que el de make abandono."""
    X, y = datos
    for nombre in bo.MODELOS:
        base = clone(bo.modelo_base(nombre)).fit(X, y)
        original = clone(a.modelos_supervisados()[nombre]).fit(X, y)
        assert (base.predict_proba(X) == original.predict_proba(X)).all()


@pytest.mark.integration
def test_busqueda_reproducible_y_mejor_modelo(datos):
    X, y = datos
    estudios = [bo.afinar_optuna("hist_gb", X, y, n_trials=8, folds=3, seed=0) for _ in range(2)]
    assert estudios[0].best_params == estudios[1].best_params  # misma semilla, mismos intentos
    estudio = estudios[0]
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert all(len(t.user_attrs["folds"]) == 3 for t in completos)

    mejor = bo.mejor_modelo(estudio, "hist_gb")
    assert mejor.get_params()["model__learning_rate"] == estudio.best_params["learning_rate"]
    nuevos = GeneradorAsociados(n=300, seed=2).generar()
    assert a.metricas_supervisado(mejor.fit(X, y), nuevos)["roc_auc"] > 0.8

    tabla = bo.resultados_estudio(estudio, top=3)
    assert len(tabla) == min(3, len(completos))
    assert tabla["AUC"].is_monotonic_decreasing
    assert "plotly" in bo.pagina_estudio(estudio, "prueba")


def test_poda_corta_intentos_malos(datos):
    """Un intento que reporta peor que la mediana de los anteriores se corta antes de la última partición."""
    X, y = datos
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0),
    )
    buenos = {"model__max_iter": 50, "model__min_samples_leaf": 5, "model__learning_rate": 0.2}
    malos = {**buenos, "model__min_samples_leaf": 100}  # ~100 filas por partición: ningún corte, AUC = 0.5
    for params in (buenos, buenos, malos):
        estudio.optimize(bo.objetivo(bo.modelo_base("hist_gb"), lambda _t, p=params: p, X, y, folds=3), n_trials=1)
    assert estudio.trials[-1].state == optuna.trial.TrialState.PRUNED
    assert len(estudio.trials[-1].user_attrs["folds"]) < 3
