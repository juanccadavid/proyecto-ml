"""Búsqueda de hiperparámetros con Optuna para los árboles del abandono: Random Forest y HistGradientBoosting.

A diferencia de la grilla (``evaluate.afinar``), Optuna no prueba todas las combinaciones: busca en
rangos y, con el muestreador TPE, propone cada intento cerca de los que mejor van. Cada intento se
evalúa con las mismas particiones (``StratifiedKFold``) y métricas (``SCORING``) que la tabla comparativa.

Poda: después de cada partición se informa el promedio parcial. Si un intento va peor que la mediana
de los anteriores en esa misma partición, se corta (``MedianPruner``) y no se gastan las demás.

Ojo, como en la grilla: el mejor puntaje del estudio es optimista (se eligió mirando esas particiones).
La prueba honesta son asociados nuevos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import SCORING
from proyecto_ml.models import get_models

if TYPE_CHECKING:
    from collections.abc import Callable

MODELOS = ("random_forest", "hist_gb")


def modelo_base(nombre: str, random_state: int = 42) -> Pipeline:
    """El modelo de ``a.modelos_supervisados()`` (mismos valores por defecto), con el preprocesamiento
    como ``Preprocesador`` para que sus opciones entren en la búsqueda."""
    return a.armar_pipeline(clone(get_models(random_state)[nombre]), buscable=True, random_state=random_state)


def _preprocesamiento(trial: optuna.Trial) -> dict:
    return {
        "preprocess__numeric_strategy": trial.suggest_categorical("numeric_strategy", ["median", "mean"]),
        "preprocess__encoding": trial.suggest_categorical("encoding", ["onehot", "ordinal"]),
        "preprocess__clip_outliers": trial.suggest_categorical("clip_outliers", [None, "iqr"]),
    }


def espacio_random_forest(trial: optuna.Trial) -> dict:
    """Bagging: más árboles estabilizan; hojas más grandes y menos variables por corte lo simplifican."""
    return {
        **_preprocesamiento(trial),
        "model__n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "model__max_depth": trial.suggest_int("max_depth", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "model__max_features": trial.suggest_float("max_features", 0.1, 1.0),  # fracción de variables por corte
        "model__class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
    }


def espacio_hist_gb(trial: optuna.Trial) -> dict:
    """Boosting. ``min_samples_leaf`` importa mucho con pocos datos: el valor por defecto (20) casi no
    deja cortar cuando cada partición tiene ~40 asociados."""
    return {
        **_preprocesamiento(trial),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "model__max_iter": trial.suggest_int("max_iter", 50, 500, step=50),
        "model__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
        "model__max_depth": trial.suggest_int("max_depth", 2, 10),
        "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "model__l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "model__class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


ESPACIOS: dict[str, Callable[[optuna.Trial], dict]] = {
    "random_forest": espacio_random_forest,
    "hist_gb": espacio_hist_gb,
}


def objetivo(modelo: Pipeline, espacio, X, y, folds: int = 5, metrica: str = "AUC", random_state: int = 0):
    """La función que Optuna maximiza: promedio de ``metrica`` en la validación cruzada, con poda por partición.

    Los parámetros del Pipeline quedan en ``trial.user_attrs["params"]`` y el puntaje de cada partición en
    ``trial.user_attrs["folds"]``, para reconstruir el mejor modelo y comparar intentos.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    particiones = list(cv.split(X, y))

    def _objetivo(trial: optuna.Trial) -> float:
        params = espacio(trial)
        trial.set_user_attr("params", params)
        puntajes = []
        for paso, (entrena, prueba) in enumerate(particiones):
            m = clone(modelo).set_params(**params).fit(X.iloc[entrena], y.iloc[entrena])
            puntajes.append(float(check_scoring(m, SCORING[metrica])(m, X.iloc[prueba], y.iloc[prueba])))
            trial.report(float(np.mean(puntajes)), paso)
            if trial.should_prune():
                trial.set_user_attr("folds", puntajes)
                raise optuna.TrialPruned()
        trial.set_user_attr("folds", puntajes)
        return float(np.mean(puntajes))

    return _objetivo


def afinar_optuna(
    nombre: str,
    X,
    y,
    n_trials: int = 60,
    folds: int = 5,
    metrica: str = "AUC",
    seed: int = 42,
    timeout: float | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """Crea (o continúa, si ``storage`` ya tiene el estudio ``abandono_<nombre>``) y corre la búsqueda.

    ``seed`` fija el muestreador TPE: mismos datos y semilla, mismos intentos. ``storage`` (ej.
    ``"sqlite:///optuna.db"``) guarda el estudio para seguirlo después o verlo con optuna-dashboard.
    """
    estudio = optuna.create_study(
        study_name=f"abandono_{nombre}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),  # deja ver 2 particiones
        storage=storage,
        load_if_exists=storage is not None,
    )
    estudio.set_user_attr("metrica", metrica)
    estudio.optimize(objetivo(modelo_base(nombre), ESPACIOS[nombre], X, y, folds, metrica), n_trials, timeout)
    return estudio


def mejor_modelo(estudio: optuna.Study, nombre: str) -> Pipeline:
    """El modelo base con los hiperparámetros del mejor intento, sin entrenar."""
    return clone(modelo_base(nombre)).set_params(**estudio.best_trial.user_attrs["params"])


def resumen_estudio(estudio: optuna.Study) -> dict[str, int]:
    estados = pd.Series([t.state.name for t in estudio.trials]).value_counts()
    return {"intentos": len(estudio.trials), **{e.lower(): int(n) for e, n in estados.items()}}


def resultados_estudio(estudio: optuna.Study, top: int = 5) -> pd.DataFrame:
    """Los ``top`` intentos completos: puntaje promedio, desviación entre particiones y sus hiperparámetros."""
    metrica = estudio.user_attrs.get("metrica", "valor")
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    filas = [
        {
            "intento": t.number,
            metrica: t.value,
            "std": float(np.std(t.user_attrs["folds"])),
            **{k: v for k, v in t.params.items()},
        }
        for t in sorted(completos, key=lambda t: t.value, reverse=True)[:top]
    ]
    return pd.DataFrame(filas).set_index("intento")


def pagina_estudio(estudio: optuna.Study, titulo: str) -> str:
    """HTML con los gráficos interactivos de Optuna (Plotly): historia, importancia, cortes y poda."""
    from optuna import visualization as v

    figuras = {
        "Historia de la búsqueda": v.plot_optimization_history(estudio),
        "Importancia de cada hiperparámetro (fANOVA)": v.plot_param_importances(estudio),
        "Puntaje según el valor de cada hiperparámetro": v.plot_slice(estudio),
        "Promedio parcial por partición (los intentos cortados no llegan al final)": v.plot_intermediate_values(
            estudio
        ),
    }
    partes = [f"<h1>{titulo}</h1>"]
    for i, (subtitulo, fig) in enumerate(figuras.items()):
        fig.update_layout(title=subtitulo, template="plotly_white")
        partes.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    cuerpo = "\n".join(partes)
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f"<title>{titulo}</title><style>body{{font-family:sans-serif;margin:24px}}</style></head>"
        f"<body>{cuerpo}</body></html>"
    )
