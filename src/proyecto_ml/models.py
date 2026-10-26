"""Catálogo de modelos y construcción del pipeline completo."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def get_models(random_state: int = 42) -> dict:
    """Modelos base. Todos manejan desbalance vía class_weight."""
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gb": HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state),
    }


def get_tabpfn(random_state: int = 42):
    """TabPFN (Hugging Face: Prior-Labs). Opcional: ``uv sync --extra hf``.

    Ojo: recomendado hasta ~50k filas; licencia 2.5 NO comercial; usar GPU (Colab).
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:  # pragma: no cover - depende de extra opcional
        raise ImportError("TabPFN no instalado. Ejecuta: uv sync --extra hf") from exc
    return TabPFNClassifier(random_state=random_state)


def build_pipeline(preprocessor, model, sampler=None):
    """Une preprocesador + (opcional) re-muestreo + modelo.

    El sampler (ej. SMOTE) solo se aplica en ``fit``: por eso se usa el Pipeline de
    imblearn, que nunca re-muestrea los datos de evaluación.
    """
    # Aplana el preprocesador si es un Pipeline (ej. ColumnTransformer + PCA):
    # imblearn no admite Pipelines anidados como pasos intermedios.
    is_pipeline = isinstance(preprocessor, Pipeline)
    pre_steps = list(preprocessor.steps) if is_pipeline else [("preprocess", preprocessor)]

    if sampler is None:
        return Pipeline([*pre_steps, ("model", model)])
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline([*pre_steps, ("sampler", sampler), ("model", model)])
