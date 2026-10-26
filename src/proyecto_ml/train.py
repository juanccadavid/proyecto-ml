"""Entrena y compara modelos, registra todo en MLflow y guarda el mejor.

Uso:
    uv run python -m proyecto_ml.train                       # usa variables del .env
    uv run python -m proyecto_ml.train --data data/processed/sintetico.parquet --pca 0.95
    uv run python -m proyecto_ml.train --tabpfn --smote
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from proyecto_ml.config import METRICS_DIR, MODELS_DIR, get_settings
from proyecto_ml.data import infer_column_types, load_dataset, split_data, split_features_target
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.features import build_preprocessor
from proyecto_ml.models import build_pipeline, get_models, get_tabpfn

SELECTION_METRIC = "pr_auc"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=Path, help="Ruta del dataset (por defecto DATA_PATH)")
    parser.add_argument("--target", help="Columna objetivo (por defecto TARGET_COLUMN)")
    parser.add_argument("--models", default="logreg,random_forest,hist_gb", help="Lista separada por comas")
    parser.add_argument("--pca", type=float, default=None, help="Componentes PCA (int) o varianza (0-1)")
    parser.add_argument("--smote", action="store_true", help="Aplicar SMOTE dentro del pipeline")
    parser.add_argument("--tabpfn", action="store_true", help="Incluir TabPFN (requiere extra hf)")
    parser.add_argument("--no-log-model", action="store_true", help="No guardar el modelo en MLflow")
    parser.add_argument("--output-dir", type=Path, default=None, help="Carpeta para métricas/modelo")
    return parser.parse_args(argv)


SKOPS_TRUSTED_TYPES = ["numpy.dtype"]


def _log_model(pipeline) -> None:
    """Registra el modelo. MLflow 3 serializa con skops (más seguro que pickle)."""
    try:
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES)
    except TypeError:  # MLflow 2.x no tiene `name` ni skops
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    except mlflow.exceptions.MlflowException as exc:  # tipo no confiable nuevo (ej. TabPFN)
        print(f"Aviso: skops rechazó el modelo ({exc}); se registra con cloudpickle.")
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")


def run(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    args = parse_args(argv)
    settings = get_settings()
    data_path = args.data or settings.data_path
    target = args.target or settings.target_column
    metrics_dir = args.output_dir or METRICS_DIR
    models_dir = args.output_dir or MODELS_DIR

    df = load_dataset(data_path)
    time = pd.to_datetime(df[settings.time_column]) if settings.time_column else None
    groups = df[settings.group_column] if settings.group_column else None
    X, y = split_features_target(df, target, drop=[settings.id_column, settings.time_column, settings.group_column])
    numeric, categorical = infer_column_types(X)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=settings.test_size, random_state=settings.random_state, time=time, groups=groups
    )

    pca = None
    if args.pca is not None:
        pca = int(args.pca) if args.pca >= 1 else args.pca

    catalog = get_models(settings.random_state)
    selected = {name: catalog[name] for name in args.models.split(",") if name}
    if args.tabpfn:
        selected["tabpfn"] = get_tabpfn(settings.random_state)

    sampler = None
    if args.smote:
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=settings.random_state)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    results: dict[str, dict[str, float]] = {}
    best_name, best_score, best_pipeline = None, float("-inf"), None
    for name, model in selected.items():
        preprocessor = build_preprocessor(numeric, categorical, pca_components=pca, random_state=settings.random_state)
        pipeline = build_pipeline(preprocessor, model, sampler)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model": name,
                    "dataset": Path(data_path).name,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "pca": pca,
                    "smote": args.smote,
                    "split": "temporal" if time is not None else "grupos" if groups is not None else "estratificado",
                }
            )
            pipeline.fit(X_train, y_train)
            scores = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, scores)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # omite NaN
            if not args.no_log_model:
                _log_model(pipeline)
        results[name] = metrics
        print(f"{name:>14}: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        if metrics[SELECTION_METRIC] > best_score:
            best_name, best_score, best_pipeline = name, metrics[SELECTION_METRIC], pipeline

    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (metrics_dir / f"metrics_{stamp}.json").write_text(
        json.dumps({"best_model": best_name, "selection_metric": SELECTION_METRIC, "results": results}, indent=2),
        encoding="utf-8",
    )
    joblib.dump(best_pipeline, models_dir / "best_model.joblib")
    metadata = {
        "model": best_name,
        "target": target,
        "numeric": numeric,
        "categorical": categorical,
        "numeric_defaults": {c: float(X_train[c].median()) for c in numeric},
        "categories": {c: sorted(X_train[c].dropna().astype(str).unique().tolist()) for c in categorical},
    }
    (models_dir / "best_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMejor modelo: {best_name} ({SELECTION_METRIC}={best_score:.3f})")
    return results


if __name__ == "__main__":
    run()
