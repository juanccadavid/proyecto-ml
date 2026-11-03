"""Compara modelos de abandono con PyCaret (AutoML) contra los pipelines hechos a mano.

PyCaret arma solo el preprocesamiento (imputa vacíos, codifica categóricas, escala) y prueba
varios algoritmos con validación cruzada: no hay que escribir ColumnTransformer ni Pipeline.
Recibe las mismas variables que `make abandono` (a.preparar: sin fuga ni datos personales) y
todos los modelos se validan con los mismos asociados nuevos.

Requiere el extra (PyCaret 4.0 alfa + LightGBM): uv sync --extra pycaret

Uso:
    uv run --extra pycaret python scripts/comparar_pycaret.py
    uv run --extra pycaret python scripts/comparar_pycaret.py --nuevos 5000 --ordenar Recall

Diferencias con nuestro preprocesador (features.py) que conviene conocer:
- Categóricas con OrdinalEncoder (un número por categoría), no OneHot.
- Vacíos numéricos con la media (nosotros: mediana).
- Sin class_weight: no compensa que los retiros sean minoría; con umbral 0.5 suele bajar el recall.
- PyCaret 4.0 cambió la API de la 3.x: ya no hay setup(), se usa ClassificationExperiment().fit().
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml.config import METRICS_DIR, MODELS_DIR, get_settings
from proyecto_ml.data import load_dataset
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import GeneradorAsociados

# Ids de PyCaret. Bagging: rf, et. Boosting: ada, gbc, lightgbm.
MODELOS = ["lr", "rf", "et", "ada", "gbc", "lightgbm"]


def experimento(
    train: pd.DataFrame, folds: int = 5, ordenar: str = "AUC", top: int = 3
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Tabla de PyCaret (validación cruzada) y los ``top`` mejores, reentrenados con todo ``train``.

    Cada modelo devuelto es un Pipeline de scikit-learn (preprocesamiento + modelo) que recibe
    ``a.preparar(df)`` tal cual, igual que los de ``a.modelos_supervisados()``.
    """
    from lightgbm import LGBMClassifier
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment(target=a.TARGET, session_id=42, fold=folds, normalize=True)
    exp.fit(a.preparar(train), train[a.TARGET])
    incluir = [LGBMClassifier(verbose=-1, random_state=42) if m == "lightgbm" else m for m in MODELOS]
    resultado = exp.compare_models(include=incluir, sort=ordenar, n_select=top)
    ids = ["lightgbm" if i == "LGBMClassifier" else i for i in resultado.ranked_ids]  # instancia propia -> su id
    tabla = resultado.leaderboard.assign(Model=ids).set_index("Model")
    mejores = {ids[i]: exp.finalize_model(m).pipeline for i, m in enumerate(resultado.models)}
    return tabla, mejores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datos", type=Path, default=get_settings().datos_asociados, help="Por defecto DATOS_ASOCIADOS del .env"
    )
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="La misma que usa make abandono")
    parser.add_argument("--folds", type=int, default=5, help="Particiones de la validación cruzada")
    parser.add_argument("--ordenar", default="AUC", help="Métrica de PyCaret para ordenar: AUC, Recall, F1...")
    parser.add_argument("--top", type=int, default=3, help="Mejores modelos de PyCaret a validar con los nuevos")
    args = parser.parse_args()

    try:
        import pycaret  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyCaret no está instalado. Ejecuta: uv sync --extra pycaret") from exc
    warnings.filterwarnings("ignore", category=UserWarning)  # avisos internos de la versión alfa

    train = load_dataset(args.datos)
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)\n")

    tabla, mejores = experimento(train, args.folds, args.ordenar, args.top)
    print(f"=== PyCaret: validación cruzada ({args.folds}) sobre el 70 % de entrenamiento, por {args.ordenar} ===")
    print(tabla.round(3).to_string())

    # Mismos asociados nuevos para todos: los `top` de PyCaret y nuestros pipelines.
    X, y = a.preparar(train), train[a.TARGET]
    Xn, yn = a.preparar(nuevos), nuevos[a.TARGET]
    comparacion = {f"pycaret_{n}": compute_metrics(yn, m.predict_proba(Xn)[:, 1]) for n, m in mejores.items()}
    for nombre, modelo in a.modelos_supervisados().items():
        comparacion[nombre] = a.metricas_supervisado(modelo.fit(X, y), nuevos)
    print(f"\n=== Comparación sobre nuevos (n={args.nuevos}): PyCaret vs. a mano ===")
    print(pd.DataFrame(comparacion).round(3).to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    mejor = next(iter(mejores))
    joblib.dump(mejores[mejor], MODELS_DIR / "abandono_pycaret.joblib")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"pycaret_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "ordenar": args.ordenar,
        "validacion_cruzada": tabla.to_dict(orient="index"),
        f"nuevos (n={args.nuevos})": comparacion,
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nMejor de PyCaret: {mejor} -> {MODELS_DIR / 'abandono_pycaret.joblib'} | métricas -> {salida}")


if __name__ == "__main__":
    main()
