"""Compara modelos de abandono con PyCaret (AutoML) contra los pipelines hechos a mano.

PyCaret arma solo el preprocesamiento (imputa vacíos, codifica categóricas, escala) y prueba
varios algoritmos con validación cruzada: no hay que escribir ColumnTransformer ni Pipeline.
Recibe las mismas variables que `make abandono` (a.preparar: sin fuga ni datos personales) y
todos los modelos se validan con los mismos asociados nuevos.

La tabla comparativa (como la de compare_models, con TT = tiempo de entrenamiento) junta los
modelos de PyCaret y los nuestros, evaluados con las mismas particiones. PyCaret 4.0 ya no
entrega la columna de tiempo, por eso la tabla se calcula con evaluate.tabla_comparativa.

Requiere el extra (PyCaret 4.0 alfa + LightGBM): uv sync --extra pycaret

Uso:
    uv run --extra pycaret python scripts/comparar_pycaret.py
    uv run --extra pycaret python scripts/comparar_pycaret.py --nuevos 5000 --ordenar Recall
    uv run --extra pycaret python scripts/comparar_pycaret.py --ordenar "TT (Sec)"

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
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, get_settings
from proyecto_ml.evaluate import COLUMNAS_COMPARATIVA, compute_metrics, resaltar, tabla_comparativa
from proyecto_ml.generador import GeneradorAsociados

# Ids de PyCaret. Bagging: rf, et. Boosting: ada, gbc, lightgbm.
MODELOS = ["lr", "rf", "et", "ada", "gbc", "lightgbm"]


def experimento(train: pd.DataFrame, folds: int = 5) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Tabla propia de PyCaret (validación cruzada sobre su 70 %) y sus modelos reentrenados con todo ``train``.

    Los modelos vienen en el orden de PyCaret (por AUC). Cada uno es un Pipeline de scikit-learn
    (preprocesamiento + modelo) que recibe ``a.preparar(df)`` tal cual, igual que los de
    ``a.modelos_supervisados()``.
    """
    from lightgbm import LGBMClassifier
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment(target=a.TARGET, session_id=42, fold=folds, normalize=True)
    exp.fit(a.preparar(train), train[a.TARGET])
    incluir = [LGBMClassifier(verbose=-1, random_state=42) if m == "lightgbm" else m for m in MODELOS]
    resultado = exp.compare_models(include=incluir, sort="AUC", n_select=len(incluir))
    ids = ["lightgbm" if i == "LGBMClassifier" else i for i in resultado.ranked_ids]  # instancia propia -> su id
    tabla = resultado.leaderboard.assign(Model=ids).set_index("Model")
    modelos = {ids[i]: exp.finalize_model(m).pipeline for i, m in enumerate(resultado.models)}
    return tabla, modelos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datos", type=Path, default=get_settings().datos_asociados, help="Por defecto DATOS_ASOCIADOS del .env"
    )
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="La misma que usa make abandono")
    parser.add_argument("--folds", type=int, default=5, help="Particiones de la validación cruzada")
    parser.add_argument(
        "--ordenar", default="AUC", choices=COLUMNAS_COMPARATIVA, help="Columna para ordenar la tabla comparativa"
    )
    args = parser.parse_args()

    try:
        import pycaret  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyCaret no está instalado. Ejecuta: uv sync --extra pycaret") from exc
    warnings.filterwarnings("ignore", category=UserWarning)  # avisos internos de la versión alfa

    train = a.cargar(args.datos)  # con limpiar_extracto
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)\n")

    tabla_pycaret, de_pycaret = experimento(train, args.folds)

    # Tabla comparativa: PyCaret y a mano, con las mismas particiones sobre todo `train`.
    X, y = a.preparar(train), train[a.TARGET]
    modelos = {f"pycaret_{n}": m for n, m in de_pycaret.items()} | a.modelos_supervisados()
    comparativa = tabla_comparativa(modelos, X, y, folds=args.folds, ordenar=args.ordenar)
    print(f"=== Tabla comparativa: validación cruzada ({args.folds}), mismas particiones, por {args.ordenar} ===")
    print(comparativa.round(4).to_string())

    # Los mismos asociados nuevos para todos.
    Xn, yn = a.preparar(nuevos), nuevos[a.TARGET]
    en_nuevos = {}
    for nombre, modelo in modelos.items():
        if nombre.startswith("pycaret_"):  # ya reentrenados por finalize_model
            en_nuevos[nombre] = compute_metrics(yn, modelo.predict_proba(Xn)[:, 1])
        else:
            en_nuevos[nombre] = a.metricas_supervisado(modelo.fit(X, y), nuevos)
    en_nuevos = pd.DataFrame(en_nuevos).T.loc[comparativa.index]  # mismo orden que la tabla
    print(f"\n=== Sobre asociados nuevos (n={args.nuevos}) ===")
    print(en_nuevos.round(3).to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mejor = next(n for n in comparativa.index if n.startswith("pycaret_"))
    joblib.dump(modelos[mejor], MODELS_DIR / "abandono_pycaret.joblib")
    tabla_html = FIGURES_DIR / "pycaret_comparativa.html"  # con el mejor de cada columna en amarillo
    tabla_html.write_text(resaltar(comparativa).to_html(), encoding="utf-8", newline="\n")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"pycaret_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "ordenar": args.ordenar,
        "comparativa": comparativa.to_dict(orient="index"),
        "pycaret_leaderboard": tabla_pycaret.to_dict(orient="index"),
        f"nuevos (n={args.nuevos})": en_nuevos.to_dict(orient="index"),
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nMejor de PyCaret: {mejor} -> {MODELS_DIR / 'abandono_pycaret.joblib'}")
    print(f"Métricas -> {salida} | tabla comparativa -> {tabla_html}")


if __name__ == "__main__":
    main()
