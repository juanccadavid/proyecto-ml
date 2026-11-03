"""Afina los hiperparámetros de Ridge (nuestro) y de la regresión logística de PyCaret (pycaret_lr).

Búsqueda en grilla (GridSearchCV) con validación cruzada: mismas particiones y métricas que la
tabla comparativa, así que los números se pueden poner lado a lado. Después se compara cada
modelo por defecto contra su versión afinada con asociados NUEVOS, que es la prueba honesta:
el puntaje de la validación cruzada del mejor es optimista porque se eligió mirando esos datos.

Requiere el extra (PyCaret 4.0 alfa): uv sync --extra pycaret

Uso:
    uv run --extra pycaret python scripts/afinar_modelos.py
    uv run --extra pycaret python scripts/afinar_modelos.py --metrica F1 --nuevos 5000

Hiperparámetros:
- Ridge:     alpha (regularización, más alto = coeficientes más pequeños) y class_weight.
- Logística: C (inverso de la regularización, más alto = menos), l1_ratio (0 = L2, 1 = L1:
             L1 puede dejar variables en 0) y class_weight. Solver liblinear: admite L1 y L2.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, get_settings
from proyecto_ml.data import load_dataset
from proyecto_ml.evaluate import (
    SCORING,
    afinar,
    figura_busqueda,
    formato_params,
    resaltar,
    resultados_busqueda,
    tabla_comparativa,
)
from proyecto_ml.generador import GeneradorAsociados

REGULARIZACION = np.logspace(-3, 3, 13)  # 0.001 ... 1000, 13 valores igual de espaciados en escala log
GRILLAS = {
    "ridge": {"model__alpha": REGULARIZACION, "model__class_weight": [None, "balanced"]},
    "pycaret_lr": {
        "lr__C": REGULARIZACION,
        "lr__l1_ratio": [0.0, 1.0],
        "lr__class_weight": [None, "balanced"],
        "lr__solver": ["liblinear"],
    },
}
EJE_X = {"ridge": "alpha", "pycaret_lr": "C"}  # el hiperparámetro de regularización de cada uno
# Variables que se muestran en la terminal (el CSV lleva todas).
COLUMNAS_MUESTRA = ["EDAD", "SEXO", "TIPOCONTRATO", "#INTENCIONRETIROS_ULT3MESES", "#SOLICITUD_CANCELACIONAHORROS"]


def modelos_base(train: pd.DataFrame, folds: int = 5) -> dict[str, Pipeline]:
    """Los dos modelos con sus valores por defecto, sin entrenar.

    pycaret_lr es el Pipeline que arma PyCaret (su preprocesamiento + LogisticRegression), el mismo
    de ``make pycaret``.
    """
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment(target=a.TARGET, session_id=42, fold=folds, normalize=True)
    exp.fit(a.preparar(train), train[a.TARGET])
    return {"ridge": a.modelo_ridge(), "pycaret_lr": clone(exp.create_model("lr").pipeline)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datos", type=Path, default=get_settings().datos_asociados, help="Por defecto DATOS_ASOCIADOS del .env"
    )
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="La misma que usa make abandono")
    parser.add_argument("--folds", type=int, default=5, help="Particiones de la validación cruzada")
    parser.add_argument("--metrica", default="AUC", choices=list(SCORING), help="Métrica que elige al mejor")
    parser.add_argument("--muestra", type=int, default=5, help="Filas de la tabla de predicciones en la terminal")
    args = parser.parse_args()

    try:
        import pycaret  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyCaret no está instalado. Ejecuta: uv sync --extra pycaret") from exc
    warnings.filterwarnings("ignore", category=UserWarning)  # avisos internos de la versión alfa

    train = load_dataset(args.datos)
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()
    X, y = a.preparar(train), train[a.TARGET]
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)")
    print(f"Elige por: {args.metrica} (validación cruzada, {args.folds} particiones)\n")

    base = modelos_base(train, args.folds)
    busquedas, modelos = {}, {}
    for nombre, modelo in base.items():
        busqueda = afinar(modelo, GRILLAS[nombre], X, y, folds=args.folds, metrica=args.metrica)
        busquedas[nombre] = busqueda
        modelos[nombre], modelos[f"{nombre}_afinado"] = modelo, busqueda.best_estimator_
        print(f"=== {nombre}: {len(busqueda.cv_results_['params'])} combinaciones ===")
        print(f"Mejor: {formato_params(busqueda.best_params_)}")
        print(resultados_busqueda(busqueda).round(4).to_string() + "\n")

    comparativa = tabla_comparativa(modelos, X, y, folds=args.folds, ordenar=args.metrica)
    print("=== Por defecto vs. afinado: validación cruzada (los afinados salen optimistas) ===")
    print(comparativa.round(4).to_string())

    # La prueba honesta: asociados que ningún modelo vio (ni en la búsqueda).
    en_nuevos = {n: a.metricas_supervisado(clone(m).fit(X, y), nuevos) for n, m in modelos.items()}
    en_nuevos = pd.DataFrame(en_nuevos).T
    print(f"\n=== Por defecto vs. afinado sobre asociados nuevos (n={args.nuevos}) ===")
    print(en_nuevos.round(3).to_string())

    # Tabla de predicciones de los afinados (como predict_model de PyCaret), sobre los nuevos.
    afinados = {f"{n}_afinado": b.best_estimator_ for n, b in busquedas.items()}
    predicciones = a.tabla_predicciones(afinados, nuevos)
    columnas = [*COLUMNAS_MUESTRA, a.TARGET, *[c for c in predicciones if c.endswith(("_label", "_score"))]]
    print(f"\n=== Predicciones de los afinados: muestra de {args.muestra} asociados nuevos ===")
    print("score = riesgo de abandono: probabilidad en la logística; en Ridge, distancia a la frontera (>0 = abandona)")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(predicciones.sample(args.muestra, random_state=0)[columnas].round(4))
    etiquetas = predicciones[[f"{n}_label" for n in afinados]]
    discrepan = etiquetas.ne(etiquetas.iloc[:, 0], axis=0).any(axis=1).sum()
    print(f"Los afinados discrepan en {discrepan} de {len(predicciones)} asociados.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, busqueda in busquedas.items():
        joblib.dump(busqueda.best_estimator_, MODELS_DIR / f"abandono_{nombre}_afinado.joblib")
    tabla_csv = PROCESSED_DIR / "predicciones_afinados.csv"  # todas las filas y variables; ignorado por Git
    predicciones.to_csv(tabla_csv, index_label="fila", encoding="utf-8-sig")  # -sig: tildes bien en Excel
    grafico = FIGURES_DIR / "afinamiento_curvas.png"
    figura_busqueda(busquedas, EJE_X).savefig(grafico, dpi=150)
    tabla_html = FIGURES_DIR / "afinamiento_comparativa.html"  # con el mejor de cada columna en amarillo
    tabla_html.write_text(resaltar(comparativa).to_html(), encoding="utf-8", newline="\n")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"afinamiento_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "metrica": args.metrica,
        "mejores": {n: formato_params(b.best_params_) for n, b in busquedas.items()},
        "top5": {n: resultados_busqueda(b).to_dict(orient="records") for n, b in busquedas.items()},
        "comparativa": comparativa.to_dict(orient="index"),
        f"nuevos (n={args.nuevos})": en_nuevos.to_dict(orient="index"),
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nModelos -> {MODELS_DIR / 'abandono_*_afinado.joblib'} | métricas -> {salida}")
    print(f"Curvas -> {grafico} | tabla comparativa -> {tabla_html}")
    print(f"Predicciones ({len(predicciones)} asociados nuevos) -> {tabla_csv}")


if __name__ == "__main__":
    main()
