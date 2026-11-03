"""Afina con Optuna los árboles del abandono: Random Forest (bagging) y HistGradientBoosting (boosting).

Cada intento propone hiperparámetros del modelo y del preprocesamiento (vacíos con mediana o media,
OneHot u ordinal, con o sin recorte IQR) y se evalúa con validación cruzada: mismas particiones y
métricas que la tabla comparativa. Los intentos que van mal se cortan a mitad de camino (poda).
Después se compara cada modelo por defecto contra su versión afinada con asociados NUEVOS: el mejor
puntaje de la búsqueda es optimista porque se eligió mirando esas particiones.

Uso:
    uv run python scripts/afinar_optuna.py                          # 60 intentos por modelo, por AUC
    uv run python scripts/afinar_optuna.py --intentos 200 --metrica F1
    uv run python scripts/afinar_optuna.py --modelos hist_gb --timeout 120   # máximo 2 minutos
    uv run python scripts/afinar_optuna.py --storage sqlite:///optuna.db     # guarda y continúa el estudio

Hiperparámetros (rangos en busqueda_optuna.py):
- Random Forest: n_estimators, max_depth, min_samples_leaf, max_features (fracción) y class_weight.
- HistGB:        learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf,
                 l2_regularization y class_weight.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
from sklearn.base import clone

from proyecto_ml import abandono as a
from proyecto_ml import busqueda_optuna as bo
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, get_settings
from proyecto_ml.evaluate import SCORING, resaltar, tabla_comparativa
from proyecto_ml.features import aplanar
from proyecto_ml.generador import GeneradorAsociados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datos", type=Path, default=get_settings().datos_asociados, help="Por defecto DATOS_ASOCIADOS del .env"
    )
    parser.add_argument(
        "--modelos",
        type=lambda v: [m.strip() for m in v.split(",") if m.strip()],
        default=list(bo.MODELOS),
        help="Separados por comas: random_forest,hist_gb (por defecto los dos)",
    )
    parser.add_argument("--intentos", type=int, default=60, help="Intentos (trials) por modelo")
    parser.add_argument("--timeout", type=float, default=None, help="Segundos máximos por modelo (opcional)")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla del muestreador TPE")
    parser.add_argument("--storage", help="Base del estudio, ej. sqlite:///optuna.db (por defecto, en memoria)")
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="La misma que usa make abandono")
    parser.add_argument("--folds", type=int, default=5, help="Particiones de la validación cruzada")
    parser.add_argument("--metrica", default="AUC", choices=list(SCORING), help="Métrica que se maximiza")
    args = parser.parse_args()
    desconocidos = set(args.modelos) - set(bo.MODELOS)
    if desconocidos:
        parser.error(f"Modelos no válidos: {sorted(desconocidos)}. Opciones: {', '.join(bo.MODELOS)}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # sin una línea por intento

    train = a.cargar(args.datos)  # con limpiar_extracto
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()
    X, y = a.preparar(train), train[a.TARGET]
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)")
    print(f"Maximiza: {args.metrica} (validación cruzada, {args.folds} particiones) · {args.intentos} intentos\n")

    estudios, modelos = {}, {}
    for nombre in args.modelos:
        estudio = bo.afinar_optuna(
            nombre, X, y, args.intentos, args.folds, args.metrica, args.semilla, args.timeout, args.storage
        )
        estudios[nombre] = estudio
        modelos[nombre], modelos[f"{nombre}_optuna"] = bo.modelo_base(nombre), bo.mejor_modelo(estudio, nombre)
        resumen = bo.resumen_estudio(estudio)
        print(f"=== {a.TITULOS[nombre]}: {resumen} ===")
        print(f"Mejor intento #{estudio.best_trial.number}: {args.metrica}={estudio.best_value:.4f}")
        print(bo.resultados_estudio(estudio).round(4).to_string())
        importancia = optuna.importance.get_param_importances(estudio)
        print("Importancia de los hiperparámetros: " + ", ".join(f"{k} {v:.2f}" for k, v in importancia.items()))
        print()

    comparativa = tabla_comparativa(modelos, X, y, folds=args.folds, ordenar=args.metrica)
    print("=== Por defecto vs. Optuna: validación cruzada (los afinados salen optimistas) ===")
    print(comparativa.round(4).to_string())

    # La prueba honesta: asociados que la búsqueda nunca vio.
    entrenados = {n: clone(m).fit(X, y) for n, m in modelos.items()}
    en_nuevos = pd.DataFrame({n: a.metricas_supervisado(m, nuevos) for n, m in entrenados.items()}).T
    print(f"\n=== Por defecto vs. Optuna sobre asociados nuevos (n={args.nuevos}) ===")
    print(en_nuevos.round(3).to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paginas = []
    for nombre, estudio in estudios.items():
        a.guardar_modelo(aplanar(entrenados[f"{nombre}_optuna"]), MODELS_DIR, f"abandono_{nombre}_optuna")
        pagina = FIGURES_DIR / f"optuna_{nombre}.html"  # interactivo: historia, importancia, cortes y poda
        pagina.write_text(bo.pagina_estudio(estudio, f"Optuna - {a.TITULOS[nombre]}"), encoding="utf-8")
        paginas.append(pagina)
    tabla_html = FIGURES_DIR / "optuna_comparativa.html"  # con el mejor de cada columna en amarillo
    tabla_html.write_text(resaltar(comparativa).to_html(), encoding="utf-8", newline="\n")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"optuna_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "metrica": args.metrica,
        "intentos": args.intentos,
        "semilla": args.semilla,
        "estudios": {
            n: {
                **bo.resumen_estudio(e),
                "mejor_valor": e.best_value,
                "mejores": e.best_params,
                "importancia": optuna.importance.get_param_importances(e),
                "top5": bo.resultados_estudio(e).reset_index().to_dict(orient="records"),
            }
            for n, e in estudios.items()
        },
        "comparativa": comparativa.to_dict(orient="index"),
        f"nuevos (n={args.nuevos})": en_nuevos.to_dict(orient="index"),
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nModelos (pipeline, _preprocesamiento y _modelo) -> {MODELS_DIR / 'abandono_*_optuna*.joblib'}")
    print(f"Métricas -> {salida} | tabla comparativa -> {tabla_html}")
    print("Gráficos interactivos de cada estudio -> " + " | ".join(str(p) for p in paginas))


if __name__ == "__main__":
    main()
