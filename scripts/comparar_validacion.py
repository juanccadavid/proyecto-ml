"""Compara KFold, StratifiedKFold y TimeSeriesSplit en el abandono (make validacion).

El mismo modelo (por defecto la logística) y los mismos datos que el campeón (la foto del asociado), partidos
de tres formas. Por partición: filas de entrenamiento y prueba, retirados en la prueba y PR-AUC.

- **KFold** baraja y parte al azar: con pocos retirados, unas pruebas tienen el doble que otras y la PR-AUC
  salta más entre particiones.
- **StratifiedKFold** deja la misma tasa de retiro en cada partición: es la que usa el campeón.
- **TimeSeriesSplit** (ordenado por fecha de ingreso) entrena con los más antiguos y prueba con los más
  nuevos: el entrenamiento crece de partición en partición y las primeras tienen pocos datos.

Guarda reports/figures/validacion_cruzada.html y reports/metrics/validacion_cruzada_<fecha>.json; en MLflow
(experimento ``abandono``), una corrida "validacion_cruzada".

Uso:
    uv run python scripts/comparar_validacion.py
    uv run python scripts/comparar_validacion.py --modelo random_forest --folds 10 --sin-mlflow
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import registro as r
from proyecto_ml import validacion as v
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (2.000, semilla 7, 10 %%)")
    parser.add_argument(
        "--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses de la foto (0 = todos al corte)"
    )
    parser.add_argument("--modelo", choices=list(a.modelos_supervisados()), default="logreg")
    parser.add_argument("--folds", type=int, default=c.FOLDS)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos)
    h = args.horizonte or None
    if h:
        df = a.foto(df, h)
    print(f"Datos: {fuente} · foto a {h or 0} meses · {len(df)} asociados, {int(df[a.TARGET].sum())} se retiran")
    print(f"Modelo: {a.TITULOS[args.modelo]} · {args.folds} particiones · TimeSeriesSplit por {v.ORDEN_TEMPORAL}\n")

    modelo = a.modelos_supervisados()[args.modelo]
    por_particion = v.comparar_validaciones(modelo, df, folds=args.folds, random_state=args.semilla)
    resumen = v.resumen_validaciones(por_particion)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("=== Cada partición ===")
        print(por_particion.round(3).to_string(index=False))
        print("\n=== Por estrategia ===")
        print(resumen.round(3).to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figura = FIGURES_DIR / "validacion_cruzada.html"
    titulo = f"PR-AUC por partición · {a.TITULOS[args.modelo]} · {args.folds} particiones"
    v.figura_validaciones(por_particion, titulo).write_html(figura, include_plotlyjs="cdn")
    params = {"datos": fuente, "filas": len(df), "horizonte_meses": h, "modelo": args.modelo, "folds": args.folds}
    resultado = {
        **params,
        "resumen": resumen.reset_index().to_dict(orient="records"),
        "particiones": por_particion.to_dict(orient="records"),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"validacion_cruzada_{stamp}.json"
    texto = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    metricas.write_text(texto, encoding="utf-8", newline="\n")
    print(f"\nFigura -> {figura} · métricas -> {metricas}")

    if args.sin_mlflow:
        return
    r.configurar()
    por_estrategia = {
        f"{r.nombre_metrica(e)}_{m}": float(fila[m])
        for e, fila in resumen.iterrows()
        for m in ("pr_auc", "pr_auc_std", "roc_auc")
    }
    with r.corrida("validacion_cruzada", params, por_estrategia, artefactos=[metricas, figura]):
        pass
    print(f"MLflow: experimento '{r.EXPERIMENTO}', corrida 'validacion_cruzada'. Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
