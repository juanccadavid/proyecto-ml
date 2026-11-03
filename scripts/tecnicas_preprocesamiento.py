"""Compara técnicas de preprocesamiento y feature engineering en el abandono, y evalúa la correlación (make tecnicas).

Sobre la foto del asociado (los mismos datos con que se elige el campeón), cada técnica cambia una sola cosa
respecto a la base y se mide con PR-AUC en validación cruzada repetida (ver ``tecnicas.py``):

- Desbalance: class_weight (base), sin balancear, SMOTE, ADASYN.
- Imputación: mediana (base), media, KNNImputer, IterativeImputer, mediana + MissingIndicator.
- Codificación: OneHotEncoder (base), OrdinalEncoder, TargetEncoder.
- Escalado: StandardScaler (base), MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer.
- Feature engineering: variables nuevas (agregaciones, interacción, log) y binning; PCA; RFE.

Los datos generados no traen vacíos: para que los imputadores se distingan, se ponen --vacios (10 %) al azar
en SALARIO, ESTRATO, EDAD, ANTIGUEDAD y #CANTIDAD_SERVICIOS (0 = sin vacíos).

Correlación (en los mismos datos, con las variables nuevas): Spearman entre numéricas, pares redundantes
(|r| >= 0,8) y la relación de cada variable con el retiro (Spearman o V de Cramér).

Guarda reports/figures/tecnicas.html, reports/figures/correlaciones.html y reports/metrics/tecnicas_<fecha>.json;
en MLflow (experimento ``abandono``), una corrida "tecnicas" con la PR-AUC de cada técnica.

Uso:
    uv run python scripts/tecnicas_preprocesamiento.py
    uv run python scripts/tecnicas_preprocesamiento.py --clasificador hist_gb --repeticiones 1 --sin-mlflow
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import correlacion as k
from proyecto_ml import registro as r
from proyecto_ml import tecnicas as t
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR


def _slug(texto: str) -> str:
    """'Imputación_mediana + indicador' -> 'imputacion_mediana_indicador' (nombre de métrica para MLflow)."""
    return r.nombre_metrica(unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (5.000, semilla 7, 10 %%)")
    parser.add_argument(
        "--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses de la foto (0 = todos al corte)"
    )
    parser.add_argument("--clasificador", choices=t.CLASIFICADORES, default="logreg")
    parser.add_argument("--folds", type=int, default=c.FOLDS)
    parser.add_argument("--repeticiones", type=int, default=c.REPETICIONES)
    parser.add_argument("--vacios", type=float, default=0.1, help="Fracción de vacíos al azar (0 = ninguno)")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos)
    h = args.horizonte or None
    if h:
        df = a.foto(df, h)
    X, y = a.preparar(df), df[a.TARGET]
    if args.vacios:
        X = t.con_vacios(X, args.vacios, seed=args.semilla)
    print(f"Datos: {fuente} · foto a {h or 0} meses · {len(X)} asociados, {int(y.sum())} se retiran ({y.mean():.1%})")
    print(f"Modelo: {args.clasificador} · {args.folds} particiones x {args.repeticiones} repeticiones")

    # ---- Correlación
    con_nuevas = a.CrearVariables().transform(X)
    m = k.matriz(con_nuevas)
    redundantes = k.pares_redundantes(m)
    senal = k.con_objetivo(con_nuevas, y)
    print("\n=== Relación con el retiro (de la más fuerte a la más débil) ===")
    print(senal.drop(columns="fuerza").round(3).to_string(index=False))
    print(f"\n=== Pares redundantes (|r| >= {k.UMBRAL_REDUNDANCIA}) ===")
    print(redundantes.round(3).to_string(index=False) if len(redundantes) else "Ninguno")

    # ---- Técnicas
    tabla = t.comparar(X, y, nombre_clasificador=args.clasificador, folds=args.folds, repeticiones=args.repeticiones)
    print("\n=== PR-AUC por técnica (cada una cambia una cosa respecto a la base) ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(tabla.drop(columns="descripcion").round(3).to_string(index=False))
    ruido = tabla.loc[tabla["tecnica"] == "base", "pr_auc_std"].iloc[0]
    concluyentes = tabla.loc[tabla["concluyente"].astype(bool), "tecnica"].tolist()
    print(
        f"\nOjo: la PR-AUC de una técnica varía ±{ruido:.3f} entre particiones, pero la diferencia con la base se mide "
        "partición a partición (dif_ic_inf / dif_ic_sup, IC 95 %). Concluyentes (el intervalo no toca el 0): "
        + (", ".join(concluyentes) if concluyentes else "ninguna")
    )

    # ---- Guardar
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fig_tecnicas = FIGURES_DIR / "tecnicas.html"
    t.figura(tabla, f"PR-AUC por técnica · {args.clasificador}").write_html(fig_tecnicas, include_plotlyjs="cdn")
    fig_corr = FIGURES_DIR / "correlaciones.html"
    k.figura_matriz(m).write_html(fig_corr, include_plotlyjs="cdn")
    params = {
        "datos": fuente,
        "filas": len(X),
        "horizonte_meses": h,
        "clasificador": args.clasificador,
        "folds": args.folds,
        "repeticiones": args.repeticiones,
        "vacios": args.vacios,
    }
    resultado = {
        **params,
        "tecnicas": tabla.to_dict(orient="records"),
        "con_objetivo": senal.to_dict(orient="records"),
        "redundantes": redundantes.to_dict(orient="records"),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"tecnicas_{stamp}.json"
    texto = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    metricas.write_text(texto, encoding="utf-8", newline="\n")
    print(f"\nTécnicas -> {fig_tecnicas} · correlaciones -> {fig_corr} · métricas -> {metricas}")

    if args.sin_mlflow:
        return
    r.configurar()
    por_tecnica = {
        f"{_slug(f'{fila.grupo}_{fila.tecnica}')}_{metrica}": getattr(fila, metrica)
        for fila in tabla.itertuples()
        for metrica in ("pr_auc", "roc_auc")
    }
    with r.corrida("tecnicas", params, por_tecnica, artefactos=[metricas, fig_tecnicas, fig_corr]):
        pass
    print(f"MLflow: experimento '{r.EXPERIMENTO}', corrida 'tecnicas'. Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
