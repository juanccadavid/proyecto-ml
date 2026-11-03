"""Segmentación no supervisada de los asociados: KMeans y DBSCAN, con su mapa interactivo (make segmentos).

Sobre la foto del asociado (los mismos datos del campeón), solo con el comportamiento y sin ver el retiro:

- KMeans con ``k`` elegido por la silueta (2 a 8), o el que se pida con --k;
- DBSCAN con ``eps`` de la curva de vecinos: los atípicos (fuera de toda zona densa) quedan aparte;
- el perfil de cada segmento y, solo al final, su tasa de retiro real: no supervisado vs. supervisado.

Guarda reports/figures/segmentos.html (mapa con botón KMeans/DBSCAN + silueta por k) y
reports/metrics/segmentos_<fecha>.json; en MLflow (experimento ``abandono``), una corrida "segmentos".

Uso:
    uv run python scripts/segmentar.py
    uv run python scripts/segmentar.py --k 4 --sin-mlflow
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
from proyecto_ml import segmentacion as s
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, aviso_origen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (5.000, semilla 7, 10 %%)")
    parser.add_argument(
        "--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses de la foto (0 = todos al corte)"
    )
    parser.add_argument("--k", type=int, help="Segmentos de KMeans (por defecto, el de mejor silueta)")
    parser.add_argument("--eps", type=float, help="Radio de DBSCAN (por defecto, de la curva de vecinos)")
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos)
    h = args.horizonte or None
    if h:
        df = a.foto(df, h)
    print(f"Datos: {fuente} · foto a {h or 0} meses · {len(df)} asociados · {len(s.VARIABLES)} variables")
    print(aviso_origen())

    seg = s.Segmentacion(k=args.k, eps=args.eps).fit(df)
    if seg.tabla_k_ is not None:
        print("\n=== KMeans: silueta e inercia según k ===")
        print(seg.tabla_k_.round(3).T.to_string())
    kmeans, dbscan = seg.kmeans(df), seg.dbscan(df)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(f"\n=== KMeans (k = {seg.k_}): perfil de cada segmento; la tasa de retiro no la vio ===")
        print(s.perfil(df, kmeans).round(3).to_string())
        print(f"\n=== DBSCAN (eps = {seg.eps_:.2f}, min_samples = {seg.min_samples_}; -1 = atípicos) ===")
        print(s.perfil(df, dbscan).round(3).to_string())
        comparacion = s.comparar_con_retiro(df, seg)
        print("\n=== No supervisado vs. retiro real: ¿el segmento 'de riesgo' encuentra a los que se van? ===")
        print(comparacion.round(3).to_string())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figura = FIGURES_DIR / "segmentos.html"
    partes = [s.figura_mapa(seg, df).to_html(full_html=False, include_plotlyjs="cdn")]
    if seg.tabla_k_ is not None:
        partes.append(s.figura_k(seg.tabla_k_, seg.k_).to_html(full_html=False, include_plotlyjs=False))
    pagina = '<!doctype html><meta charset="utf-8"><title>Segmentos de asociados</title>' + "".join(partes)
    figura.write_text(pagina + "\n", encoding="utf-8", newline="\n")
    params = {
        **r.parametros_datos(df),
        "datos": fuente,
        "filas": len(df),
        "horizonte_meses": h,
        "k": seg.k_,
        "eps": seg.eps_,
        "min_samples": seg.min_samples_,
    }
    resultado = {
        **params,
        "silueta": seg.tabla_k_["silueta"].to_dict() if seg.tabla_k_ is not None else None,
        "kmeans": s.perfil(df, kmeans).reset_index().to_dict(orient="records"),
        "dbscan": s.perfil(df, dbscan).reset_index().to_dict(orient="records"),
        "comparacion": comparacion.reset_index(names="metodo").to_dict(orient="records"),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"segmentos_{stamp}.json"
    texto = json.dumps(resultado, ensure_ascii=False, indent=2, default=float) + "\n"
    metricas.write_text(texto, encoding="utf-8", newline="\n")
    print(f"\nMapa -> {figura} · métricas -> {metricas}")

    if args.sin_mlflow:
        return
    r.configurar()
    propias = {
        f"{'kmeans' if metodo.startswith('KMeans') else 'dbscan'}_{m}": float(fila[m])
        for metodo, fila in comparacion.iterrows()
        for m in ("tasa_retiro", "precision", "recall", "ari")
    }
    with r.corrida("segmentos", params, propias, artefactos=[metricas, figura]):
        r.registrar_datos(df, fuente, "eleccion", fuente)
    print(f"MLflow: experimento '{r.EXPERIMENTO}', corrida 'segmentos'. Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
