"""Entrena los modelos de abandono (Ridge y KMeans) y los valida con asociados que nunca vieron.

Uso:
    uv run python scripts/modelo_abandono.py                         # entrena con data/ejemplo, valida con 1000 nuevos
    uv run python scripts/modelo_abandono.py --nuevos 5000 --semilla-nuevos 7
    uv run python scripts/modelo_abandono.py --validar otro_archivo.parquet
    uv run python scripts/modelo_abandono.py --exportar-nuevos data/processed/nuevos.parquet

Los "nuevos" salen del mismo generador con OTRA semilla: son asociados distintos que siguen
el mismo comportamiento. Las métricas sobre ellos son las que cuentan; las de entrenamiento
son optimistas porque el modelo ya conoce esas filas.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from generar_asociados_ejemplo import make_asociados  # scripts/ está en sys.path al ejecutar este script
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from proyecto_ml import abandono as a
from proyecto_ml.config import DATA_DIR, METRICS_DIR, MODELS_DIR
from proyecto_ml.data import load_dataset
from proyecto_ml.evaluate import compute_metrics


def _tabla(columnas: dict[str, dict]) -> str:
    return pd.DataFrame(columnas).round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, default=DATA_DIR / "ejemplo" / "asociados_ejemplo.parquet")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularización de Ridge (más alto = más simple)")
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="Distinta de 42 (la de data/ejemplo)")
    parser.add_argument("--validar", type=Path, help="Archivo adicional para validar (mismo esquema)")
    parser.add_argument("--exportar-nuevos", type=Path, help="Guardar los asociados nuevos generados")
    args = parser.parse_args()

    train = load_dataset(args.datos)
    validaciones = {f"nuevos (n={args.nuevos})": make_asociados(args.nuevos, seed=args.semilla_nuevos)}
    if args.validar:
        validaciones[args.validar.name] = load_dataset(args.validar)
    if args.exportar_nuevos:
        args.exportar_nuevos.parent.mkdir(parents=True, exist_ok=True)
        next(iter(validaciones.values())).to_parquet(args.exportar_nuevos, index=False)
        print(f"Asociados nuevos guardados en {args.exportar_nuevos}")
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)\n")

    # ---- Ridge (supervisado)
    X, y = a.preparar(train), train[a.TARGET]
    ridge = a.modelo_ridge(args.alpha).fit(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    puntaje_cv = cross_val_predict(a.modelo_ridge(args.alpha), X, y, cv=cv, method="decision_function")
    m_ridge = {
        "entrenamiento": a.metricas_ridge(ridge, train),
        "validación cruzada (5)": compute_metrics(y, puntaje_cv, threshold=0.0),
        **{nombre: a.metricas_ridge(ridge, df) for nombre, df in validaciones.items()},
    }
    print(f"=== Ridge (alpha={args.alpha}): ¿abandona? ===")
    print(_tabla(m_ridge))
    print("\nVariables con más peso (+ empuja a abandono):")
    print(a.coeficientes(ridge, 8).round(3).to_string())

    # ---- KMeans (sin objetivo)
    agrupador = a.AgrupadorAbandono().fit(train)
    m_kmeans = {
        "entrenamiento": a.metricas_agrupacion(agrupador, train),
        **{nombre: a.metricas_agrupacion(agrupador, df) for nombre, df in validaciones.items()},
    }
    print("\n=== KMeans (2 grupos, sin ver #RETIROS) vs. abandono real ===")
    print(_tabla(m_kmeans))
    print("\nPerfil promedio de cada grupo:")
    print(agrupador.perfil_.T.round(2).to_string())

    # ---- Guardar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, MODELS_DIR / "abandono_ridge.joblib")
    joblib.dump(agrupador, MODELS_DIR / "abandono_kmeans.joblib")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"abandono_{stamp}.json"
    resumen = {"datos": str(args.datos.name), "alpha": args.alpha, "ridge": m_ridge, "kmeans": m_kmeans}
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nModelos -> {MODELS_DIR / 'abandono_*.joblib'} | métricas -> {salida}")


if __name__ == "__main__":
    main()
