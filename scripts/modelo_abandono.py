"""Entrena los modelos de abandono y los valida con asociados que nunca vieron.

Supervisados: Ridge, regresión logística, Random Forest (bagging) y HistGradientBoosting (boosting).
Sin objetivo: KMeans.

Uso:
    uv run python scripts/modelo_abandono.py                         # entrena con DATOS_ASOCIADOS del .env
    uv run python scripts/modelo_abandono.py --nuevos 5000 --semilla-nuevos 7
    uv run python scripts/modelo_abandono.py --validar otro_archivo.parquet
    uv run python scripts/modelo_abandono.py --exportar-nuevos data/processed/nuevos.parquet
    uv run python scripts/modelo_abandono.py --faltantes      # valida también con nuevos con datos vacíos

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from proyecto_ml import abandono as a
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, get_settings
from proyecto_ml.data import load_dataset
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import FALTANTES, GeneradorAsociados


def _tabla(columnas: dict[str, dict]) -> str:
    return pd.DataFrame(columnas).round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--datos", type=Path, default=get_settings().datos_asociados, help="Por defecto DATOS_ASOCIADOS del .env"
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularización de Ridge (más alto = más simple)")
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar")
    parser.add_argument("--semilla-nuevos", type=int, default=2026, help="Distinta de 42 (la de data/ejemplo)")
    parser.add_argument("--validar", type=Path, help="Archivo adicional para validar (mismo esquema)")
    parser.add_argument("--exportar-nuevos", type=Path, help="Guardar los asociados nuevos generados")
    parser.add_argument("--faltantes", action="store_true", help="Validar además con nuevos con vacíos")
    args = parser.parse_args()

    train = load_dataset(args.datos)
    validaciones = {f"nuevos (n={args.nuevos})": GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()}
    if args.faltantes:
        sucios = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos, faltantes=FALTANTES).generar()
        validaciones["nuevos con vacíos"] = sucios
    if args.validar:
        validaciones[args.validar.name] = load_dataset(args.validar)
    if args.exportar_nuevos:
        args.exportar_nuevos.parent.mkdir(parents=True, exist_ok=True)
        next(iter(validaciones.values())).to_parquet(args.exportar_nuevos, index=False)
        print(f"Asociados nuevos guardados en {args.exportar_nuevos}")
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)\n")

    # ---- Supervisados: Ridge, logística, Random Forest (bagging), HistGB (boosting)
    X, y = a.preparar(train), train[a.TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    supervisados, m_supervisados = {}, {}
    for nombre, modelo in a.modelos_supervisados(args.alpha).items():
        metodo = a.metodo_puntaje(modelo)
        puntaje_cv = cross_val_predict(modelo, X, y, cv=cv, method=metodo)  # clona: no toca `modelo`
        puntaje_cv = puntaje_cv[:, 1] if puntaje_cv.ndim == 2 else puntaje_cv
        supervisados[nombre] = modelo.fit(X, y)
        m_supervisados[nombre] = {
            "entrenamiento": a.metricas_supervisado(modelo, train),
            "validación cruzada (5)": compute_metrics(y, puntaje_cv, threshold=a.UMBRAL[metodo]),
            **{nombre_df: a.metricas_supervisado(modelo, df) for nombre_df, df in validaciones.items()},
        }
        print(f"=== {a.TITULOS[nombre]}: ¿abandona? ===")
        print(_tabla(m_supervisados[nombre]) + "\n")
    print("Variables con más peso en Ridge (+ empuja a abandono):")
    print(a.coeficientes(supervisados["ridge"], 8).round(3).to_string())

    nombre_nuevos = next(iter(validaciones))  # las predicciones que cuentan: asociados nunca vistos
    print(f"\n=== Comparación sobre {nombre_nuevos} ===")
    print(_tabla({n: m[nombre_nuevos] for n, m in m_supervisados.items()}))

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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grafico = FIGURES_DIR / "abandono_matriz_confusion.png"
    fig = a.figura_matrices_confusion(supervisados, agrupador, validaciones[nombre_nuevos])
    fig.suptitle(f"Matriz de confusión - {nombre_nuevos}", fontsize=14)
    fig.savefig(grafico, dpi=150)
    for nombre, modelo in supervisados.items():
        joblib.dump(modelo, MODELS_DIR / f"abandono_{nombre}.joblib")
    joblib.dump(agrupador, MODELS_DIR / "abandono_kmeans.joblib")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"abandono_{stamp}.json"
    resumen = {"datos": str(args.datos.name), "alpha": args.alpha, **m_supervisados, "kmeans": m_kmeans}
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nModelos -> {MODELS_DIR / 'abandono_*.joblib'} | métricas -> {salida} | gráfico -> {grafico}")


if __name__ == "__main__":
    main()
