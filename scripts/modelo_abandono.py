"""Entrena los modelos de abandono y los valida con asociados que nunca vieron.

Supervisados: Ridge, regresión logística, Random Forest (bagging) y HistGradientBoosting (boosting).
Sin objetivo: KMeans.

Uso:
    uv run python scripts/modelo_abandono.py                         # entrena con DATOS_ASOCIADOS del .env
    uv run python scripts/modelo_abandono.py --nuevos 5000 --semilla-nuevos 7
    uv run python scripts/modelo_abandono.py --validar otro_archivo.parquet
    uv run python scripts/modelo_abandono.py --exportar-nuevos data/processed/nuevos.parquet
    uv run python scripts/modelo_abandono.py --faltantes      # valida también con nuevos con datos vacíos
    uv run python scripts/modelo_abandono.py --sin-limpieza   # sin la limpieza de cleaning.py, para comparar
    uv run python scripts/modelo_abandono.py --horizonte 3    # foto del asociado: ¿se retira en 3 meses?
    uv run python scripts/modelo_abandono.py --sin-mlflow     # sin registrar en MLflow

Los "nuevos" salen del mismo generador con OTRA semilla: son asociados distintos que siguen
el mismo comportamiento. Las métricas sobre ellos son las que cuentan; las de entrenamiento
son optimistas porque el modelo ya conoce esas filas.

Cada modelo queda en MLflow (experimento ``abandono``, corrida "modelo_abandono" con una hija por modelo:
métricas entrenamiento_, cv_, nuevos_ y los gráficos adjuntos). En models/ no se guarda nada: el modelo
que se usa es el campeón (make campeon).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from proyecto_ml import abandono as a
from proyecto_ml import registro as r
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, get_settings
from proyecto_ml.evaluate import (
    COLUMNAS_COMPARATIVA,
    compute_metrics,
    figura_importancia,
    figura_importancia_interactiva,
    importancia_arbol,
    importancia_variables,
    resaltar,
    tabla_comparativa,
    top_importancia,
)
from proyecto_ml.features import pagina_pipelines
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
    parser.add_argument(
        "--ordenar", default="AUC", choices=COLUMNAS_COMPARATIVA, help="Columna para ordenar la tabla comparativa"
    )
    parser.add_argument(
        "--clip-outliers",
        choices=["iqr", "quantile"],
        default=a.CLIP_OUTLIERS,
        help="Cómo recortar atípicos de SALARIO, EDAD y ANTIGUEDAD (aprendido en train). Por defecto: iqr",
    )
    parser.add_argument(
        "--horizonte",
        type=int,
        metavar="MESES",
        help="Foto del asociado en t0 = corte - MESES: activos en t0, objetivo = se retira entre t0 y el corte.\n"
        "Sin la opción, todos los asociados al corte (como antes). Con los 50 de ejemplo quedan muy pocos retiros",
    )
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar los modelos en MLflow")
    parser.add_argument(
        "--sin-limpieza",
        action="store_true",
        help="Apagar la limpieza (extracto, ColumnDropper y recorte) para comparar",
    )
    args = parser.parse_args()
    limpiar = not args.sin_limpieza
    clip, max_nulos = (args.clip_outliers, a.MAX_NULOS) if limpiar else (None, None)

    train = a.cargar(args.datos, limpiar)
    validaciones = {f"nuevos (n={args.nuevos})": GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()}
    if args.faltantes:
        sucios = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos, faltantes=FALTANTES).generar()
        validaciones["nuevos con vacíos"] = sucios
    if args.validar:
        validaciones[args.validar.name] = a.cargar(args.validar, limpiar)
    if args.exportar_nuevos:
        args.exportar_nuevos.parent.mkdir(parents=True, exist_ok=True)
        next(iter(validaciones.values())).to_parquet(args.exportar_nuevos, index=False)
        print(f"Asociados nuevos guardados en {args.exportar_nuevos}")
    if args.horizonte:
        print(f"Foto del asociado (corte {a.FECHA_CORTE.date()}), según el horizonte:")
        print(a.resumen_foto(train).round(3).to_string() + "\n")
        train = a.foto(train, args.horizonte)
        validaciones = {nombre: a.foto(df, args.horizonte) for nombre, df in validaciones.items()}
        minimo = int(train[a.TARGET].value_counts().reindex([0, 1], fill_value=0).min())
        if minimo < 5:  # la validación cruzada estratificada necesita al menos 5 de cada clase
            raise SystemExit(
                f"Con --horizonte {args.horizonte} la clase minoritaria tiene {minimo} asociados (mínimo 5). "
                'Usa más filas (make ejemplo ARGS="--rows 2000 ...") o un horizonte mayor.'
            )
        print(f"Horizonte: {args.horizonte} meses (t0 = {a.fecha_observacion(args.horizonte).date()})")
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)")
    if limpiar:
        print(
            f"Limpieza: extracto (limpiar_extracto) + ColumnDropper (> {a.MAX_NULOS:.0%} vacíos o constantes)"
            f" + recorte {clip} en {', '.join(a.COLUMNAS_RECORTE)}\n"
        )
    else:
        print("Limpieza: apagada (--sin-limpieza)\n")

    # ---- Supervisados: Ridge, logística, Random Forest (bagging), HistGB (boosting)
    X, y = a.separar(train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    supervisados, m_supervisados = {}, {}
    for nombre, modelo in a.modelos_supervisados(args.alpha, clip_outliers=clip, max_nulos=max_nulos).items():
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

    comparativa = tabla_comparativa(supervisados, X, y, ordenar=args.ordenar)  # clona: entrena de cero por partición
    print(f"\n=== Tabla comparativa: validación cruzada (5), mismas particiones, por {args.ordenar} ===")
    print(comparativa.round(4).to_string())

    # Qué variables usa cada modelo: permutation importance sobre los asociados nuevos (no vistos).
    con_objetivo = validaciones[nombre_nuevos]
    importancias = importancia_variables(supervisados, a.preparar(con_objetivo), con_objetivo[a.TARGET])
    print(f"\n=== Variables que más aportan a cada modelo: caída de AUC al desordenarlas ({nombre_nuevos}) ===")
    print(top_importancia(importancias).to_string())
    importancia_rf = importancia_arbol(supervisados["random_forest"])  # la propia del bosque, por columna transformada
    print("\n=== Random Forest: feature_importances_ por columna transformada (top 10) ===")
    print(importancia_rf.head(10).round(4).to_string(index=False))

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
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grafico = FIGURES_DIR / "abandono_matriz_confusion.png"
    fig = a.figura_matrices_confusion(supervisados, agrupador, validaciones[nombre_nuevos])
    fig.suptitle(f"Matriz de confusión - {nombre_nuevos}", fontsize=14)
    fig.savefig(grafico, dpi=150)
    tabla_html = FIGURES_DIR / "abandono_comparativa.html"  # con el mejor de cada columna en amarillo
    tabla_html.write_text(resaltar(comparativa).to_html(), encoding="utf-8", newline="\n")
    importancia_png = FIGURES_DIR / "abandono_importancia.png"
    figura_importancia(importancias, titulos=a.TITULOS).savefig(importancia_png, dpi=150)
    importancia_rf_html = FIGURES_DIR / "abandono_importancia_rf.html"  # interactivo: pasar el mouse por las barras
    titulo_rf = "Importancia de las variables - Random Forest (feature_importances_)"
    figura_importancia_interactiva(importancia_rf, titulo_rf).write_html(importancia_rf_html)
    diagramas = FIGURES_DIR / "abandono_pipelines.html"  # diagrama interactivo de cada pipeline
    pipelines = {a.TITULOS[n]: m for n, m in supervisados.items()} | {"KMeans (sin objetivo)": agrupador.pipeline}
    diagramas.write_text(pagina_pipelines(pipelines, "Pipelines del modelo de abandono"), encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"abandono_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "alpha": args.alpha,
        "limpieza": limpiar,
        "horizonte_meses": args.horizonte,
        "clip_outliers": clip,
        "max_nulos": max_nulos,
        **m_supervisados,
        "kmeans": m_kmeans,
        "comparativa": comparativa.to_dict(orient="index"),
        "importancia": importancias.to_dict(orient="records"),
        "importancia_rf": importancia_rf.to_dict(orient="records"),
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\nMétricas -> {salida}")
    print(f"Gráfico -> {grafico} | tabla comparativa -> {tabla_html}")
    print(f"Diagramas de los pipelines -> {diagramas} | importancia -> {importancia_png}")
    print(f"Importancia del Random Forest (interactiva) -> {importancia_rf_html}")

    if args.sin_mlflow:
        return
    # Prefijos de MLflow: entrenamiento_, cv_ (validación cruzada), nuevos_ (asociados que no vio)...
    prefijos = {"entrenamiento": "entrenamiento", "validación cruzada (5)": "cv", nombre_nuevos: "nuevos"}
    prefijos |= {"nuevos con vacíos": "nuevos_vacios", **({args.validar.name: "validar"} if args.validar else {})}

    def con_prefijos(por_conjunto: dict) -> dict[str, float]:
        return r.unir(*({"m": r.con_prefijo(prefijos[k], m)} for k, m in por_conjunto.items()))["m"]

    metricas = {n: con_prefijos(m) for n, m in m_supervisados.items()} | {"kmeans": con_prefijos(m_kmeans)}
    params = {"datos": args.datos.name, "filas": len(train), "limpieza": limpiar, "clip_outliers": clip}
    params |= {"max_nulos": max_nulos, "horizonte_meses": args.horizonte, "alpha": args.alpha}
    r.configurar()
    artefactos = [salida, grafico, tabla_html, importancia_png, importancia_rf_html, diagramas]
    with r.corrida("modelo_abandono", params, artefactos=artefactos):
        r.registrar_modelos({**supervisados, "kmeans": agrupador}, params, metricas, titulos=a.TITULOS)
    print(f"MLflow: experimento '{r.EXPERIMENTO}', corrida 'modelo_abandono' (make mlflow-ui)")


if __name__ == "__main__":
    main()
