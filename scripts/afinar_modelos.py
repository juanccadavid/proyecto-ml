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
- Preprocesamiento (los dos): vacíos numéricos con mediana o media, categóricas en OneHot u ordinal,
             y con o sin recorte IQR de SALARIO, EDAD y ANTIGUEDAD (a.COLUMNAS_RECORTE).

La tabla por partición (pipeline | fold | Accuracy) muestra, para cada preprocesamiento, su mejor
combinación del resto de hiperparámetros en cada una de las particiones de la validación cruzada.

En MLflow (experimento ``abandono``) queda una corrida "afinar" con una hija por modelo (cv_ y nuevos_).
En models/ no se guarda nada: el modelo que se usa es el campeón (make campeon).
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml import registro as r
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, get_settings
from proyecto_ml.evaluate import (
    SCORING,
    afinar,
    figura_busqueda,
    figura_importancia,
    figura_por_fold,
    formato_params,
    importancia_variables,
    mejores_por_grupo,
    resaltar,
    resultados_busqueda,
    resultados_por_fold,
    tabla_comparativa,
    top_importancia,
)
from proyecto_ml.features import aplanar, pagina_pipelines
from proyecto_ml.generador import GeneradorAsociados

REGULARIZACION = np.logspace(-3, 3, 13)  # 0.001 ... 1000, 13 valores igual de espaciados en escala log
PREPROCESAMIENTO = {
    "preprocess__numeric_strategy": ["median", "mean"],
    "preprocess__encoding": ["onehot", "ordinal"],
    "preprocess__clip_outliers": [None, "iqr"],
}
PARAMS_PREPROCESAMIENTO = ("numeric_strategy", "encoding", "clip_outliers")  # sin el prefijo del paso
GRILLAS = {
    "ridge": {**PREPROCESAMIENTO, "model__alpha": REGULARIZACION, "model__class_weight": [None, "balanced"]},
    "pycaret_lr": {
        **PREPROCESAMIENTO,
        "model__C": REGULARIZACION,
        "model__l1_ratio": [0.0, 1.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["liblinear"],
    },
}
EJE_X = {"ridge": "alpha", "pycaret_lr": "C"}  # el hiperparámetro de regularización de cada uno
NOMBRE_FOLD = {"ridge": "Ridge", "pycaret_lr": "LogisticRegression"}  # prefijo en la tabla por partición
# Variables que se muestran en la terminal (el CSV lleva todas).
COLUMNAS_MUESTRA = ["EDAD", "SEXO", "TIPOCONTRATO", "#INTENCIONRETIROS_ULT3MESES", "#SOLICITUD_CANCELACIONAHORROS"]


def modelos_base(train: pd.DataFrame, folds: int = 5) -> dict[str, Pipeline]:
    """Los dos modelos con sus valores por defecto, sin entrenar, con el preprocesamiento como ``Preprocesador``.

    - ridge: el de ``a.modelo_ridge()``: mediana + OneHot, con la limpieza de cleaning.py activa
      (ColumnDropper y recorte IQR de COLUMNAS_RECORTE).
    - pycaret_lr: la LogisticRegression que arma PyCaret (``make pycaret``) con el preprocesamiento de
      PyCaret reproducido con ``build_preprocessor``: media + ordinal. Predice igual que el Pipeline de
      PyCaret, pero así el preprocesamiento entra en la búsqueda como un hiperparámetro más.
    """
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment(target=a.TARGET, session_id=42, fold=folds, normalize=True)
    exp.fit(a.preparar(train), train[a.TARGET])

    # pycaret_lr: el preprocesamiento de PyCaret (media + ordinal, sin recorte ni ColumnDropper)
    lr = clone(exp.create_model("lr").pipeline[-1])
    sin_limpieza = {"numeric_strategy": "mean", "encoding": "ordinal", "clip_outliers": None, "max_nulos": None}
    return {
        "ridge": a.armar_pipeline(clone(a.modelo_ridge()[-1]), buscable=True),
        "pycaret_lr": a.armar_pipeline(lr, buscable=True, **sin_limpieza),
    }


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
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    try:
        import pycaret  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyCaret no está instalado. Ejecuta: uv sync --extra pycaret") from exc
    warnings.filterwarnings("ignore", category=UserWarning)  # avisos internos de la versión alfa

    train = a.cargar(args.datos)  # con limpiar_extracto
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos).generar()
    X, y = a.separar(train)
    print(f"Entrenamiento: {args.datos} ({len(train)} filas, {train[a.TARGET].mean():.0%} abandono)")
    print(f"Elige por: {args.metrica} (validación cruzada, {args.folds} particiones)\n")

    base = modelos_base(train, args.folds)
    busquedas, modelos = {}, {}
    for nombre, modelo in base.items():
        busqueda = afinar(modelo, GRILLAS[nombre], X, y, folds=args.folds, metrica=args.metrica, n_jobs=-1)
        busquedas[nombre] = busqueda
        modelos[nombre], modelos[f"{nombre}_afinado"] = modelo, busqueda.best_estimator_
        print(f"=== {nombre}: {len(busqueda.cv_results_['params'])} combinaciones ===")
        print(f"Mejor: {formato_params(busqueda.best_params_)}")
        print(resultados_busqueda(busqueda).round(4).to_string() + "\n")

    # Tabla por partición (como pipeline | fold | accuracy): un pipeline por preprocesamiento y modelo.
    metricas_fold = tuple(dict.fromkeys(["Accuracy", args.metrica]))  # sin repetir si --metrica Accuracy
    por_fold = pd.concat(
        [
            resultados_por_fold(b, list(PARAMS_PREPROCESAMIENTO), NOMBRE_FOLD[n], metricas_fold)
            for n, b in busquedas.items()
        ],
        ignore_index=True,
    )
    print(f"=== Por partición: cada preprocesamiento con su mejor {'/'.join(EJE_X.values())} y class_weight ===")
    with pd.option_context("display.max_rows", None):
        print(por_fold.round(6))
    por_preprocesamiento = por_fold.groupby("pipeline", sort=False)[list(metricas_fold)].agg(["mean", "std"])
    # La prueba honesta de cada preprocesamiento: su mejor combinación, reentrenada, sobre asociados nuevos.
    Xn, yn = a.separar(nuevos)
    for n, b in busquedas.items():
        for pipeline, params in mejores_por_grupo(b, list(PARAMS_PREPROCESAMIENTO), NOMBRE_FOLD[n]).items():
            modelo = clone(base[n]).set_params(**params).fit(X, y)
            for m in metricas_fold:
                por_preprocesamiento.loc[pipeline, (m, "nuevos")] = check_scoring(modelo, SCORING[m])(modelo, Xn, yn)
    por_preprocesamiento = por_preprocesamiento[[(m, e) for m in metricas_fold for e in ("mean", "std", "nuevos")]]
    por_preprocesamiento = por_preprocesamiento.sort_values((args.metrica, "mean"), ascending=False)
    print(f"\n=== Preprocesamientos: promedio y desviación entre particiones, y sobre nuevos, por {args.metrica} ===")
    print(por_preprocesamiento.round(4).to_string() + "\n")
    en_nuevos_por_pipeline = por_preprocesamiento.xs("nuevos", axis=1, level=1)  # pipeline x métrica

    comparativa = tabla_comparativa(modelos, X, y, folds=args.folds, ordenar=args.metrica)
    print("=== Por defecto vs. afinado: validación cruzada (los afinados salen optimistas) ===")
    print(comparativa.round(4).to_string())

    # La prueba honesta: asociados que ningún modelo vio (ni en la búsqueda).
    en_nuevos = {n: a.metricas_supervisado(clone(m).fit(X, y), nuevos) for n, m in modelos.items()}
    en_nuevos = pd.DataFrame(en_nuevos).T
    print(f"\n=== Por defecto vs. afinado sobre asociados nuevos (n={args.nuevos}) ===")
    print(en_nuevos.round(3).to_string())

    # Tabla de predicciones de los afinados (como predict_model de PyCaret), sobre los nuevos.
    afinados = {f"{n}_afinado": aplanar(b.best_estimator_) for n, b in busquedas.items()}  # ramas a la vista
    predicciones = a.tabla_predicciones(afinados, nuevos)
    columnas = [*COLUMNAS_MUESTRA, a.TARGET, *[c for c in predicciones if c.endswith(("_label", "_score"))]]
    print(f"\n=== Predicciones de los afinados: muestra de {args.muestra} asociados nuevos ===")
    print("score = riesgo de abandono: probabilidad en la logística; en Ridge, distancia a la frontera (>0 = abandona)")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(predicciones.sample(args.muestra, random_state=0)[columnas].round(4))
    etiquetas = predicciones[[f"{n}_label" for n in afinados]]
    discrepan = etiquetas.ne(etiquetas.iloc[:, 0], axis=0).any(axis=1).sum()
    print(f"Los afinados discrepan en {discrepan} de {len(predicciones)} asociados.")

    importancias = importancia_variables(afinados, Xn, yn)  # sobre los nuevos: datos que no vieron
    print("\n=== Variables que más aportan a los afinados: caída de AUC al desordenarlas (asociados nuevos) ===")
    print(top_importancia(importancias).to_string())

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabla_csv = PROCESSED_DIR / "predicciones_afinados.csv"  # todas las filas y variables; ignorado por Git
    predicciones.to_csv(tabla_csv, index_label="fila", encoding="utf-8-sig")  # -sig: tildes bien en Excel
    grafico = FIGURES_DIR / "afinamiento_curvas.png"
    figura_busqueda(busquedas, EJE_X, fijar=PARAMS_PREPROCESAMIENTO).savefig(grafico, dpi=150)
    tabla_html = FIGURES_DIR / "afinamiento_comparativa.html"  # con el mejor de cada columna en amarillo
    tabla_html.write_text(resaltar(comparativa).to_html(), encoding="utf-8", newline="\n")
    bigotes = FIGURES_DIR / "afinamiento_bigotes.png"
    figura_por_fold(por_fold, list(metricas_fold), nuevos=en_nuevos_por_pipeline, ordenar=args.metrica).savefig(
        bigotes, dpi=150
    )
    importancia_png = FIGURES_DIR / "afinamiento_importancia.png"
    figura_importancia(importancias).savefig(importancia_png, dpi=150)
    diagramas = FIGURES_DIR / "afinamiento_pipelines.html"  # los afinados, con los hiperparámetros elegidos
    diagramas.write_text(pagina_pipelines(afinados, "Pipelines afinados"), encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = METRICS_DIR / f"afinamiento_{stamp}.json"
    resumen = {
        "datos": str(args.datos.name),
        "metrica": args.metrica,
        "mejores": {n: formato_params(b.best_params_) for n, b in busquedas.items()},
        "top5": {n: resultados_busqueda(b).to_dict(orient="records") for n, b in busquedas.items()},
        "comparativa": comparativa.to_dict(orient="index"),
        "importancia": importancias.to_dict(orient="records"),
        "por_fold": por_fold.to_dict(orient="records"),
        "preprocesamientos": {
            p: {f"{m}_{e}": v for (m, e), v in fila.items()} for p, fila in por_preprocesamiento.iterrows()
        },
        f"nuevos (n={args.nuevos})": en_nuevos.to_dict(orient="index"),
    }
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Métricas -> {salida}")
    print(f"Curvas -> {grafico} | tabla comparativa -> {tabla_html}")
    print(f"Predicciones ({len(predicciones)} asociados nuevos) -> {tabla_csv}")
    print(f"Diagramas de los pipelines -> {diagramas}")
    print(f"Bigotes por partición -> {bigotes} | importancia -> {importancia_png}")

    if args.sin_mlflow:
        return
    r.configurar()
    params = {"datos": args.datos.name, "filas": len(train), "metrica": args.metrica, "folds": args.folds}
    metricas = r.unir(r.metricas_tabla(comparativa.drop(columns="Model"), "cv"), r.metricas_tabla(en_nuevos, "nuevos"))
    artefactos = [salida, grafico, tabla_html, bigotes, importancia_png, diagramas]
    with r.corrida("afinar", params, artefactos=artefactos):
        for nombre, b in busquedas.items():
            afinado = f"{nombre}_afinado"
            propios = {"modelo": afinado, **params, **{k.split("__")[-1]: v for k, v in b.best_params_.items()}}
            with r.corrida(afinado, propios, metricas.get(afinado), anidada=True):
                r.registrar_modelo(afinados[afinado])
    print(f"MLflow: experimento '{r.EXPERIMENTO}', corrida 'afinar' (make mlflow-ui)")


if __name__ == "__main__":
    main()
