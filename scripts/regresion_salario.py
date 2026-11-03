"""Regresión de ejemplo: estimar el SALARIO del asociado con MAE, MSE, RMSE y R² (make regresion).

Compara la referencia (predecir siempre el promedio) con Ridge, Random Forest y HistGB en validación cruzada
KFold, y valida el mejor con asociados nuevos (ver ``regresion.py``). Solo usa a quienes reportan salario; al
final estima el salario de quienes no lo reportan.

Ojo con los datos generados: el generador saca el salario al azar (lognormal), sin relación con las demás
variables, así que ningún modelo debería ganarle al promedio (R² cerca de 0). Eso es lo correcto: un R² alto
ahí sería fuga. Con el extracto real (salario según cargo, antigüedad, estrato...) el R² debería subir.

Guarda reports/figures/regresion_salario.html y reports/metrics/regresion_<fecha>.json; en MLflow
(experimento ``salario``), una corrida "regresion_salario" con una hija por modelo.

Uso:
    uv run python scripts/regresion_salario.py
    uv run python scripts/regresion_salario.py --datos data/raw/supabase_asociados.parquet --sin-mlflow
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from proyecto_ml import campeon as c
from proyecto_ml import registro as r
from proyecto_ml import regresion as g
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, aviso_origen, es_experimental
from proyecto_ml.generador import DESARROLLO, FALTANTES, GeneradorAsociados

EXPERIMENTO = "salario"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (5.000, semilla 7)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar el mejor")
    parser.add_argument("--semilla-nuevos", type=int, default=2026)
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos)
    X, y = g.separar(df)
    print(f"Datos: {fuente} · {len(X)} asociados con salario (de {len(df)}) · objetivo {g.OBJETIVO}")
    print(aviso_origen())
    print(f"Salario: mediana ${y.median():,.0f} · promedio ${y.mean():,.0f} · desviación ${y.std():,.0f}\n")

    candidatos = g.modelos()
    tabla = g.comparar(candidatos, X, y, folds=args.folds)
    print(f"=== Validación cruzada KFold ({args.folds} particiones), en pesos ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(tabla.to_string())
    mejor = tabla.drop(index="media").index[0]  # el mejor modelo de verdad; la referencia es para comparar
    gana = tabla.loc[mejor, "MAE"] < tabla.loc["media", "MAE"]
    print(
        f"\nMejor modelo por R²: {g.TITULOS[mejor]} (R² {tabla.loc[mejor, 'R2']:.3f}). "
        + ("Le gana" if gana else "NO le gana")
        + f" a predecir siempre el promedio (MAE ${tabla.loc['media', 'MAE']:,.0f})."
    )

    # Asociados nuevos (con vacíos): el mejor y la referencia, entrenados con todo. Salen del generador: con datos
    # reales serían de otro origen, así que solo se usan en modo sintético.
    en_nuevos = {}
    if es_experimental():
        nuevos = GeneradorAsociados(
            args.nuevos, seed=args.semilla_nuevos, tasa_retiro=DESARROLLO["tasa_retiro"], faltantes=FALTANTES
        ).generar()
        Xn, yn = g.separar(nuevos)
        entrenados = {n: clone(candidatos[n]).fit(X, y) for n in (mejor, "media")}
        en_nuevos = {n: g.metricas(yn, m.predict(Xn)) for n, m in entrenados.items()}
        print(f"\n=== Asociados nuevos ({len(Xn)} con salario) ===")
        with pd.option_context("display.float_format", "{:,.3f}".format):
            print(pd.DataFrame(en_nuevos).T.to_string())
        sin_salario = nuevos[pd.to_numeric(nuevos[g.OBJETIVO], errors="coerce").isna()]
        if len(sin_salario):
            estimado = entrenados[mejor].predict(sin_salario)
            print(
                f"\n{len(sin_salario)} nuevos no reportan salario: "
                f"estimado entre ${estimado.min():,.0f} y ${estimado.max():,.0f}"
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figura = FIGURES_DIR / "regresion_salario.html"
    prediccion = g.prediccion_fuera_de_particion(candidatos[mejor], X, y, folds=args.folds)
    titulo = f"Salario real vs. predicho · {g.TITULOS[mejor]} · R² {tabla.loc[mejor, 'R2']:.3f} (KFold)"
    g.figura_real_vs_predicho(y, prediccion, titulo).write_html(figura, include_plotlyjs="cdn")
    params = {"datos": fuente, "filas": len(X), "objetivo": g.OBJETIVO, "folds": args.folds}
    params |= r.parametros_datos(X)
    resultado = {**params, "cv": tabla.reset_index(names="modelo").to_dict(orient="records"), "nuevos": en_nuevos}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"regresion_{stamp}.json"
    texto = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    metricas.write_text(texto, encoding="utf-8", newline="\n")
    print(f"\nFigura -> {figura} · métricas -> {metricas}")

    if args.sin_mlflow:
        return
    r.configurar(EXPERIMENTO)
    with r.corrida("regresion_salario", params, artefactos=[metricas, figura], tags={"mejor": mejor}):
        r.registrar_datos(X, fuente, "entrenamiento", fuente, objetivo=g.OBJETIVO)  # solo quienes reportan salario
        for nombre, fila in tabla.iterrows():
            propias = r.con_prefijo("cv", fila.to_dict())
            if nombre in en_nuevos:
                propias |= r.con_prefijo("nuevos", en_nuevos[nombre])
            with r.corrida(nombre, {**params, "modelo": nombre}, propias, anidada=True):
                pass
    print(f"MLflow: experimento '{EXPERIMENTO}', corrida 'regresion_salario'. Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
