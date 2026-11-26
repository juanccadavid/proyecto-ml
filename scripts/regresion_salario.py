"""Regresión de ejemplo: estimar el SALARIO del asociado con MAE, MSE, RMSE y R² (make regresion).

Aparta primero una reserva del 20 % que no se usa para elegir. Con el resto compara la referencia (predecir siempre
el promedio) con Ridge, Random Forest y HistGB en validación cruzada KFold y elige el mejor; después mide una sola
vez al elegido y a la referencia en la reserva (sin sesgo de selección; ver ``regresion.py``). Con sintéticos,
además, en asociados nuevos del generador. Solo usa a quienes reportan un salario plausible; al final estima el
salario de quienes no lo reportan. Es una estimación transversal: describe el salario en el mismo corte.

En sintéticos el salario se muestrea por cuantiles y recibe un factor adicional en las etiquetas sintéticas de
retiro; no es una variable lognormal independiente. En la ejecución actual los modelos no superan al promedio en
la reserva sintética. Ese resultado solo describe el generador. El ajuste sobre el extracto real estima el salario
en el mismo corte y no valida el clasificador de abandono.

Con ``ORIGEN_DATOS=real_anonimizado`` entrena con el extracto real (no necesita la etiqueta de retiro; make
regresion-real). Los salarios imposibles (relleno 987.654, miles de millones) no se aprenden: quedan entre los
que el modelo estima, como los que no reportan salario.

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
from proyecto_ml.figuras import marcar_origen
from proyecto_ml.generador import FALTANTES, asociados_nuevos

EXPERIMENTO = "salario"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (5.000, semilla 7)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos para validar el mejor")
    parser.add_argument("--semilla-nuevos", type=int, default=2026)
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos, supervisado=False)  # no usa la etiqueta: sirve con datos reales
    X_todos, y_todos = g.separar(df)
    X, X_reserva, y, y_reserva = g.separar_reserva(X_todos, y_todos)  # la reserva no se mira al elegir
    print(
        f"Datos: {fuente} · {len(X_todos)} asociados con salario plausible (de {len(df)}) · objetivo {g.OBJETIVO} · "
        f"{len(X)} para elegir y {len(X_reserva)} de reserva"
    )
    print(aviso_origen())
    print(f"Salario: mediana ${y.median():,.0f} · promedio ${y.mean():,.0f} · desviación ${y.std():,.0f}\n")

    candidatos = g.modelos()
    tabla = g.comparar(candidatos, X, y, folds=args.folds)
    print(f"=== Elección: validación cruzada KFold ({args.folds} particiones) sin la reserva, en pesos ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(tabla.to_string())
    mejor = tabla.drop(index="media").index[0]  # el mejor modelo de verdad; la referencia es para comparar
    print(f"\nElegido por R² en validación cruzada: {g.TITULOS[mejor]} (R² {tabla.loc[mejor, 'R2']:.3f}, optimista).")

    # La reserva mide al elegido una sola vez: es la cifra que se reporta.
    en_reserva_pred = g.predecir_reserva(candidatos, [mejor, "media"], X, y, X_reserva)
    en_reserva = {n: g.metricas(y_reserva, p) for n, p in en_reserva_pred.items()}
    gana = en_reserva[mejor]["MAE"] < en_reserva["media"]["MAE"]
    print(f"\n=== Reserva ({len(X_reserva)} asociados que no se usaron para elegir) ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(pd.DataFrame(en_reserva).T.to_string())
    print(
        f"En la reserva, {g.TITULOS[mejor]} tiene R² {en_reserva[mejor]['R2']:.3f} y "
        + ("le gana" if gana else "NO le gana")
        + f" a predecir siempre el promedio (MAE ${en_reserva[mejor]['MAE']:,.0f} frente a "
        f"${en_reserva['media']['MAE']:,.0f})."
    )
    # Intervalo de predicción: se calibra con los errores fuera de partición de la parte de elección (nunca con la
    # reserva) y la reserva mide si cubre lo prometido, en total y por tramo.
    fuera_de_particion = g.prediccion_fuera_de_particion(candidatos[mejor], X, y, folds=args.folds)
    calibrado = g.calibrar_intervalo(y, fuera_de_particion)
    intervalo = g.cobertura(y_reserva, en_reserva_pred[mejor], calibrado)
    print(
        f"Intervalo de predicción al {intervalo['nivel']:.0%}: [estimado ÷ f, estimado × f], con f entre "
        f"{intervalo['factor_min']:.2f} y {intervalo['factor_max']:.2f} según el tramo (calibrado fuera de "
        f"partición); en la reserva cubre el {intervalo['cobertura']:.1%} de los salarios reales."
    )
    tramos = g.error_por_tramo(y_reserva, en_reserva_pred[mejor], intervalo=calibrado)
    print("\n=== Error y cobertura en la reserva por tramo de salario predicho (quintiles; agregados) ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(tramos.to_string())

    # Asociados nuevos (con vacíos): el mejor y la referencia, entrenados sin la reserva. Salen del generador: con datos
    # reales serían de otro origen, así que solo se usan en modo sintético.
    en_nuevos = {}
    if es_experimental():
        nuevos = asociados_nuevos(args.nuevos, args.semilla_nuevos, faltantes=FALTANTES)
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
    veredicto = "le gana al promedio" if gana else "NO le gana al promedio"
    if es_experimental():  # sintéticos: un punto por asociado (predicción fuera de partición)
        prediccion = fuera_de_particion
        titulo = (
            f"Salario real vs. predicho (fuera de partición) · {g.TITULOS[mejor]} · "
            f"R² en la reserva {en_reserva[mejor]['R2']:.3f} · {veredicto}"
        )
        fig = g.figura_real_vs_predicho(y, prediccion, titulo)
    else:  # datos reales: solo agregados por tramo, ningún salario individual (docs/privacidad.md)
        titulo = (
            f"Salario por tramo en la reserva · {g.TITULOS[mejor]} · R² {en_reserva[mejor]['R2']:.3f} · {veredicto}"
        )
        fig = g.figura_por_tramo(tramos, titulo)
    marcar_origen(fig).write_html(figura, include_plotlyjs="cdn")
    params = {
        "datos": fuente,
        "filas": len(X),
        "filas_reserva": len(X_reserva),
        "objetivo": g.OBJETIVO,
        "folds": args.folds,
    }
    params |= r.parametros_datos(X)
    resultado = {
        **params,
        "supera_referencia": bool(gana),  # en la reserva: si el elegido le gana a predecir el promedio (MAE)
        "cv": tabla.reset_index(names="modelo").to_dict(orient="records"),  # la elección (optimista para el mejor)
        "reserva": en_reserva,
        "intervalo": intervalo,  # nivel prometido, cobertura en la reserva y factor multiplicativo
        "reserva_por_tramo": tramos.reset_index().to_dict(orient="records"),  # agregados por quintil
        "nuevos": en_nuevos,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"regresion_{stamp}.json"
    texto = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    metricas.write_text(texto, encoding="utf-8", newline="\n")
    print(f"\nFigura -> {figura} · métricas -> {metricas}")

    if args.sin_mlflow:
        return
    r.configurar(EXPERIMENTO)
    with r.corrida(
        "regresion_salario",
        params,
        artefactos=[metricas, figura],
        tags={"mejor": mejor, "supera_referencia": str(bool(gana)).lower()},
    ):
        r.registrar_datos(X, fuente, "entrenamiento", fuente, objetivo=g.OBJETIVO)  # solo quienes reportan salario
        for nombre, fila in tabla.iterrows():
            propias = r.con_prefijo("cv", fila.to_dict())
            if nombre in en_reserva:
                propias |= r.con_prefijo("reserva", en_reserva[nombre])
            if nombre == mejor:
                propias |= r.con_prefijo("intervalo", intervalo)
            if nombre in en_nuevos:
                propias |= r.con_prefijo("nuevos", en_nuevos[nombre])
            with r.corrida(nombre, {**params, "modelo": nombre}, propias, anidada=True):
                pass
    print(f"MLflow: experimento '{r.con_origen(EXPERIMENTO)}', corrida 'regresion_salario'. Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
