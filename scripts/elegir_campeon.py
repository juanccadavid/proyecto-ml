"""Elige el modelo campeón del abandono, su umbral y sus zonas, y lo registra en MLflow (make campeon).

Candidatos: Ridge, logística, Random Forest y HistGB (``abandono.modelos_supervisados``). Gana el de mejor
PR-AUC en validación cruzada repetida (5 particiones x 3 repeticiones) sobre la foto del asociado. Los
umbrales salen de las predicciones fuera de partición (ver ``campeon.py``). Después:

- asociados nuevos (misma tasa de retiro) y escenarios menos favorables (tasa 10 %, señal al 50 % y 25 %):
  cuánto se degrada cada candidato con su umbral, en vez de una cifra perfecta;
- KMeans con los mismos datos: perfil y tasa de retiro real de cada grupo.

Guarda en models/ solo el campeón (abandono_campeon.joblib, abandono_kmeans.joblib y campeon.json). En MLflow
(experimento ``abandono``): una corrida "campeon" con una hija por candidato (cv_, nuevos_, esc_) y el
campeón en el registro de modelos con el alias ``campeon`` (models:/abandono@campeon).

Datos: --datos, o data/processed/asociados_desarrollo.parquet (make datos-desarrollo), o los mismos 2.000
generados en memoria (semilla 7, 10 % de retiro).

Uso:
    uv run python scripts/elegir_campeon.py                    # foto a 3 meses
    uv run python scripts/elegir_campeon.py --horizonte 12 --precision-min 0.9
    uv run python scripts/elegir_campeon.py --datos data/raw/supabase_asociados.parquet --sin-mlflow
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import registro as r
from proyecto_ml import validacion as v
from proyecto_ml.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import DESARROLLO, GeneradorAsociados


def _slug(escenario: v.Escenario) -> str:
    return f"esc_t{round(escenario.tasa_retiro * 100)}_s{round(escenario.fuerza_senal * 100)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (2.000, semilla 7, 10 %%)")
    parser.add_argument(
        "--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses de la foto (0 = todos al corte)"
    )
    parser.add_argument("--folds", type=int, default=c.FOLDS)
    parser.add_argument("--repeticiones", type=int, default=c.REPETICIONES)
    parser.add_argument("--beta", type=float, default=c.BETA, help="F-beta del umbral de alerta (2 = F2)")
    parser.add_argument("--precision-min", type=float, default=c.PRECISION_MIN, help="Precisión de la zona 'se retira'")
    parser.add_argument("--nuevos", type=int, default=1000, help="Asociados nuevos (y por escenario)")
    parser.add_argument("--semilla-nuevos", type=int, default=2026)
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()

    df, fuente = c.datos_eleccion(args.datos)
    h = args.horizonte or None
    if h:
        df = a.foto(df, h)
    minimo = int(df[a.TARGET].value_counts().reindex([0, 1], fill_value=0).min())
    if minimo < args.folds:
        raise SystemExit(f"La clase minoritaria tiene {minimo} asociados (mínimo {args.folds}): usa más filas.")
    t0 = a.fecha_observacion(h).date() if h else a.FECHA_CORTE.date()
    print(
        f"Datos: {fuente} · foto a {h or 0} meses (t0 = {t0}) · {len(df)} asociados, {df[a.TARGET].mean():.1%} retiro"
    )
    print(f"Validación cruzada: {args.folds} particiones x {args.repeticiones} repeticiones · criterio {c.CRITERIO}\n")

    campeon = c.elegir(
        df,
        folds=args.folds,
        repeticiones=args.repeticiones,
        beta=args.beta,
        precision_min=args.precision_min,
        info={"datos": fuente, "horizonte_meses": h, "fecha_observacion": str(t0)},
    )
    print("=== Candidatos (fuera de partición; umbral = máximo F2, umbral_alto = precisión mínima) ===")
    print(campeon.tabla().round(3).to_string())
    alto = "sin zona 'se retira'" if campeon.umbral_alto is None else f"{campeon.umbral_alto:.3f}"
    print(f"\nCampeón: {a.TITULOS[campeon.nombre]} · umbral {campeon.umbral:.3f} · umbral_alto {alto}")

    # Todos los candidatos entrenados con todo, cada uno con su umbral, para nuevos y escenarios.
    X, y = a.separar(df)
    entrenados = {n: clone(cand.modelo).fit(X, y) for n, cand in campeon.candidatos.items()}
    umbrales = {n: cand.umbrales["umbral"] for n, cand in campeon.candidatos.items()}
    nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos, tasa_retiro=DESARROLLO["tasa_retiro"])
    nuevos = nuevos.generar()
    nuevos = a.foto(nuevos, h) if h else nuevos
    Xn, yn = a.separar(nuevos)
    en_nuevos = {n: compute_metrics(yn, a.puntaje(m, Xn), threshold=umbrales[n]) for n, m in entrenados.items()}
    print(f"\n=== Asociados nuevos ({len(nuevos)}, {yn.mean():.1%} retiro), cada uno con su umbral ===")
    print(pd.DataFrame(en_nuevos).T.round(3).to_string())

    zonas = pd.Series(campeon.zona(Xn), index=nuevos.index, name="zona")
    por_zona = nuevos.groupby(zonas)[a.TARGET].agg(asociados="size", se_retiran="sum", tasa_retiro="mean")
    por_zona = por_zona.reindex([z for z in c.ZONAS if z in por_zona.index])
    print("\n=== Zonas del campeón sobre los nuevos ===")
    print(por_zona.round(3).to_string())

    resultados = v.evaluar_escenarios(entrenados, umbrales, n=args.nuevos, seed=args.semilla_nuevos, horizonte=h)
    escenarios = v.tabla_escenarios(resultados)
    print("\n=== Escenarios: ¿cuánto se degrada cada candidato si el fondo real no es como el generador? ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(escenarios.round(2).to_string())

    print("\n=== KMeans (sin objetivo) en los mismos datos ===")
    print(pd.DataFrame({g: {k: d[k] for k in ("asociados", "tasa_retiro")} for g, d in campeon.grupos.items()}))

    # ---- Guardar: en models/ solo el campeón
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figura = FIGURES_DIR / "campeon_umbrales.html"
    c.figura_umbrales(campeon).write_html(figura, include_plotlyjs="cdn")
    tabla_escenarios = FIGURES_DIR / "campeon_escenarios.html"
    tabla_escenarios.write_text(escenarios.round(3).to_html(), encoding="utf-8", newline="\n")
    extra = {
        "nuevos": {"asociados": len(nuevos), **en_nuevos[campeon.nombre]},
        "zonas_nuevos": por_zona.reset_index().to_dict(orient="records"),
        "escenarios": resultados.to_dict(orient="records"),
    }
    rutas = c.guardar(campeon, MODELS_DIR, extra)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"campeon_{stamp}.json"
    metricas.write_text(rutas["info"].read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"\nCampeón -> {rutas['modelo']} · {rutas['info']} · KMeans -> {rutas['grupos']}")
    print(f"Métricas -> {metricas} · umbrales -> {figura} · escenarios -> {tabla_escenarios}")

    if args.sin_mlflow:
        return
    r.configurar()
    params = {"datos": fuente, "filas": len(df), "horizonte_meses": h, "folds": args.folds}
    params |= {"repeticiones": args.repeticiones, "beta": args.beta, "precision_min": args.precision_min}
    slugs = {e.nombre: _slug(e) for e in v.ESCENARIOS}
    por_escenario = {
        n: {
            f"{slugs[f.escenario]}_{m}": getattr(f, m)
            for f in resultados[resultados["modelo"] == n].itertuples()
            for m in v.METRICAS_ESCENARIO[1:]
        }
        for n in entrenados
    }
    metricas_mlflow = r.unir(
        {n: r.con_prefijo("cv", cand.metricas) for n, cand in campeon.candidatos.items()},
        {n: r.con_prefijo("nuevos", m) for n, m in en_nuevos.items()},
        por_escenario,
    )
    artefactos = [rutas["info"], figura, tabla_escenarios]
    with r.corrida("campeon", params, artefactos=artefactos, tags={"campeon": campeon.nombre}):
        r.registrar_modelos(entrenados, params, metricas_mlflow, titulos=a.TITULOS, guardar_modelo=False)
        propios = {**params, "modelo": campeon.nombre, "umbral": campeon.umbral, "umbral_alto": campeon.umbral_alto}
        with r.corrida(f"campeon_{campeon.nombre}", propios, metricas_mlflow[campeon.nombre], anidada=True):
            info = r.registrar_modelo(campeon.con_zonas(), registrar_como=r.MODELO_REGISTRADO)
    r.asignar_alias(info.registered_model_version)
    print(f"MLflow: experimento '{r.EXPERIMENTO}', modelo models:/{r.MODELO_REGISTRADO}@{r.ALIAS_CAMPEON}")
    print("        (versión " + str(info.registered_model_version) + "). Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
