"""Elige el modelo campeón del abandono, su umbral y sus zonas, y lo registra en MLflow (make campeon).

Candidatos (``campeon.lista_candidatos``): Ridge, logística, Random Forest, HistGB y las técnicas que mejoraron
en ``make tecnicas`` (variables nuevas; indicador de vacíos si hay vacíos); con --optuna N, también Random
Forest y HistGB afinados con Optuna (N intentos, por PR-AUC) en estos mismos datos. Gana el de mejor PR-AUC en
validación cruzada repetida (5 particiones x 3 repeticiones) sobre la foto del asociado. Los umbrales salen de
las predicciones fuera de partición (ver ``campeon.py``).

Incertidumbre: PR-AUC ± desviación entre particiones e intervalo bootstrap; el campeón contra cada candidato
partición a partición (si el intervalo de la diferencia toca el 0, la elección no es concluyente); y la regla
de umbrales medida en validación anidada (sin el optimismo de elegir y medir sobre lo mismo). Después:

- asociados nuevos (misma tasa de retiro) y escenarios menos favorables (tasa 10 %, señal al 50 % y 25 %):
  cuánto se degrada cada candidato con su umbral, en vez de una cifra perfecta;
- KMeans con los mismos datos: perfil y tasa de retiro real de cada grupo;
- curva ROC y matriz de confusión del campeón con sus umbrales (fuera de partición);
- SHAP: qué variables mueven el puntaje de cada asociado nuevo (importancia global y la explicación del de
  mayor riesgo).

Guarda en models/ solo el campeón (abandono_campeon.joblib, abandono_kmeans.joblib y campeon.json). En MLflow
(experimento ``abandono``): una corrida "campeon" con una hija por candidato (cv_, nuevos_, esc_) y el
campeón en el registro de modelos con el alias ``campeon`` (models:/abandono@campeon).

Datos: --datos, o data/processed/asociados_desarrollo.parquet (make datos-desarrollo), o los mismos 5.000
generados en memoria (semilla 7, 10 % de retiro).

Origen de los datos (``ORIGEN_DATOS`` del .env o --origen): con **sintetico** (por defecto) los resultados son
EXPERIMENTALES y se validan además con asociados nuevos y escenarios del generador. Con **real_anonimizado** hay
que dar --datos; el archivo debe cumplir el contrato, los nuevos y los escenarios (sintéticos) se omiten para no
mezclar orígenes, y MLflow usa el experimento ``abandono_real`` y el modelo ``abandono_real``.

Uso:
    uv run python scripts/elegir_campeon.py                    # foto a 3 meses
    uv run python scripts/elegir_campeon.py --horizonte 12 --precision-min 0.9
    uv run python scripts/elegir_campeon.py --optuna 30         # + Random Forest y HistGB afinados
    uv run python scripts/elegir_campeon.py --origen real_anonimizado --datos data/raw/extracto_anon.parquet
    uv run python scripts/elegir_campeon.py --datos data/raw/supabase_asociados.parquet --sin-mlflow
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from proyecto_ml import abandono as a
from proyecto_ml import busqueda_optuna as bo
from proyecto_ml import campeon as c
from proyecto_ml import explicar as e
from proyecto_ml import registro as r
from proyecto_ml import validacion as v
from proyecto_ml.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ORIGENES,
    aviso_origen,
    es_experimental,
    origen_datos,
    temporalidad_confirmada,
)
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.figuras import marcar_origen
from proyecto_ml.generador import DESARROLLO, GeneradorAsociados


def _slug(escenario: v.Escenario) -> str:
    return f"esc_t{round(escenario.tasa_retiro * 100)}_s{round(escenario.fuerza_senal * 100)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, help="Por defecto los de desarrollo (5.000, semilla 7, 10 %%)")
    parser.add_argument(
        "--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses de la foto (0 = todos al corte)"
    )
    parser.add_argument("--folds", type=int, default=c.FOLDS)
    parser.add_argument("--repeticiones", type=int, default=c.REPETICIONES)
    parser.add_argument("--beta", type=float, default=c.BETA, help="F-beta del umbral de alerta (2 = F2)")
    parser.add_argument("--precision-min", type=float, default=c.PRECISION_MIN, help="Precisión de la zona 'se retira'")
    parser.add_argument("--nuevos", type=int, default=5000, help="Asociados nuevos (y por escenario)")
    parser.add_argument("--optuna", type=int, default=0, help="Intentos de Optuna para sumar RF y HistGB afinados")
    parser.add_argument("--semilla-nuevos", type=int, default=2026)
    parser.add_argument("--shap", type=int, default=100, help="Asociados nuevos a explicar con SHAP (0 = ninguno)")
    parser.add_argument("--semilla", type=int, default=42, help="De los modelos (bosques, boosting, KMeans)")
    parser.add_argument("--origen", choices=ORIGENES, help="Por defecto ORIGEN_DATOS del .env, o sintetico")
    parser.add_argument("--sin-mlflow", action="store_true", help="No registrar en MLflow")
    args = parser.parse_args()
    if args.origen:
        os.environ["ORIGEN_DATOS"] = args.origen  # lo leen datos_eleccion, MLflow y los reportes
    origen, experimental = origen_datos(), es_experimental()

    df, fuente = c.datos_eleccion(args.datos)
    h = args.horizonte or None
    if h:
        df = a.foto(df, h)
    minimo = int(df[a.TARGET].value_counts().reindex([0, 1], fill_value=0).min())
    if minimo < args.folds:
        raise SystemExit(f"La clase minoritaria tiene {minimo} asociados (mínimo {args.folds}): usa más filas.")
    t0 = a.fecha_observacion(h).date() if h else a.FECHA_CORTE.date()
    print(
        f"Datos ({origen}): {fuente} · foto a {h or 0} meses (t0 = {t0}) · {len(df)} asociados, "
        f"{df[a.TARGET].mean():.1%} retiro"
    )
    print(f"Validación cruzada: {args.folds} particiones x {args.repeticiones} repeticiones · criterio {c.CRITERIO}")
    if experimental:
        print(aviso_origen() + "\n")
    else:
        print(aviso_origen())
        if temporalidad_confirmada():
            print("Temporalidad confirmada (TEMPORALIDAD_CONFIRMADA=si): los conteos se miden a t0.\n")
        else:
            print(
                "OJO: TEMPORALIDAD SIN CONFIRMAR. Los conteos de comportamiento no traen su fecha de medición;\n"
                "si son a la fecha de extracción y no a t0, para los retirados incluyen lo que pasó después (fuga).\n"
                "La evaluación queda marcada como no válida hasta que el fondo lo confirme\n"
                "(TEMPORALIDAD_CONFIRMADA=si en el .env).\n"
            )

    X, y = a.separar(df)
    candidatos = c.lista_candidatos(X, random_state=args.semilla)
    if args.optuna:  # afinados con estos mismos datos: su PR-AUC de CV es algo optimista (los eligió mirándolos)
        for nombre in bo.MODELOS:
            estudio = bo.afinar_optuna(
                nombre, X, y, n_trials=args.optuna, folds=args.folds, metrica="average_precision", seed=0
            )
            candidatos[f"{nombre}_optuna"] = bo.mejor_modelo(estudio, nombre)
            print(f"Optuna {nombre}: {args.optuna} intentos, mejor PR-AUC {estudio.best_value:.3f}")
    campeon = c.elegir(
        df,
        candidatos=candidatos,
        folds=args.folds,
        repeticiones=args.repeticiones,
        beta=args.beta,
        precision_min=args.precision_min,
        info={
            "datos": fuente,
            "horizonte_meses": h,
            "fecha_observacion": str(t0),
            "fecha_corte": str(a.FECHA_CORTE.date()),  # FECHA_CORTE efectiva (.env o la de los sintéticos)
            # con datos reales, la evaluación vale solo si el fondo confirmó que los conteos son a t0
            "temporalidad_confirmada": None if experimental else temporalidad_confirmada(),
            **r.parametros_datos(df),
            "experimental": experimental,
            "semilla": args.semilla,
            "validacion": f"StratifiedKFold({args.folds}) x {args.repeticiones} repeticiones (semilla = repetición)",
            "variables": a.NUMERICAS + a.CATEGORICAS,
            "versiones": {k: v for k, v in r.contexto_reproducible().items() if k.startswith("version.")},
        },
    )
    print("=== Candidatos (fuera de partición; umbral = máximo F2, umbral_alto = precisión mínima) ===")
    columnas = ["pr_auc", "pr_auc_std", "pr_auc_ic_inf", "pr_auc_ic_sup", "roc_auc", "umbral", "precision"]
    columnas += ["recall", "umbral_alto", "precision_alto", "recall_alto", "mcc", "kappa"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(campeon.tabla().reindex(columns=columnas).round(3).to_string())
    alto = "sin zona 'se retira'" if campeon.umbral_alto is None else f"{campeon.umbral_alto:.3f}"
    cv = campeon.candidatos[campeon.nombre].metricas
    print(f"\nCampeón: {c.TITULOS[campeon.nombre]} · umbral {campeon.umbral:.3f} · umbral_alto {alto}")
    print(
        f"PR-AUC {cv['pr_auc']:.3f} ± {cv['pr_auc_std']:.3f} entre particiones "
        f"(IC 95 % bootstrap {cv['pr_auc_ic_inf']:.3f}–{cv['pr_auc_ic_sup']:.3f}); "
        f"entre repeticiones solo ± {cv['pr_auc_std_repeticiones']:.3f}"
    )
    print("\n=== Campeón − cada candidato, partición a partición (t corregida de Nadeau y Bengio) ===")
    print(campeon.comparacion().round(3).to_string())
    empatados = campeon.empatados()
    if empatados:
        nombres = ", ".join(c.TITULOS.get(n, n) for n in empatados)
        print(f"NO concluyente frente a: {nombres} (el intervalo de la diferencia incluye el 0)")
    else:
        print("Concluyente: el campeón supera a todos los candidatos (ningún intervalo toca el 0)")
    if campeon.honestos:
        hon = campeon.honestos
        print(
            f"\n=== Umbrales en validación anidada (elegidos solo con el entrenamiento de cada partición) ===\n"
            f"alerta: precisión {hon['precision']:.2f}, recall {hon['recall']:.2f} (fuera de partición: "
            f"{cv['precision']:.2f} / {cv['recall']:.2f})\n"
            f"se retira: precisión {hon['precision_alto']:.2f} (pedida {args.precision_min:.2f}), "
            f"recall {hon['recall_alto']:.2f}, {hon['alertas_alto']} alertas; "
            f"{hon['particiones_sin_alto']} particiones sin zona 'se retira'"
        )

    # Todos los candidatos entrenados con todo, cada uno con su umbral, para nuevos y escenarios.
    entrenados = {n: clone(cand.modelo).fit(X, y) for n, cand in campeon.candidatos.items()}
    umbrales = {n: cand.umbrales["umbral"] for n, cand in campeon.candidatos.items()}
    nuevos, en_nuevos, por_zona, resultados, escenarios = None, {}, None, None, None
    if experimental:  # nuevos y escenarios salen del generador: solo tienen sentido con datos sintéticos
        nuevos = GeneradorAsociados(args.nuevos, seed=args.semilla_nuevos, tasa_retiro=DESARROLLO["tasa_retiro"])
        nuevos = nuevos.generar()
        nuevos = a.foto(nuevos, h) if h else nuevos
        Xn, yn = a.separar(nuevos)
        en_nuevos = {
            n: compute_metrics(yn, a.puntaje(m, Xn), threshold=umbrales[n], metodo=a.metodo_puntaje(m))
            for n, m in entrenados.items()
        }
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
    else:
        print("\nDatos reales: sin asociados nuevos ni escenarios del generador (serían de otro origen). La validación")
        print("de afuera es la temporal: validacion.particion_temporal (foto vieja -> foto reciente).")

    print("\n=== KMeans (sin objetivo) en los mismos datos ===")
    print(pd.DataFrame({g: {k: d[k] for k in ("asociados", "tasa_retiro")} for g, d in campeon.grupos.items()}))

    print("\n=== Matriz de confusión del campeón (fuera de partición; filas = real, columnas = predicción) ===")
    for nombre, cm in c.matrices_confusion(campeon).items():
        print(f"Umbral {nombre}:")
        print(pd.DataFrame(cm, index=["real 0", "real 1"], columns=["pred 0", "pred 1"]).to_string())

    explicaciones = None
    if args.shap:
        modelo = campeon.con_zonas()
        explicador = e.Explicador(modelo, X)
        base_shap = Xn if experimental else X  # con datos reales, una muestra de los mismos datos de elección
        muestra = base_shap.sample(min(args.shap, len(base_shap)), random_state=0)
        importancia_shap = e.importancia(explicador.contribuciones(muestra))
        mayor_riesgo = muestra.iloc[[int(a.puntaje(modelo, muestra).argmax())]]
        explicaciones = (explicador, importancia_shap, mayor_riesgo)
        donde = "asociados nuevos" if experimental else "asociados de elección"
        print(f"\n=== SHAP: |contribución| promedio en {len(muestra)} {donde} (top 8) ===")
        print(importancia_shap.head(8).round(4).to_string())

    # ---- Guardar: en models/ solo el campeón
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    figura = FIGURES_DIR / "campeon_umbrales.html"
    marcar_origen(c.figura_umbrales(campeon)).write_html(figura, include_plotlyjs="cdn")
    tabla_escenarios = FIGURES_DIR / "campeon_escenarios.html"
    if escenarios is not None:
        tabla_escenarios.write_text(escenarios.round(3).to_html(), encoding="utf-8", newline="\n")
    figura_roc = FIGURES_DIR / "campeon_roc.html"
    marcar_origen(c.figura_roc(campeon)).write_html(figura_roc, include_plotlyjs="cdn")
    figura_confusion = FIGURES_DIR / "campeon_confusion.html"
    marcar_origen(c.figura_confusion(campeon)).write_html(figura_confusion, include_plotlyjs="cdn")
    figuras = [figura, *([tabla_escenarios] if escenarios is not None else []), figura_roc, figura_confusion]
    calibracion = c.figura_calibracion(campeon)
    if calibracion is not None:  # solo si el puntaje es una probabilidad (no en Ridge)
        figura_calibracion = FIGURES_DIR / "campeon_calibracion.html"
        marcar_origen(calibracion).write_html(figura_calibracion, include_plotlyjs="cdn")
        figuras.append(figura_calibracion)
    if explicaciones:
        explicador, importancia_shap, mayor_riesgo = explicaciones
        aviso = " · DATOS SINTÉTICOS: explicación experimental" if experimental else ""
        titulo = f"SHAP del campeón ({c.TITULOS[campeon.nombre]}) en {donde}{aviso}"
        global_ = e.figura_importancia(importancia_shap, campeon.metodo, titulo)
        local = e.figura_explicacion(explicador.explicar(mayor_riesgo), explicador.base, campeon.metodo)
        local.update_layout(title="El asociado nuevo de mayor riesgo: de dónde sale su puntaje")
        figura_shap = FIGURES_DIR / "campeon_shap.html"
        partes = [
            marcar_origen(global_).to_html(full_html=False, include_plotlyjs="cdn"),
            marcar_origen(local).to_html(full_html=False),
        ]
        pagina = '<!doctype html><meta charset="utf-8"><title>SHAP del campeón</title>' + "".join(partes)
        figura_shap.write_text(pagina + "\n", encoding="utf-8", newline="\n")
        figuras.append(figura_shap)
    extra = {}
    if experimental:
        extra = {
            "nuevos": {"asociados": len(nuevos), **en_nuevos[campeon.nombre]},
            "zonas_nuevos": por_zona.reset_index().to_dict(orient="records"),
            "escenarios": resultados.to_dict(orient="records"),
        }
    if explicaciones:
        extra["shap_nuevos"] = {k: float(val) for k, val in explicaciones[1].items()}
    rutas = c.guardar(campeon, MODELS_DIR, extra)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metricas = METRICS_DIR / f"campeon_{stamp}.json"
    metricas.write_text(rutas["info"].read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"\nCampeón -> {rutas['modelo']} · {rutas['info']} · KMeans -> {rutas['grupos']}")
    print(f"Métricas -> {metricas} · figuras en {FIGURES_DIR}: {', '.join(f.name for f in figuras)}")

    if args.sin_mlflow:
        return
    r.configurar()
    params = {"datos": fuente, "filas": len(df), "horizonte_meses": h, "folds": args.folds}
    params |= {"repeticiones": args.repeticiones, "beta": args.beta, "precision_min": args.precision_min}
    params |= r.parametros_datos(df)
    params |= {"semilla": args.semilla, "validacion": campeon.info["validacion"], "experimental": experimental}
    params["variables"] = ",".join(campeon.info["variables"])
    slugs = {e.nombre: _slug(e) for e in v.ESCENARIOS}
    por_escenario = {
        n: {
            f"{slugs[f.escenario]}_{m}": getattr(f, m)
            for f in resultados[resultados["modelo"] == n].itertuples()
            for m in v.METRICAS_ESCENARIO[1:]
        }
        for n in entrenados
        if resultados is not None
    }
    metricas_mlflow = r.unir(
        {n: r.con_prefijo("cv", cand.metricas) for n, cand in campeon.candidatos.items()},
        {n: r.con_prefijo("nuevos", m) for n, m in en_nuevos.items()},
        por_escenario,
    )
    artefactos = [rutas["info"], *figuras]
    with r.corrida("campeon", params, artefactos=artefactos, tags={"campeon": campeon.nombre}):
        r.registrar_datos(df, fuente, "eleccion", fuente)  # versión de los datos (huella, sin copiarlos)
        if nuevos is not None:
            r.registrar_datos(nuevos, "nuevos", "nuevos", f"GeneradorAsociados(seed={args.semilla_nuevos})")
        r.registrar_modelos(entrenados, params, metricas_mlflow, titulos=c.TITULOS, guardar_modelo=False)
        propios = {**params, "modelo": campeon.nombre, "umbral": campeon.umbral, "umbral_alto": campeon.umbral_alto}
        with r.corrida(f"campeon_{campeon.nombre}", propios, metricas_mlflow[campeon.nombre], anidada=True):
            info = r.registrar_modelo(campeon.con_zonas(), registrar_como=r.modelo_registrado())
    r.asignar_alias(info.registered_model_version)
    print(
        f"MLflow: experimento '{r.con_origen(r.EXPERIMENTO)}', modelo models:/{r.modelo_registrado()}@{r.ALIAS_CAMPEON}"
    )
    print("        (versión " + str(info.registered_model_version) + "). Ver con: make mlflow-ui")


if __name__ == "__main__":
    main()
