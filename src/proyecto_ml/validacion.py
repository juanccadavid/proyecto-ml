"""Validación más allá de los "nuevos" del generador: escenarios menos favorables y partición temporal.

Las métricas sobre asociados nuevos generados miden si el modelo recupera la regla del generador, no si
funciona en el fondo real (validación circular). Mientras llega el extracto, dos cosas la hacen defendible:

- **Escenarios** (``evaluar_escenarios``): validar con asociados nuevos con otra tasa de retiro y una señal
  más débil (``GeneradorAsociados(fuerza_senal=...)``). En vez de una cifra perfecta, muestra cuánto se
  degrada el modelo si el fondo real no se parece al generador.
- **Partición temporal sintética** (``particion_temporal``): prueba el mecanismo con la etiqueta creada por el
  generador. No convierte el único extracto real en una serie temporal ni fabrica seguimiento futuro.
- **Estrategias de validación cruzada** (``comparar_validaciones``): KFold, StratifiedKFold y TimeSeriesSplit
  sobre los mismos datos, para ver por qué el campeón usa la estratificada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import GeneradorAsociados

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Escenario:
    nombre: str
    tasa_retiro: float
    fuerza_senal: float


# La tasa del generador (30 %) infla la precisión; un fondo real está más cerca del 10 %. La señal al
# 50 % y al 25 % simula que los retirados reales se diferencian menos que en el generador.
ESCENARIOS = (
    Escenario("tasa 30 %, señal 100 % (generador)", 0.30, 1.0),
    Escenario("tasa 10 %, señal 100 %", 0.10, 1.0),
    Escenario("tasa 10 %, señal 50 %", 0.10, 0.5),
    Escenario("tasa 10 %, señal 25 %", 0.10, 0.25),
)
METRICAS_ESCENARIO = ["base_rate", "roc_auc", "pr_auc", "precision", "recall"]


def datos_escenario(escenario: Escenario, n: int = 1000, seed: int = 2026, horizonte: int | None = None):
    """Asociados nuevos del escenario. Con ``horizonte``, su foto (como el entrenamiento)."""
    df = GeneradorAsociados(n, seed=seed, tasa_retiro=escenario.tasa_retiro, fuerza_senal=escenario.fuerza_senal)
    df = df.generar()
    return a.foto(df, horizonte) if horizonte else df


def evaluar_escenarios(
    modelos: dict[str, Pipeline],
    umbrales: dict[str, float] | None = None,
    escenarios=ESCENARIOS,
    n: int = 1000,
    seed: int = 2026,
    horizonte: int | None = None,
) -> pd.DataFrame:
    """Métricas de cada modelo ya entrenado en cada escenario: una fila por (escenario, modelo).

    ``umbrales`` da el umbral de cada modelo (el del campeón, por ejemplo); si falta, el de siempre
    (``a.UMBRAL``: 0,5 en probabilidad, 0 en Ridge). Los mismos asociados para todos los modelos.
    """
    umbrales = umbrales or {}
    filas = []
    for escenario in escenarios:
        df = datos_escenario(escenario, n, seed, horizonte)
        X, y = a.separar(df)
        for nombre, modelo in modelos.items():
            umbral = umbrales.get(nombre, a.UMBRAL[a.metodo_puntaje(modelo)])
            m = compute_metrics(y, a.puntaje(modelo, X), threshold=umbral, metodo=a.metodo_puntaje(modelo))
            filas.append({"escenario": escenario.nombre, "modelo": nombre, **{k: m[k] for k in METRICAS_ESCENARIO}})
    return pd.DataFrame(filas)


def tabla_escenarios(resultados: pd.DataFrame, metricas=("roc_auc", "precision", "recall")) -> pd.DataFrame:
    """Como la tabla de la presentación: filas = escenario, columnas = (modelo, métrica)."""
    orden = list(dict.fromkeys(resultados["escenario"]))
    tabla = resultados.pivot_table(index="escenario", columns="modelo", values=list(metricas)).reindex(orden)
    modelos = list(dict.fromkeys(resultados["modelo"]))
    return tabla.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([modelos, list(metricas)]))


# ---------------------------------------------------------------- partición temporal (extracto real)


def particion_temporal(
    df: pd.DataFrame,
    horizonte_meses: int = a.HORIZONTE_MESES,
    meses_prueba: int = 3,
    corte: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento = foto vieja (t0 = corte − ``meses_prueba``), prueba = foto del corte (t0 = corte).

    Con varios extractos, cada foto tendría sus variables medidas en su t0 y su evento en (t0, t0 + H]. Con UN
    solo extracto las dos fotos son las mismas filas con las mismas variables y la misma etiqueta sintética: solo
    prueba el mecanismo (fechas, orden), no es una validación fuera de tiempo. Con el extracto real no aplica:
    no hay evento observado después del corte.
    """
    if meses_prueba < 1:
        raise ValueError("meses_prueba debe ser al menos 1")
    corte_ts = a.fecha_observacion(corte)
    entrenamiento = a.foto(df, horizonte_meses, corte_ts - pd.DateOffset(months=meses_prueba))
    prueba = a.foto(df, horizonte_meses, corte_ts)
    return entrenamiento, prueba


# ---------------------------------------------------------------- estrategias de validación cruzada

# TimeSeriesSplit necesita las filas en orden de tiempo. La foto no tiene una fecha por fila (todos se observan
# en t0), así que se ordena por fecha de ingreso: entrena con los asociados más antiguos y prueba con los
# más nuevos (cohortes). Con un solo corte real no hay validación temporal de verdad: ``particion_temporal`` solo prueba
# el mecanismo con la etiqueta del generador (ver docs/datos_reales.md).
ORDEN_TEMPORAL = "FECHAINGRESOLABORA"


def estrategias_cv(folds: int = 5, random_state: int = 0) -> dict:
    """Las tres formas de partir: KFold (al azar), StratifiedKFold (misma tasa de retiro en cada partición)
    y TimeSeriesSplit (cada partición prueba con filas posteriores a todas las de su entrenamiento)."""
    return {
        "KFold": KFold(folds, shuffle=True, random_state=random_state),
        "StratifiedKFold": StratifiedKFold(folds, shuffle=True, random_state=random_state),
        "TimeSeriesSplit": TimeSeriesSplit(folds),
    }


def _puntuar(y, puntaje, metrica) -> float:
    return float(metrica(y, puntaje)) if 0 < y.sum() < len(y) else float("nan")  # sin retirados no se define


def comparar_validaciones(
    modelo: Pipeline, df: pd.DataFrame, folds: int = 5, random_state: int = 0, orden: str = ORDEN_TEMPORAL
) -> pd.DataFrame:
    """Una fila por estrategia y partición: tamaños, retirados en la prueba, PR-AUC y ROC-AUC.

    ``df`` son filas crudas con el objetivo (p. ej. ``a.foto(...)``); se ordenan por ``orden`` para
    TimeSeriesSplit (KFold y StratifiedKFold barajan, así que el orden no les cambia nada).
    """
    df = df.sort_values(orden, kind="stable").reset_index(drop=True)
    X, y = a.separar(df)
    filas = []
    for estrategia, cv in estrategias_cv(folds, random_state).items():
        for k, (entrena, prueba) in enumerate(cv.split(X, y), start=1):
            y_entrena, y_prueba = y.iloc[entrena], y.iloc[prueba]
            fila = {
                "estrategia": estrategia,
                "particion": k,
                "filas_entrenamiento": len(entrena),
                "filas_prueba": len(prueba),
                "retirados_prueba": int(y_prueba.sum()),
                "tasa_prueba": float(y_prueba.mean()),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
            }
            if y_entrena.nunique() == 2:  # sin las dos clases en el entrenamiento no hay modelo
                puntaje = a.puntaje(clone(modelo).fit(X.iloc[entrena], y_entrena), X.iloc[prueba])
                fila["pr_auc"] = _puntuar(y_prueba, puntaje, average_precision_score)
                fila["roc_auc"] = _puntuar(y_prueba, puntaje, roc_auc_score)
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_validaciones(por_particion: pd.DataFrame) -> pd.DataFrame:
    """Por estrategia: PR-AUC media, su desviación entre particiones y cuántos retirados tuvo cada prueba."""
    g = por_particion.groupby("estrategia", sort=False)
    return pd.DataFrame(
        {
            "pr_auc": g["pr_auc"].mean(),
            "pr_auc_std": g["pr_auc"].std(),
            "roc_auc": g["roc_auc"].mean(),
            "retirados_min": g["retirados_prueba"].min(),
            "retirados_max": g["retirados_prueba"].max(),
            "tasa_prueba_std": g["tasa_prueba"].std(),
            "entrenamiento_medio": g["filas_entrenamiento"].mean().round().astype(int),
        }
    )


def figura_validaciones(por_particion: pd.DataFrame, titulo: str):
    """Plotly: la PR-AUC de cada partición (puntos) y su promedio (raya), una fila por estrategia."""
    import plotly.graph_objects as go

    colores = {"KFold": "#2a78d6", "StratifiedKFold": "#eb6834", "TimeSeriesSplit": "#1baf7a"}
    fig = go.Figure()
    for estrategia, datos in por_particion.groupby("estrategia", sort=False):
        color = colores.get(estrategia, "#898781")
        fig.add_scatter(
            x=datos["pr_auc"],
            y=[estrategia] * len(datos),
            mode="markers",
            name=estrategia,
            marker={"size": 11, "color": color, "line": {"color": "#fcfcfb", "width": 2}},
            customdata=np.column_stack([datos["particion"], datos["retirados_prueba"], datos["filas_prueba"]]),
            hovertemplate=f"{estrategia} · partición %{{customdata[0]}}<br>PR-AUC=%{{x:.3f}}"
            "<br>%{customdata[1]} retirados de %{customdata[2]}<extra></extra>",
        )
        fig.add_scatter(
            x=[datos["pr_auc"].mean()],
            y=[estrategia],
            mode="markers",
            showlegend=False,
            marker={"symbol": "line-ns", "size": 26, "line": {"color": "#0b0b0b", "width": 2}},
            hovertemplate=f"{estrategia}: promedio %{{x:.3f}}<extra></extra>",
        )
    fig.update_layout(
        title=titulo,
        xaxis={"title": "PR-AUC de la partición (raya negra = promedio)", "gridcolor": "#e1e0d9"},
        yaxis={"autorange": "reversed"},
        template="plotly_white",
        height=360,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


# ---------------------------------------------------------------- desplazamiento: sintéticos frente a reales


def _normalizar_dominio(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la misma limpieza determinista a ambos orígenes, sin aceptar filas deduplicadas en silencio."""
    from proyecto_ml import contrato

    limpio = contrato.limpiar_extracto(df)
    if len(limpio) != len(df):
        raise ValueError("la normalización eliminaría registros duplicados; resuélvelos antes de comparar los dominios")
    return limpio


def desplazamiento(referencia: pd.DataFrame, lote: pd.DataFrame, folds: int = 5, random_state: int = 42) -> dict:
    """¿Se parece ``lote`` (p. ej. el extracto real) a ``referencia`` (los sintéticos con que se entrenó el campeón)?

    Validación adversaria: un clasificador intenta distinguir las filas de un origen (0) y del otro (1) usando
    únicamente las variables predictoras del modelo (sin identificadores ni columnas de fuga) y el mismo
    preprocesamiento del campeón ajustado dentro de cada partición.

    - ROC-AUC fuera de partición cerca de 0,5 = distribuciones marginales/conjuntas ``P(X)`` indistinguibles.
      Ojo: **un ROC-AUC cercano a 0,5 NO valida que el clasificador de abandono acierte en datos reales**, porque
      solo compara ``P(X)`` y no la relación condicional ``P(retiro_3m | X)``.
    - Cerca de 1,0 = ``lote`` está fuera del soporte de entrenamiento y el puntaje extrapola.
    - No pliega ``max(auc, 1 - auc)``: un AUC < 0,5 fuera de partición es ruido muestral, no señal de separación.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    referencia, lote = _normalizar_dominio(referencia), _normalizar_dominio(lote)
    llave = a.IDENTIFICADORES[0]
    if llave in referencia and referencia[llave].duplicated().any():
        raise ValueError(f"referencia tiene asociados duplicados en {llave}")
    if llave in lote and lote[llave].duplicated().any():
        raise ValueError(f"lote tiene asociados duplicados en {llave}")
    if llave in referencia and llave in lote and (set(referencia[llave]) & set(lote[llave])):
        raise ValueError(f"referencia y lote comparten identificadores en {llave}")

    X = pd.concat([a.preparar(referencia), a.preparar(lote)], ignore_index=True)
    y = np.r_[np.zeros(len(referencia)), np.ones(len(lote))]
    cv = StratifiedKFold(folds, shuffle=True, random_state=random_state)

    def auc_cv(columnas: list[str]) -> tuple[float, float]:
        numericas = [c for c in columnas if c in a.NUMERICAS]
        categoricas = [c for c in columnas if c in a.CATEGORICAS]
        con_recorte = any(c in a.COLUMNAS_RECORTE for c in numericas)
        modelo = a.armar_pipeline(
            HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
            numeric=numericas,
            categorical=categoricas,
            clip_outliers=a.CLIP_OUTLIERS if con_recorte else None,
            clip_columns=[col for col in a.COLUMNAS_RECORTE if col in numericas],
            max_nulos=a.MAX_NULOS,
            random_state=random_state,
        )
        puntajes = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")
        return float(puntajes.mean()), float(puntajes.std())

    columnas = a.NUMERICAS + a.CATEGORICAS
    senales = [*a.COMPORTAMIENTO, "#SOLICITUD_DISMINUCIONCUOTAS", a.RETIROS_HISTORICOS]
    constantes = {c for c in columnas if (X[c].fillna("__vacio__") == X[c].fillna("__vacio__").iloc[0]).all()}
    por_variable = {c: auc_cv([c])[0] for c in columnas if c not in constantes}
    auc_todas, std_todas = auc_cv(columnas)
    auc_comp, std_comp = auc_cv(a.COMPORTAMIENTO)
    auc_sen, std_sen = auc_cv(senales)
    return {
        "filas_referencia": len(referencia),
        "filas_lote": len(lote),
        "roc_auc": auc_todas,
        "roc_auc_std": std_todas,
        "roc_auc_comportamiento": auc_comp,
        "roc_auc_comportamiento_std": std_comp,
        "roc_auc_senales": auc_sen,
        "roc_auc_senales_std": std_sen,
        "por_variable": dict(sorted(por_variable.items(), key=lambda kv: -abs(kv[1] - 0.5))),
    }


def comparar_distribuciones(referencia: pd.DataFrame, lote: pd.DataFrame) -> pd.DataFrame:
    """Variable por variable del modelo, cómo difiere ``lote`` (el extracto real) de ``referencia`` (los sintéticos).
    Solo agregados, para ver QUÉ hace que la validación adversaria los distinga:

    - numéricas: percentiles 1, 5, 50, 95 y 99 de cada origen, % de ceros, proporción del lote fuera de los
      percentiles 1–99 de referencia y Kolmogórov-Smirnov (``ks``: 0 = iguales, 1 = sin solapamiento);
    - categóricas: distancia de variación total (``ks`` en la tabla: mitad de la suma de |p − q| por categoría) y
      ``fuera_de_soporte``: % de filas del lote con una categoría que la referencia no tiene (el modelo nunca la vio).
    """
    from scipy.stats import ks_2samp

    R, L = a.preparar(_normalizar_dominio(referencia)), a.preparar(_normalizar_dominio(lote))
    filas = []
    for col in a.NUMERICAS:
        r, l_ = R[col].dropna(), L[col].dropna()
        ref_p01, ref_p99 = r.quantile([0.01, 0.99])
        filas.append(
            {
                "variable": col,
                "tipo": "numérica",
                "ref_p01": ref_p01,
                "ref_p5": r.quantile(0.05),
                "ref_p50": r.median(),
                "ref_p95": r.quantile(0.95),
                "ref_p99": ref_p99,
                "lote_p01": l_.quantile(0.01),
                "lote_p5": l_.quantile(0.05),
                "lote_p50": l_.median(),
                "lote_p95": l_.quantile(0.95),
                "lote_p99": l_.quantile(0.99),
                "ref_ceros": float((r == 0).mean()),
                "lote_ceros": float((l_ == 0).mean()),
                "ks": float(ks_2samp(r, l_).statistic),
                "lote_fuera_ref_p01_p99": float(((l_ < ref_p01) | (l_ > ref_p99)).mean()),
                "fuera_de_soporte": float("nan"),
            }
        )
    for col in a.CATEGORICAS:
        p = R[col].fillna("(vacío)").value_counts(normalize=True)
        q = L[col].fillna("(vacío)").value_counts(normalize=True)
        tvd = 0.5 * float(p.reindex(p.index.union(q.index), fill_value=0).sub(q, fill_value=0).abs().sum())
        filas.append(
            {
                "variable": col,
                "tipo": "categórica",
                "ks": tvd,
                "fuera_de_soporte": float((~L[col].fillna("(vacío)").isin(p.index)).mean()),
            }
        )
    return pd.DataFrame(filas).set_index("variable").sort_values("ks", ascending=False)
