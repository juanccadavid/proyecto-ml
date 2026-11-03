"""Abandono de asociados del fondo de empleados: modelos supervisados y KMeans (sin objetivo).

Supervisados: Ridge, regresión logística, Random Forest (bagging) y HistGradientBoosting (boosting).
Todos usan las mismas variables, elegidas para no hacer trampa: se excluyen las columnas que
solo existen PORQUE el asociado ya se retiró (fecha y motivo de retiro, ESTADO, fechas de
actualización que coinciden con el retiro...). Con ellas cualquier modelo acierta el 100 %
pero no sirve para anticipar un retiro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_ml.cleaning import ColumnDropper
from proyecto_ml.config import fecha_corte
from proyecto_ml.contrato import limpiar_extracto
from proyecto_ml.data import load_dataset
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.features import Preprocesador, build_preprocessor
from proyecto_ml.models import get_models

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

TARGET = "#RETIROS"
FECHA_CORTE = pd.Timestamp(fecha_corte())  # FECHA_CORTE del .env; por defecto la de los sintéticos
# Foto del asociado: se observa en t0 = FECHA_CORTE - HORIZONTE_MESES y se predice si se retira entre t0 y el
# corte. La columna FECHA_OBSERVACION (la agrega ``foto``) le dice a ``preparar`` a qué fecha calcular.
HORIZONTE_MESES = 3
FECHA_OBSERVACION = "FECHA_OBSERVACION"

# Columnas que delatan el resultado (fuga de información): nunca entran a los modelos.
FUGA = [
    TARGET,
    "FECHARETIRO",
    "MOTIVORETIRO",
    "ESTADO",
    "ESTADO_LABORAL",
    "FECHAULTACTUALIZACION",
    "ULTIMAACTUALIZACION",
    "FECHATERMINACONTRATO",
]

# Comportamiento con el fondo en los últimos meses: la señal principal de abandono.
COMPORTAMIENTO = [
    "#SOLICITUD_DISMINUCIONCUOTAS",
    "#SOLICITUD_CANCELACIONAHORROS",
    "#SOLICITUD_RETIROPARCIALAHORROS",
    "#CANTIDAD_AHORROSALAVISTA",
    "#CANTIDAD_AHORROSCONTRACTUALES",
    "CANCELACIONCREDITOS_ULT3MESES",
    "#SOLICITUDCREDITOS_ULT3MESES",
    "#CANTIDAD_SERVICIOS",
    "#INTENCIONRETIROS_ULT3MESES",
]
NUMERICAS = [*COMPORTAMIENTO, "SALARIO", "ESTRATO", "EDAD", "ANTIGUEDAD"]
CATEGORICAS = ["SEXO", "TIPOCONTRATO", "TIPOPACTO", "PERIODODEDUCE", "TIPOCUENTA", "CARGO", "CODCIUDAD", "AGENCIA"]
# Las únicas que tiene sentido recortar (clip_outliers): continuas. Los conteos de COMPORTAMIENTO casi
# siempre valen 0, su IQR queda en [0, 0] y recortarlos borraría a quienes sí tienen solicitudes.
COLUMNAS_RECORTE = ["SALARIO", "EDAD", "ANTIGUEDAD"]

# Señales que, sin mirar el objetivo, indican intención de irse (+) o de quedarse (-).
# Se usan para decidir cuál de los dos grupos de KMeans es el de "riesgo".
SENAL_RIESGO = {
    "#SOLICITUD_DISMINUCIONCUOTAS": 1,
    "#SOLICITUD_CANCELACIONAHORROS": 1,
    "#SOLICITUD_RETIROPARCIALAHORROS": 1,
    "CANCELACIONCREDITOS_ULT3MESES": 1,
    "#INTENCIONRETIROS_ULT3MESES": 1,
    "#CANTIDAD_AHORROSCONTRACTUALES": -1,
    "#SOLICITUDCREDITOS_ULT3MESES": -1,
}


def fecha_observacion(horizonte_meses: int = HORIZONTE_MESES, corte: pd.Timestamp = FECHA_CORTE) -> pd.Timestamp:
    """t0: la fecha de la foto, ``horizonte_meses`` antes del corte."""
    return pd.Timestamp(corte) - pd.DateOffset(months=horizonte_meses)


def foto(df: pd.DataFrame, horizonte_meses: int = HORIZONTE_MESES, corte: pd.Timestamp = FECHA_CORTE) -> pd.DataFrame:
    """La foto del asociado en t0 = corte - horizonte: quién estaba activo y quién se retiró después.

    - Población: asociados activos en t0 (ingresaron a laborar antes de t0 y no se habían retirado).
      Los que se retiraron antes de t0 salen: en la foto ya no eran asociados.
    - Objetivo (#RETIROS): 1 si se retira entre t0 y el corte, 0 si sigue activo.
    - Agrega FECHA_OBSERVACION = t0, para que ``preparar`` calcule EDAD y ANTIGUEDAD a esa fecha.

    Usa FECHARETIRO, que el contrato exige si y solo si #RETIROS = 1 (regla retiro_coherente).
    Ojo: los conteos de solicitudes (ULT3MESES...) se toman como vienen. Si el extracto los mide a la
    fecha de extracción y no a t0, para los retirados incluyen lo que pasó después: fuga de información.
    """
    t0 = fecha_observacion(horizonte_meses, corte)
    retiro = pd.to_datetime(df["FECHARETIRO"])
    activos = (pd.to_datetime(df["FECHAINGRESOLABORA"]) <= t0) & (retiro.isna() | (retiro > t0))
    vista = df[activos].copy()
    vista[TARGET] = (retiro[activos] <= pd.Timestamp(corte)).astype(int)  # NaT <= corte es False: activo
    vista[FECHA_OBSERVACION] = t0
    return vista


def resumen_foto(df: pd.DataFrame, horizontes=(1, 3, 6, 12), corte: pd.Timestamp = FECHA_CORTE) -> pd.DataFrame:
    """Para el EDA: por horizonte, la fecha t0, cuántos asociados entran a la foto y cuántos se retiran."""
    filas = []
    for h in horizontes:
        vista = foto(df, h, corte)
        filas.append(
            {
                "horizonte_meses": h,
                "fecha_observacion": fecha_observacion(h, corte).date(),
                "asociados": len(vista),
                "se_retiran": int(vista[TARGET].sum()),
                "tasa_retiro": float(vista[TARGET].mean()) if len(vista) else float("nan"),
                "fuera_de_la_foto": len(df) - len(vista),  # retirados antes de t0 o ingresados después
            }
        )
    return pd.DataFrame(filas).set_index("horizonte_meses")


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Variables del modelo: edad y antigüedad (años a la fecha de observación) + columnas elegidas. Sin fuga ni PII.

    La fecha de observación es FECHA_OBSERVACION si ``df`` viene de ``foto``; si no, FECHA_CORTE.
    """
    X = pd.DataFrame(index=df.index)
    t0 = pd.to_datetime(df[FECHA_OBSERVACION]) if FECHA_OBSERVACION in df else FECHA_CORTE
    X["EDAD"] = (t0 - pd.to_datetime(df["FECHANACIMIENTO"])).dt.days / 365.25
    X["ANTIGUEDAD"] = (t0 - pd.to_datetime(df["FECHAINGRESOLABORA"])).dt.days / 365.25
    for c in NUMERICAS:
        if c not in X:
            X[c] = pd.to_numeric(df[c]).astype(float)  # Int64 con vacíos -> float con NaN (lo imputa el pipeline)
    for c in CATEGORICAS:
        # Los códigos (ciudad, agencia) son categorías, no cantidades. El vacío sigue vacío (no el texto "nan").
        X[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(v))
    return X[NUMERICAS + CATEGORICAS]


def separar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (las filas sin el objetivo, tal cual: el pipeline las prepara) e y (#RETIROS)."""
    return df.drop(columns=TARGET), df[TARGET]


class PrepararVariables(TransformerMixin, BaseEstimator):
    """``preparar`` como primer paso del pipeline: el modelo recibe la fila tal como llega (extracto, Supabase,
    formulario) y calcula él mismo EDAD y ANTIGUEDAD. Sin estado: no aprende nada de train.

    Si ``X`` ya viene preparada (no trae FECHANACIMIENTO), solo ordena sus columnas: así el mismo modelo
    sirve con ``preparar(df)`` (importancia por permutación, simulador, vacíos simulados).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return preparar(X) if "FECHANACIMIENTO" in X else X[NUMERICAS + CATEGORICAS]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERICAS + CATEGORICAS, dtype=object)


# Feature engineering (creación y transformación): variables nuevas a partir de las de ``preparar``.
SALIDA = [c for c, signo in SENAL_RIESGO.items() if signo > 0]  # solicitudes que anuncian un retiro
PRODUCTOS = ["#CANTIDAD_AHORROSALAVISTA", "#CANTIDAD_AHORROSCONTRACTUALES", "#CANTIDAD_SERVICIOS"]
VARIABLES_NUEVAS = ["SENALES_SALIDA", "PRODUCTOS", "SALIDA_POR_PRODUCTO", "SALARIO_LOG"]


class CrearVariables(TransformerMixin, BaseEstimator):
    """Agrega VARIABLES_NUEVAS a la salida de ``preparar`` (sin estado: no aprende nada de train).

    - SENALES_SALIDA: suma de las solicitudes de SALIDA (agregación).
    - PRODUCTOS: ahorros a la vista + contractuales + servicios (agregación).
    - SALIDA_POR_PRODUCTO: SENALES_SALIDA / (1 + PRODUCTOS): pedir salir teniendo poco que perder (interacción).
    - SALARIO_LOG: log(1 + SALARIO), que acerca a una normal un salario sesgado a la derecha (transformación).

    Un vacío en una de las columnas no vacía la suma (cuenta como 0); si todas están vacías, queda vacía.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["SENALES_SALIDA"] = X[SALIDA].sum(axis=1, min_count=1)
        X["PRODUCTOS"] = X[PRODUCTOS].sum(axis=1, min_count=1)
        X["SALIDA_POR_PRODUCTO"] = X["SENALES_SALIDA"] / (1 + X["PRODUCTOS"])
        X["SALARIO_LOG"] = np.log1p(X["SALARIO"])
        return X


# ---------------------------------------------------------------- supervisados

TITULOS = {
    "ridge": "Ridge",
    "logreg": "Regresión logística",
    "random_forest": "Random Forest (bagging)",
    "hist_gb": "HistGradientBoosting (boosting)",
}
# Umbral de "abandona" según el puntaje que da cada modelo.
UMBRAL = {"predict_proba": 0.5, "decision_function": 0.0}


# Limpieza con estado de cleaning.py, activa por defecto (se aprende solo con train):
CLIP_OUTLIERS = "iqr"  # recorte de atípicos en COLUMNAS_RECORTE
MAX_NULOS = 0.5  # ColumnDropper: descarta variables con más de 50 % vacías o con un solo valor


def cargar(path, limpiar: bool = True) -> pd.DataFrame:
    """Lee los asociados y, por defecto, aplica la limpieza sin estado (``contrato.limpiar_extracto``).

    Quita espacios, vuelve vacíos los textos tipo 'N/A', convierte a número las columnas enteras escritas
    como texto y elimina filas repetidas, sin romper el contrato. Es segura antes de partir en train/test.
    """
    df = load_dataset(path)
    return limpiar_extracto(df) if limpiar else df


def _muestreador(nombre: str, random_state: int):
    from imblearn.over_sampling import ADASYN, SMOTE

    if nombre not in ("smote", "adasyn"):
        raise ValueError(f"muestreo debe ser 'smote' o 'adasyn', no '{nombre}'")
    return {"smote": SMOTE, "adasyn": ADASYN}[nombre](random_state=random_state)


def armar_pipeline(
    modelo,
    *,
    crear_variables: bool = False,
    muestreo: str | None = None,
    buscable: bool = False,
    random_state: int = 42,
    **preproceso,
) -> Pipeline:
    """El pipeline del abandono, el mismo para todos los scripts::

        preparar -> [crear variables] -> preprocess -> [SMOTE/ADASYN] -> model

    - ``preparar`` (PrepararVariables): recibe la fila cruda o ya preparada.
    - ``crear_variables``: agrega VARIABLES_NUEVAS (CrearVariables) y las preprocesa como numéricas.
    - ``preproceso``: parámetros de ``features.Preprocesador`` que cambian respecto al de los supervisados
      (mediana, OneHot, StandardScaler, recorte IQR de COLUMNAS_RECORTE, ColumnDropper al ``MAX_NULOS``), p. ej.
      ``numeric_strategy="knn"``, ``encoding="target"``, ``scaler="robust"``, ``pca=0.95``, ``rfe_features=15``.
    - ``muestreo`` ("smote"/"adasyn"): re-muestrea solo en ``fit`` (Pipeline de imblearn).
    - ``buscable``: el preprocesamiento queda como ``Preprocesador``, para buscar sus opciones con
      ``preprocess__<opción>`` (GridSearchCV, Optuna). Si no, es el preprocesador ya armado (ramas a la vista).
      Con ``muestreo`` siempre es ``Preprocesador``: imblearn no admite un Pipeline anidado como paso intermedio.
    """
    p = {
        "numeric": [*NUMERICAS, *VARIABLES_NUEVAS] if crear_variables else NUMERICAS,
        "categorical": CATEGORICAS,
        "clip_outliers": CLIP_OUTLIERS,
        "clip_columns": COLUMNAS_RECORTE,
        "max_nulos": MAX_NULOS,
        "random_state": random_state,
        **preproceso,
    }
    if buscable or muestreo is not None:
        pre = Preprocesador(**p)
    else:
        max_nulos, pca = p.pop("max_nulos"), p.pop("pca", None)
        pre = build_preprocessor(
            p.pop("numeric"),
            p.pop("categorical"),
            drop_columns=ColumnDropper(max_missing=max_nulos) if max_nulos is not None else None,
            pca_components=pca,
            **p,
        )
    pasos = [("preparar", PrepararVariables())]
    pasos += [("crear", CrearVariables())] if crear_variables else []
    pasos.append(("preprocess", pre))
    if muestreo is None:
        return Pipeline([*pasos, ("model", modelo)])
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline([*pasos, ("sampler", _muestreador(muestreo, random_state)), ("model", modelo)])


def preprocesador(clip_outliers: str | None = CLIP_OUTLIERS, max_nulos: float | None = MAX_NULOS):
    """El preprocesador de los supervisados, con la limpieza con estado de cleaning.py.

    - ``max_nulos``: ``ColumnDropper`` antes del ColumnTransformer (None = no descartar).
    - ``clip_outliers`` ("iqr"/"quantile"): recorta solo COLUMNAS_RECORTE (None = no recortar).
    """
    return build_preprocessor(
        NUMERICAS,
        CATEGORICAS,
        clip_outliers=clip_outliers,
        clip_columns=COLUMNAS_RECORTE,
        drop_columns=ColumnDropper(max_missing=max_nulos) if max_nulos is not None else None,
    )


def modelo_ridge(
    alpha: float = 1.0, clip_outliers: str | None = CLIP_OUTLIERS, max_nulos: float | None = MAX_NULOS
) -> Pipeline:
    """Ridge para clasificación: regresión Ridge sobre la clase codificada como -1/+1.

    class_weight="balanced" compensa que los retiros son minoría. El puntaje
    (decision_function) es positivo cuando el modelo predice abandono.
    """
    modelo = RidgeClassifier(alpha=alpha, class_weight="balanced")
    return armar_pipeline(modelo, clip_outliers=clip_outliers, max_nulos=max_nulos)


def modelos_supervisados(
    alpha: float = 1.0,
    random_state: int = 42,
    clip_outliers: str | None = CLIP_OUTLIERS,
    max_nulos: float | None = MAX_NULOS,
) -> dict[str, Pipeline]:
    """Ridge + el catálogo de models.py (logística, Random Forest, HistGB), con el mismo preprocesador.

    Random Forest es *bagging*: muchos árboles independientes, cada uno con una muestra bootstrap,
    que votan. HistGradientBoosting es *boosting*: árboles en secuencia, cada uno corrige los errores
    de los anteriores (la versión de scikit-learn inspirada en LightGBM).
    """
    catalogo = {  # mismo esqueleto que Ridge: preparar -> preprocess -> model
        nombre: armar_pipeline(modelo, clip_outliers=clip_outliers, max_nulos=max_nulos, random_state=random_state)
        for nombre, modelo in get_models(random_state).items()
    }
    return {"ridge": modelo_ridge(alpha, clip_outliers, max_nulos), **catalogo}


def metodo_puntaje(modelo: Pipeline) -> str:
    """Probabilidad si el modelo la da; si no (Ridge), la distancia a la frontera."""
    return "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"


def puntaje(modelo: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Puntaje de abandono: más alto = más riesgo. Se compara con ``UMBRAL[metodo_puntaje(modelo)]``."""
    valores = getattr(modelo, metodo_puntaje(modelo))(X)
    return valores[:, 1] if valores.ndim == 2 else valores


def metricas_supervisado(modelo: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    return compute_metrics(df[TARGET], puntaje(modelo, df), threshold=UMBRAL[metodo_puntaje(modelo)])


def guardar_modelo(pipeline: Pipeline, carpeta: Path, nombre: str) -> dict[str, Path]:
    """Guarda en .joblib el pipeline completo y, por separado, su preprocesamiento y su modelo.

    - ``<nombre>.joblib``: preparar + preprocesamiento + modelo. Es el que se usa para predecir: ``.predict(df)``.
    - ``<nombre>_preprocesamiento.joblib``: solo el preprocesamiento ya ajustado (imputación, escalado,
      codificación), para transformar datos nuevos o revisar lo que aprendió (medianas, categorías...).
    - ``<nombre>_modelo.joblib``: solo el estimador (coeficientes, hiperparámetros). Recibe datos ya
      transformados: ``modelo.predict(preprocesamiento.transform(df))``.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "pipeline": carpeta / f"{nombre}.joblib",
        "preprocesamiento": carpeta / f"{nombre}_preprocesamiento.joblib",
        "modelo": carpeta / f"{nombre}_modelo.joblib",
    }
    joblib.dump(pipeline, rutas["pipeline"])
    joblib.dump(pipeline[:-1], rutas["preprocesamiento"])  # todos los pasos menos el último
    joblib.dump(pipeline[-1], rutas["modelo"])
    return rutas


def tabla_predicciones(modelos: dict[str, Pipeline], df: pd.DataFrame) -> pd.DataFrame:
    """Como ``predict_model`` de PyCaret: las variables, el abandono real (si está) y lo que predice cada modelo.

    Por modelo, ``<nombre>_label`` (1 = abandona) y ``<nombre>_score``: la probabilidad de abandono si el
    modelo la da (0 a 1) o, si no (Ridge), la distancia a la frontera (> 0 = abandona). A diferencia de
    PyCaret, que da la confianza en la etiqueta predicha, aquí el score siempre mide riesgo de abandono:
    más alto = más riesgo, así sirve para ordenar a quién contactar primero.
    """
    X = preparar(df)
    tabla = X.assign(**{TARGET: df[TARGET].to_numpy()}) if TARGET in df else X.copy()
    for nombre, modelo in modelos.items():
        tabla[f"{nombre}_label"] = modelo.predict(X)
        tabla[f"{nombre}_score"] = puntaje(modelo, X)
    return tabla


def coeficientes(modelo: Pipeline, top: int = 10) -> pd.Series:
    """Variables con más peso (positivo = empuja hacia abandono)."""
    nombres = modelo.named_steps["preprocess"].get_feature_names_out()
    coef = pd.Series(np.ravel(modelo.named_steps["model"].coef_), index=nombres)  # 1D o (1, n) según versión
    return coef.reindex(coef.abs().sort_values(ascending=False).index)[:top]


# ---------------------------------------------------------------- sin objetivo: KMeans


class AgrupadorAbandono:
    """KMeans con 2 grupos sobre el comportamiento del asociado, entrenado SIN el objetivo.

    Después de agrupar, el grupo de "riesgo" se elige mirando solo los centroides: el que
    tiene más solicitudes de cancelación/disminución e intención de retiro (SENAL_RIESGO).
    El objetivo real solo se usa después, para medir qué tan bien quedó la separación.
    """

    def __init__(self, random_state: int = 42):
        self.pipeline = Pipeline(
            [
                ("preparar", PrepararVariables()),  # fila cruda o ya preparada
                ("comportamiento", ColumnTransformer([("c", "passthrough", COMPORTAMIENTO)])),
                ("imputer", SimpleImputer(strategy="median")),  # KMeans no admite vacíos; mediana de train
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=2, n_init=20, random_state=random_state)),
            ]
        )

    def fit(self, df: pd.DataFrame) -> AgrupadorAbandono:
        self.pipeline.fit(df)
        centros = pd.DataFrame(self.pipeline.named_steps["kmeans"].cluster_centers_, columns=COMPORTAMIENTO)
        senal = sum(centros[c] * peso for c, peso in SENAL_RIESGO.items())
        self.grupo_riesgo_ = int(senal.idxmax())
        self.perfil_ = pd.DataFrame(
            self.pipeline.named_steps["scaler"].inverse_transform(centros), columns=COMPORTAMIENTO
        ).rename(index={self.grupo_riesgo_: "riesgo", 1 - self.grupo_riesgo_: "estable"})
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """1 = grupo de riesgo de abandono, 0 = grupo estable. ``df`` crudo o ya preparado."""
        if "preparar" not in self.pipeline.named_steps:  # guardado antes de que el pipeline preparara solo
            df = preparar(df)[COMPORTAMIENTO] if "FECHANACIMIENTO" in df else df[COMPORTAMIENTO]
        return (self.pipeline.predict(df) == self.grupo_riesgo_).astype(int)

    def grupos(self, X: pd.DataFrame) -> np.ndarray:
        """Igual que ``predict`` (se conserva por compatibilidad)."""
        return self.predict(X)


def metricas_agrupacion(agrupador: AgrupadorAbandono, df: pd.DataFrame) -> dict[str, float]:
    """Compara los grupos con el abandono real (que el modelo nunca vio)."""
    y, grupo = df[TARGET].to_numpy(), agrupador.predict(df)
    X = agrupador.pipeline[:-1].transform(df)  # preparado, imputado y escalado, sin KMeans
    tn, fp, fn, tp = confusion_matrix(y, grupo, labels=[0, 1]).ravel()
    return {
        "base_rate": float(y.mean()),
        "pct_en_grupo_riesgo": float(grupo.mean()),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,  # de los marcados en riesgo, cuántos se fueron
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,  # de los que se fueron, cuántos quedaron en riesgo
        "accuracy": float((tp + tn) / len(y)),
        "ari": float(adjusted_rand_score(y, grupo)),  # 0 = azar, 1 = grupos idénticos al abandono real
        "silhouette": float(silhouette_score(X, grupo)) if len(set(grupo)) > 1 else float("nan"),
    }


# ---------------------------------------------------------------- gráfico: matrices de confusión


def figura_matrices_confusion(
    supervisados: dict[str, Pipeline], agrupador: AgrupadorAbandono, df: pd.DataFrame, columnas: int = 3
) -> Figure:
    """Matriz de confusión de cada modelo sobre ``df`` (filas = real, columnas = predicción).

    Un panel por modelo supervisado (en el orden del dict) y uno final para KMeans.
    Cada celda muestra el conteo y el % de su fila: en la fila 1, el % de la columna 1 es el recall.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure  # sin pyplot: no abre ventanas ni depende del backend

    y, X = df[TARGET].to_numpy(), preparar(df)
    reales = ["0\nse queda", "1\nabandona"]
    paneles = {TITULOS.get(nombre, nombre): (modelo.predict(X), reales) for nombre, modelo in supervisados.items()}
    paneles["KMeans (sin objetivo)"] = (agrupador.predict(df), ["0\nestable", "1\nriesgo"])
    columnas = min(columnas, len(paneles))
    filas = -(-len(paneles) // columnas)  # división hacia arriba
    fig = Figure(figsize=(4.5 * columnas, 4.4 * filas), layout="constrained")
    fig.suptitle(f"Matriz de confusión (n={len(y)})", fontsize=14)
    ejes = fig.subplots(filas, columnas, squeeze=False).ravel()
    for ax in ejes[len(paneles) :]:
        ax.remove()  # cuadrícula incompleta: sin paneles vacíos
    for ax, (titulo, (pred, etiquetas)) in zip(ejes, paneles.items(), strict=False):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap=colormaps["Blues"], vmin=0, vmax=cm.max())
        por_fila = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        for (i, j), n in np.ndenumerate(cm):
            color = "white" if n > cm.max() / 2 else "#1f2937"  # texto legible sobre celda oscura o clara
            ax.text(j, i, f"{n}\n({por_fila[i, j]:.0%})", ha="center", va="center", fontsize=12, color=color)
        ax.set_title(titulo)
        ax.set_xticks([0, 1], etiquetas)
        ax.set_yticks([0, 1], reales)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(length=0)
        for borde in ax.spines.values():
            borde.set_visible(False)
    return fig
