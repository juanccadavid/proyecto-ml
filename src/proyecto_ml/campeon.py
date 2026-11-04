"""El modelo campeón del abandono: el candidato con mejor PR-AUC en validación cruzada repetida, con su umbral.

Los afinados, Optuna y PyCaret son evidencia de experimentos (van a MLflow); el campeón es el único
"producto": lo usa el simulador y queda en ``models/abandono_campeon.joblib`` + ``models/campeon.json``.

- **Criterio:** PR-AUC promedio en validación cruzada repetida (no en los "nuevos" del generador, que
  miden si el modelo recupera la regla del generador). PR-AUC y no AUC: con pocos retiros, el AUC se ve
  alto aunque la mitad de las alertas sean falsas.
- **Umbrales**, elegidos con las predicciones fuera de partición (cada asociado puntuado por un modelo que
  no lo vio), en la escala del puntaje de cada modelo (probabilidad, o distancia a la frontera en Ridge):

  - ``umbral``: maximiza F2 (el recall pesa el doble: perder a quien se va cuesta más que una llamada de más).
  - ``umbral_alto``: el menor puntaje desde el cual al menos ``precision_min`` (0,8) de las alertas son
    retiros reales. Si ningún umbral la alcanza, no hay zona "se retira": todas las alertas son zona gris.

  Tres zonas: **se retira** (>= umbral_alto), **zona gris** (entre los dos) y **se queda** (< umbral).
- **KMeans** (sin objetivo) se entrena con los mismos datos: el simulador muestra en qué grupo cae el
  asociado y la tasa de retiro real de ese grupo.

**Incertidumbre** (para no sobrevender una cifra):

- ``pr_auc_std`` es la desviación **entre particiones** (5 x 3 = 15 puntajes), no entre repeticiones (esa,
  mucho menor, queda en ``pr_auc_std_repeticiones``). ``pr_auc_ic_inf`` / ``pr_auc_ic_sup``: intervalo del
  95 % por bootstrap de asociados.
- ``comparacion``: el campeón contra cada candidato, partición a partición (todos usan las mismas), con la
  prueba t corregida de Nadeau y Bengio para validación cruzada repetida. Si el intervalo de la diferencia
  incluye el 0, la elección **no es concluyente** (empate estadístico) y así se reporta.
- ``umbrales_honestos``: los umbrales se eligen y se miden sobre las mismas predicciones, lo que es optimista.
  Aquí se repite la regla en validación anidada: en cada partición externa los umbrales salen solo del
  entrenamiento y se miden en la prueba.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.models import get_models

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline

CRITERIO = "pr_auc"
FOLDS = 5
REPETICIONES = 3
BETA = 2.0  # F2
PRECISION_MIN = 0.8  # zona "se retira": 8 de cada 10 alertas son retiros reales (fuera de partición)
ARCHIVO_MODELO = "abandono_campeon.joblib"
ARCHIVO_INFO = "campeon.json"
ARCHIVO_GRUPOS = "abandono_kmeans.joblib"
ZONAS = ("se retira", "zona gris", "se queda")
NIVEL = 0.95  # de los intervalos (bootstrap y diferencia entre candidatos)
BOOTSTRAP = 1000
TITULOS = {  # los 4 supervisados + los candidatos extra (lista_candidatos y --optuna)
    **a.TITULOS,
    "logreg_sin_balancear": "Logística sin balancear",
    "logreg_target": "Logística + TargetEncoder",
    "logreg_indicador": "Logística + indicador de vacíos",
    "random_forest_optuna": "Random Forest (Optuna)",
    "hist_gb_optuna": "HistGB (Optuna)",
}


class DatosInvalidos(ValueError):
    """El lote no cumple el contrato de datos: no se entrena con él."""


def datos_eleccion(ruta: Path | None = None, origen: str | None = None) -> tuple[pd.DataFrame, str]:
    """Los asociados con que se elige el campeón y de dónde salieron, según el origen (``config.origen_datos``).

    - **sintetico** (por defecto): ``ruta``, o el archivo de ``make datos-desarrollo``, o los mismos 5.000
      generados en memoria (Streamlit Community Cloud, CI).
    - **real_anonimizado**: solo ``ruta`` (el extracto anonimizado). Nunca cae en los sintéticos: sin ``ruta``
      falla, para no etiquetar como reales unos datos generados.

    Todo archivo debe cumplir el contrato (``contrato.validar``); si no, ``DatosInvalidos``. Con datos reales el
    mensaje no lleva valores ni filas de ejemplo (``contrato.sin_valores``). Los generados en memoria no se
    validan aquí: el generador cumple el contrato por construcción (lo prueba tests/data).
    """
    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, PROCESSED_DIR, origen_datos
    from proyecto_ml.generador import NOMBRE_DESARROLLO, datos_desarrollo

    origen = origen_datos(origen)
    if origen == ORIGEN_REAL and ruta is None:
        raise DatosInvalidos(
            "ORIGEN_DATOS=real_anonimizado: indica el extracto anonimizado con --datos (make ... DATA=...). "
            "En este modo no se usan datos sintéticos."
        )
    desarrollo = PROCESSED_DIR / f"{NOMBRE_DESARROLLO}.parquet"
    if ruta is None and not desarrollo.exists():
        # La misma limpieza que a.cargar aplica al leer el archivo: así los dos caminos dan los mismos datos.
        return contrato.limpiar_extracto(datos_desarrollo()), "datos_desarrollo() en memoria"
    archivo = ruta or desarrollo
    df = a.cargar(archivo)
    # Datos reales: el contrato del dataset ANONIMIZADO (sin PII, con seudónimos); sintéticos: el de origen.
    exigido = contrato.para_ml() if origen == ORIGEN_REAL else contrato.cargar()
    errores = contrato.validar(df, exigido)
    for aviso in contrato.advertencias(df, exigido):  # heurísticas sin regla de negocio: se avisa, no se rechaza
        warnings.warn(aviso, contrato.AdvertenciaContrato, stacklevel=2)
    if origen == ORIGEN_REAL:  # fechas después del corte: FECHA_CORTE no es la del extracto y la foto queda corrida
        for columna in ("FECHARETIRO", "FECHAULTACTUALIZACION", "FECHAINGRESOLABORA"):
            if columna in df and (pd.to_datetime(df[columna], errors="coerce") > a.FECHA_CORTE).any():
                errores.append(
                    f"{columna} tiene fechas después de FECHA_CORTE ({a.FECHA_CORTE.date()}): pon en el .env "
                    "FECHA_CORTE = la fecha de extracción del archivo"
                )
    if errores:
        mostrar = contrato.sin_valores(errores) if origen == ORIGEN_REAL else errores
        detalle = "\n".join(f"  - {e}" for e in mostrar[:10])
        raise DatosInvalidos(
            f"{archivo.name} no cumple el contrato de datos ({len(errores)} incumplimientos):\n{detalle}\n"
            f"Revísalo con: make {'contrato-ml' if origen == ORIGEN_REAL else 'contrato'} DATA={archivo}"
        )
    return df, archivo.name


def datos_exploracion(ruta: Path, script: str, limpiar: bool = True) -> pd.DataFrame:
    """Los datos de los scripts EXPLORATORIOS (make abandono, afinar, optuna, pycaret), o ``DatosInvalidos``.

    Esos scripts validan contra asociados "nuevos" del generador: solo tienen sentido con datos sintéticos. Con
    ``ORIGEN_DATOS=real_anonimizado`` mezclarían orígenes (y registrarían en el experimento real de MLflow), así
    que se niegan; la evaluación con datos reales es ``make campeon``. Con sintéticos, el archivo debe cumplir el
    contrato de origen, igual que en ``datos_eleccion``.
    """
    from pathlib import Path

    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, origen_datos

    if origen_datos() == ORIGEN_REAL:
        raise DatosInvalidos(
            f"{script} es exploratorio: compara contra asociados generados y solo tiene sentido con "
            "ORIGEN_DATOS=sintetico. Con datos reales usa make campeon DATA=<extracto anonimizado>."
        )
    df = a.cargar(ruta, limpiar)
    if errores := contrato.validar(df):
        detalle = "\n".join(f"  - {e}" for e in errores[:10])
        raise DatosInvalidos(
            f"{Path(ruta).name} no cumple el contrato de datos ({len(errores)} incumplimientos):\n{detalle}\n"
            f"Revísalo con: make contrato DATA={ruta}"
        )
    return df


def lista_candidatos(X: pd.DataFrame | None = None, random_state: int = 42) -> dict[str, Pipeline]:
    """Los 4 supervisados + las técnicas que en ``make tecnicas`` le ganan a la base de forma concluyente
    (intervalo de la diferencia partición a partición sin el 0), aplicadas a la logística, que es esa base:

    - ``logreg_sin_balancear``: sin ``class_weight`` (+0,057; IC 0,028 a 0,086).
    - ``logreg_target``: categóricas con TargetEncoder (+0,049; IC 0,024 a 0,073).
    - ``logreg_indicador``: indicador de vacíos, solo si ``X`` tiene vacíos en las numéricas (sin vacíos no
      agrega columnas). En ``make tecnicas`` no fue concluyente: allí los vacíos se pusieron al azar.

    Las variables nuevas y el binning no entran: su diferencia con la base cabe en el ruido (variables nuevas
    −0,003, IC −0,020 a 0,015).

    Los de Optuna se suman con ``make campeon ARGS="--optuna 30"`` (se afinan con estos mismos datos). Los
    afinados por grilla de ``make afinar`` (Ridge, pycaret_lr) no entran: son Ridge y la logística con otro
    alpha/C, buscados con otros datos (DATOS_ASOCIADOS del .env), no con los de elección.
    """
    salida = a.modelos_supervisados(random_state=random_state)
    logistica = get_models(random_state)["logreg"]
    sin_balancear = clone(logistica).set_params(class_weight=None)
    salida["logreg_sin_balancear"] = a.armar_pipeline(sin_balancear, random_state=random_state)
    salida["logreg_target"] = a.armar_pipeline(clone(logistica), encoding="target", random_state=random_state)
    if X is not None and a.preparar(X)[a.NUMERICAS].isna().any().any():
        salida["logreg_indicador"] = a.armar_pipeline(
            clone(logistica), missing_indicator=True, random_state=random_state
        )
    return salida


def puntajes_fuera_de_particion(modelo: Pipeline, X, y, folds: int = FOLDS, repeticiones: int = REPETICIONES):
    """(puntajes, pr_auc_particion): el puntaje de cada asociado dado por un modelo que no lo vio (repeticiones x
    filas) y la PR-AUC de cada partición (repeticiones x folds).

    Cada repetición baraja con su propia semilla, la misma para todos los candidatos: así se pueden comparar
    partición a partición (``comparar_particiones``).
    """
    metodo, y = a.metodo_puntaje(modelo), np.asarray(y)
    puntajes, por_particion = np.empty((repeticiones, len(y))), np.empty((repeticiones, folds))
    for r in range(repeticiones):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)
        for k, (entrena, prueba) in enumerate(cv.split(X, y)):
            p = getattr(clone(modelo).fit(X.iloc[entrena], y[entrena]), metodo)(X.iloc[prueba])
            puntajes[r, prueba] = p[:, 1] if p.ndim == 2 else p
            por_particion[r, k] = average_precision_score(y[prueba], puntajes[r, prueba])
    return puntajes, por_particion


def intervalo_bootstrap(y, puntaje, n: int = BOOTSTRAP, nivel: float = NIVEL, seed: int = 0) -> tuple[float, float]:
    """Intervalo de la PR-AUC remuestreando asociados con reemplazo (dentro de cada clase, para que la tasa de
    retiro no cambie entre remuestras)."""
    rng, y, puntaje = np.random.default_rng(seed), np.asarray(y), np.asarray(puntaje)
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    valores = []
    for _ in range(n):
        i = np.concatenate([rng.choice(positivos, len(positivos)), rng.choice(negativos, len(negativos))])
        valores.append(average_precision_score(y[i], puntaje[i]))
    bajo, alto = np.quantile(valores, [(1 - nivel) / 2, (1 + nivel) / 2])
    return float(bajo), float(alto)


def comparar_particiones(a_particion: np.ndarray, b_particion: np.ndarray, folds: int, nivel: float = NIVEL) -> dict:
    """Diferencia de PR-AUC (a − b) partición a partición, con la prueba t corregida de Nadeau y Bengio.

    En validación cruzada las particiones comparten casi todo el entrenamiento: sus puntajes no son
    independientes y la prueba t común da intervalos demasiado estrechos. La corrección suma a la varianza
    ``n_prueba / n_entrenamiento`` (= 1 / (folds − 1)).
    """
    d = (np.asarray(a_particion) - np.asarray(b_particion)).ravel()
    j = d.size
    error = np.sqrt((1 / j + 1 / (folds - 1)) * d.var(ddof=1))
    margen = stats.t.ppf((1 + nivel) / 2, j - 1) * error
    # Sin variación (todas las diferencias iguales, p. ej. un candidato contra sí mismo): p = 1 si son 0, si no 0.
    p_valor = 2 * stats.t.sf(abs(d.mean()) / error, j - 1) if error > 0 else float(d.mean() == 0)
    return {
        "diferencia": float(d.mean()),
        "ic_inf": float(d.mean() - margen),
        "ic_sup": float(d.mean() + margen),
        "p_valor": float(p_valor),
        "gana_en": float((d > 0).mean()),  # fracción de particiones en que a supera a b
    }


def umbrales_honestos(
    modelo: Pipeline, X, y, folds: int = FOLDS, beta: float = BETA, precision_min: float = PRECISION_MIN, seed=100
) -> dict:
    """Precisión y recall de la REGLA de umbrales medidos sin optimismo (validación anidada).

    En cada partición externa: predicciones fuera de partición internas (3 particiones) del entrenamiento ->
    ``elegir_umbrales`` -> modelo entrenado con todo el entrenamiento -> se aplican esos umbrales a la prueba,
    que no participó en elegirlos. Al final se juntan las pruebas.
    """
    metodo, y = a.metodo_puntaje(modelo), np.asarray(y)
    alerta, alto = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)
    sin_alto = 0
    for entrena, prueba in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        Xe, ye = X.iloc[entrena], y[entrena]
        interno = cross_val_predict(
            modelo, Xe, ye, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method=metodo
        )
        u = elegir_umbrales(ye, interno[:, 1] if interno.ndim == 2 else interno, beta, precision_min)
        p = a.puntaje(clone(modelo).fit(Xe, ye), X.iloc[prueba])
        alerta[prueba] = p >= u["umbral"]
        if u["umbral_alto"] is None:
            sin_alto += 1
        else:
            alto[prueba] = p >= u["umbral_alto"]
    positivos = max(int(y.sum()), 1)

    def _precision(marcados) -> float:
        return float((marcados & (y == 1)).sum() / marcados.sum()) if marcados.any() else float("nan")

    return {
        "precision": _precision(alerta),
        "recall": float((alerta & (y == 1)).sum() / positivos),
        "precision_alto": _precision(alto),
        "recall_alto": float((alto & (y == 1)).sum() / positivos),
        "alertas_alto": int(alto.sum()),
        "particiones_sin_alto": sin_alto,
    }


def validacion_temporal(
    modelo: Pipeline,
    df: pd.DataFrame,
    horizonte_meses: int = a.HORIZONTE_MESES,
    meses_prueba: int | None = None,
    folds: int = FOLDS,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
) -> dict:
    """Validación de afuera en el tiempo: entrena con la foto vieja y mide en la foto reciente, sin tocarla.

    ``df`` es el extracto completo (antes de ``a.foto``). Con ``validacion.particion_temporal``:

    1. foto vieja (``meses_prueba`` antes, por defecto ``horizonte_meses``): predicciones fuera de partición ->
       ``elegir_umbrales``. El umbral sale SOLO de la foto vieja;
    2. el modelo se entrena con toda la foto vieja;
    3. se mide en la foto reciente con ese umbral fijo: ni el modelo ni el umbral vieron esas etiquetas.

    Es la evaluación que vale con datos reales (no hay "asociados nuevos" del generador). Límite: con un solo
    extracto, los conteos sin fecha (``a.CONTEOS_SIN_FECHA``) son los mismos números en las dos fotos; si se
    midieron a la fecha de extracción, las dos tienen fuga. Ver ``config.temporalidad_confirmada``.
    """
    from proyecto_ml.validacion import particion_temporal

    meses_prueba = meses_prueba or horizonte_meses
    vieja, reciente = particion_temporal(df, horizonte_meses, meses_prueba)
    Xv, yv = a.separar(vieja)
    Xr, yr = a.separar(reciente)
    for nombre, y in (("vieja", yv), ("reciente", yr)):
        if int(y.sum()) < folds or int((y == 0).sum()) < folds:
            raise ValueError(f"La foto {nombre} tiene menos de {folds} asociados en alguna clase: usa más datos.")
    puntajes, _ = puntajes_fuera_de_particion(modelo, Xv, yv, folds, repeticiones=1)
    umbrales = elegir_umbrales(yv, puntajes.mean(axis=0), beta, precision_min)
    entrenado = clone(modelo).fit(Xv, yv)
    metricas = compute_metrics(
        yr, a.puntaje(entrenado, Xr), threshold=umbrales["umbral"], metodo=a.metodo_puntaje(entrenado)
    )
    return {
        "t0_entrenamiento": str(vieja[a.FECHA_OBSERVACION].iloc[0].date()),
        "t0_prueba": str(reciente[a.FECHA_OBSERVACION].iloc[0].date()),
        "asociados_entrenamiento": len(vieja),
        "retiros_entrenamiento": int(yv.sum()),
        "asociados_prueba": len(reciente),
        "retiros_prueba": int(yr.sum()),
        "umbral": umbrales["umbral"],
        **metricas,
    }


def elegir_umbrales(y, puntaje, beta: float = BETA, precision_min: float = PRECISION_MIN) -> dict:
    """``umbral`` (máximo F-beta) y ``umbral_alto`` (precisión >= precision_min), con sus métricas."""
    precision, recall, cortes = precision_recall_curve(y, puntaje)
    precision, recall = precision[:-1], recall[:-1]  # el último punto (recall 0) no tiene corte
    fbeta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    i = int(np.argmax(fbeta))
    resultado = {
        "umbral": float(cortes[i]),
        "fbeta": float(fbeta[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "umbral_alto": None,
        "precision_alto": None,
        "recall_alto": None,
    }
    seguros = np.flatnonzero((precision >= precision_min) & (cortes >= cortes[i]))
    if len(seguros):
        j = int(seguros[0])  # cortes crece: el primero es el más bajo que cumple
        resultado |= {
            "umbral_alto": float(cortes[j]),
            "precision_alto": float(precision[j]),
            "recall_alto": float(recall[j]),
        }
    return resultado


BRIER_SKILL_MIN = 0.5  # por debajo, la "probabilidad" del modelo no es creíble como probabilidad


def etiqueta_puntaje(metodo: str, metricas: dict) -> str:
    """Cómo mostrar el puntaje: "Probabilidad de retiro" solo si es una probabilidad CALIBRADA.

    Con ``class_weight="balanced"`` la probabilidad queda inflada (Brier skill cerca de 0 o negativo): llamarla
    "probabilidad de retiro" exageraría el riesgo. ``metricas`` son las del campeón (``metricas_cv``).
    """
    if metodo != "predict_proba":
        return "Puntaje del modelo"
    habilidad = metricas.get("brier_skill")
    if habilidad is not None and not np.isnan(habilidad) and habilidad < BRIER_SKILL_MIN:
        return "Puntaje de riesgo (no calibrado)"
    return "Probabilidad de retiro"


def zona(puntaje: float, umbral: float, umbral_alto: float | None) -> str:
    """'se retira', 'zona gris' o 'se queda' según los dos umbrales del campeón."""
    if umbral_alto is not None and puntaje >= umbral_alto:
        return "se retira"
    return "zona gris" if puntaje >= umbral else "se queda"


class ModeloZonas(BaseEstimator):
    """El campeón listo para usar: su pipeline y sus dos umbrales en un solo objeto.

    ``predict(df)`` da la zona de cada asociado ("se retira", "zona gris" o "se queda") a partir de la fila cruda
    (o ya preparada). ``alerta(df)`` da 1 desde el umbral de alerta. ``predict_proba`` o ``decision_function``
    (el que tenga el modelo) dan el puntaje, así que ``abandono.puntaje`` también funciona con él. Es lo que se
    guarda en models/ y en MLflow (``models:/abandono@campeon``).
    """

    def __init__(self, modelo=None, umbral: float = 0.5, umbral_alto: float | None = None):
        self.modelo = modelo
        self.umbral = umbral
        self.umbral_alto = umbral_alto

    def fit(self, X, y):
        """Reentrena el pipeline con los mismos umbrales (normalmente llega ya entrenado)."""
        self.modelo.fit(X, y)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return check_is_fitted(self.modelo) is None

    @property
    def metodo(self) -> str:
        return a.metodo_puntaje(self.modelo)

    def puntaje(self, X) -> np.ndarray:
        return a.puntaje(self.modelo, X)

    def predict(self, X) -> np.ndarray:
        return np.array([zona(p, self.umbral, self.umbral_alto) for p in self.puntaje(X)], dtype=object)

    def alerta(self, X) -> np.ndarray:
        return (self.puntaje(X) >= self.umbral).astype(int)

    @available_if(lambda self: hasattr(self.modelo, "predict_proba"))
    def predict_proba(self, X):
        return self.modelo.predict_proba(X)

    @available_if(lambda self: hasattr(self.modelo, "decision_function"))
    def decision_function(self, X):
        return self.modelo.decision_function(X)


@dataclass
class Candidato:
    nombre: str
    modelo: Pipeline  # sin entrenar
    puntajes: np.ndarray  # fuera de partición, repeticiones x filas
    umbrales: dict
    metricas: dict  # pr_auc (std entre particiones, IC bootstrap), roc_auc, umbrales, mcc y kappa al umbral...
    pr_auc_particion: np.ndarray | None = None  # repeticiones x folds, mismas particiones en todos

    @property
    def promedio(self) -> np.ndarray:
        """El puntaje fuera de partición de cada asociado, promedio de las repeticiones."""
        return self.puntajes.mean(axis=0)


def evaluar(nombre: str, modelo: Pipeline, X, y, folds: int = FOLDS, repeticiones: int = REPETICIONES, **kw):
    puntajes, por_particion = puntajes_fuera_de_particion(modelo, X, y, folds, repeticiones)
    pr = [average_precision_score(y, p) for p in puntajes]
    roc = [roc_auc_score(y, p) for p in puntajes]
    promedio = puntajes.mean(axis=0)  # el umbral se elige sobre el promedio de las repeticiones
    umbrales = elegir_umbrales(y, promedio, **kw)
    alerta = (promedio >= umbrales["umbral"]).astype(int)
    metodo = a.metodo_puntaje(modelo)
    al_umbral = compute_metrics(y, promedio, threshold=umbrales["umbral"], metodo=metodo)
    es_proba = metodo == "predict_proba"  # por el método, no por el rango de los valores
    tasa = float(np.mean(y))
    brier = al_umbral["brier"]
    ic_inf, ic_sup = intervalo_bootstrap(y, promedio)
    metricas = {
        "pr_auc": float(np.mean(pr)),
        "pr_auc_std": float(np.std(por_particion, ddof=1)),  # entre particiones: la variación honesta
        "pr_auc_std_repeticiones": float(np.std(pr)),
        "pr_auc_ic_inf": ic_inf,
        "pr_auc_ic_sup": ic_sup,
        "roc_auc": float(np.mean(roc)),
        **{k: v for k, v in umbrales.items() if v is not None},
        "mcc": float(matthews_corrcoef(y, alerta)),  # al umbral de alerta, corrigen el desbalance
        "kappa": float(cohen_kappa_score(y, alerta)),
        # al umbral de alerta (F2 incluido) y sin umbral (precision@10 %, Brier: calibración de la probabilidad)
        **{k: al_umbral[k] for k in ("f2", "specificity", "balanced_accuracy", "precision_at_10pct")},
        "brier": brier,
        # Brier skill score: 1 = perfecto, 0 = igual que decir siempre la tasa de retiro (mal calibrado)
        "brier_skill": 1 - brier / (tasa * (1 - tasa)) if es_proba and 0 < tasa < 1 else float("nan"),
    }
    return Candidato(nombre, modelo, puntajes, umbrales, metricas, por_particion)


@dataclass
class Campeon:
    nombre: str
    modelo: Pipeline  # entrenado con todos los datos
    umbral: float
    umbral_alto: float | None
    metodo: str  # predict_proba o decision_function
    candidatos: dict[str, Candidato]
    agrupador: a.AgrupadorAbandono
    grupos: dict  # perfil y tasa de retiro real de cada grupo de KMeans
    info: dict = field(default_factory=dict)  # datos, filas, horizonte, fecha...
    y: np.ndarray | None = None  # objetivo de los datos de elección (para la figura de umbrales)
    honestos: dict | None = None  # umbrales_honestos del campeón (validación anidada)

    def tabla(self) -> pd.DataFrame:
        """Candidatos x métricas de la validación cruzada, ordenados por el criterio."""
        tabla = pd.DataFrame({n: c.metricas for n, c in self.candidatos.items()}).T
        return tabla.sort_values(CRITERIO, ascending=False)

    def comparacion(self) -> pd.DataFrame:
        """El campeón contra cada candidato: diferencia de PR-AUC, su intervalo, p y si es concluyente."""
        propio = self.candidatos[self.nombre].pr_auc_particion
        folds = self.info.get("folds", FOLDS)
        filas = {
            n: comparar_particiones(propio, cand.pr_auc_particion, folds)
            for n, cand in self.candidatos.items()
            if n != self.nombre and cand.pr_auc_particion is not None
        }
        tabla = pd.DataFrame(filas).T
        if len(tabla):
            tabla["concluyente"] = tabla["ic_inf"] > 0  # el intervalo de (campeón − candidato) no toca el 0
        return tabla.sort_values("diferencia") if len(tabla) else tabla

    def empatados(self) -> list[str]:
        """Candidatos que el campeón no supera de forma concluyente (empate estadístico)."""
        tabla = self.comparacion()
        return list(tabla.index[~tabla["concluyente"].astype(bool)]) if len(tabla) else []

    def con_zonas(self) -> ModeloZonas:
        """El pipeline entrenado con sus dos umbrales: lo que se guarda y se registra en MLflow."""
        return ModeloZonas(self.modelo, self.umbral, self.umbral_alto)

    def zona(self, X: pd.DataFrame) -> list[str]:
        return self.con_zonas().predict(X).tolist()


def perfil_grupos(agrupador: a.AgrupadorAbandono, df: pd.DataFrame) -> dict:
    """Por grupo (riesgo/estable): tamaño, tasa de retiro real (que KMeans nunca vio) y perfil promedio."""
    grupo = pd.Series(agrupador.predict(df), index=df.index).map({1: "riesgo", 0: "estable"})
    salida = {}
    for nombre in ("riesgo", "estable"):
        en_grupo = grupo == nombre
        salida[nombre] = {
            "asociados": int(en_grupo.sum()),
            "tasa_retiro": float(df.loc[en_grupo, a.TARGET].mean()) if en_grupo.any() else float("nan"),
            "perfil": {c: float(v) for c, v in agrupador.perfil_.loc[nombre].items()},
        }
    return salida


def elegir(
    df: pd.DataFrame,
    candidatos: dict[str, Pipeline] | None = None,
    folds: int = FOLDS,
    repeticiones: int = REPETICIONES,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
    info: dict | None = None,
    honestos: bool = True,
) -> Campeon:
    """Evalúa los candidatos (por defecto ``lista_candidatos(X)``), elige el de mejor PR-AUC y lo entrena con todo.

    ``honestos``: mide además la regla de umbrales del campeón en validación anidada (``umbrales_honestos``).
    """
    X, y = a.separar(df)  # filas crudas: el pipeline prepara las variables (PrepararVariables)
    candidatos = candidatos or lista_candidatos(X)
    evaluados = {
        n: evaluar(n, m, X, y, folds, repeticiones, beta=beta, precision_min=precision_min)
        for n, m in candidatos.items()
    }
    nombre = max(evaluados, key=lambda n: evaluados[n].metricas[CRITERIO])
    elegido = evaluados[nombre]
    agrupador = a.AgrupadorAbandono().fit(df)
    info = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "filas": len(df),
        "tasa_retiro": float(y.mean()),
        "folds": folds,
        "repeticiones": repeticiones,
        "beta": beta,
        "precision_min": precision_min,
        **(info or {}),
    }
    return Campeon(
        nombre=nombre,
        modelo=clone(elegido.modelo).fit(X, y),
        umbral=elegido.umbrales["umbral"],
        umbral_alto=elegido.umbrales["umbral_alto"],
        metodo=a.metodo_puntaje(elegido.modelo),
        candidatos=evaluados,
        agrupador=agrupador,
        grupos=perfil_grupos(agrupador, df),
        info=info,
        y=y.to_numpy(),
        honestos=umbrales_honestos(elegido.modelo, X, y, folds, beta, precision_min) if honestos else None,
    )


def resumen(campeon: Campeon) -> dict:
    """Lo que va en campeon.json: modelo, umbrales, métricas de todos los candidatos, grupos y datos."""
    return {
        "modelo": campeon.nombre,
        "titulo": TITULOS.get(campeon.nombre, campeon.nombre),
        "metodo": campeon.metodo,
        "umbral": campeon.umbral,
        "umbral_alto": campeon.umbral_alto,
        "criterio": CRITERIO,
        "metricas_cv": campeon.candidatos[campeon.nombre].metricas,
        "candidatos": {n: c.metricas for n, c in campeon.candidatos.items()},
        "comparacion": campeon.comparacion().to_dict(orient="index"),  # campeón − candidato, partición a partición
        "empatados": campeon.empatados(),
        "umbrales_honestos": campeon.honestos,
        "confusion_cv": {  # fuera de partición, con cada umbral: tn, fp, fn, tp
            n: dict(zip(("tn", "fp", "fn", "tp"), map(int, m.ravel()), strict=True))
            for n, m in (matrices_confusion(campeon).items() if campeon.y is not None else ())
        },
        "grupos": campeon.grupos,
        **campeon.info,
    }


def guardar(campeon: Campeon, carpeta: Path, extra: dict | None = None) -> dict[str, Path]:
    """``abandono_campeon.joblib`` (ModeloZonas), ``abandono_kmeans.joblib`` y ``campeon.json`` en ``carpeta``."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "modelo": carpeta / ARCHIVO_MODELO,
        "grupos": carpeta / ARCHIVO_GRUPOS,
        "info": carpeta / ARCHIVO_INFO,
    }
    joblib.dump(campeon.con_zonas(), rutas["modelo"])
    joblib.dump(campeon.agrupador, rutas["grupos"])
    texto = json.dumps({**resumen(campeon), **(extra or {})}, indent=2, ensure_ascii=False, default=str)
    rutas["info"].write_text(texto + "\n", encoding="utf-8", newline="\n")
    return rutas


def cargar(carpeta: Path):
    """(modelo, agrupador, info) desde ``carpeta``; None si falta alguno de los tres archivos.

    ``modelo`` es un ModeloZonas. Un campeón guardado antes (solo el pipeline) se envuelve con los umbrales de
    ``campeon.json``.
    """
    rutas = [carpeta / ARCHIVO_MODELO, carpeta / ARCHIVO_GRUPOS, carpeta / ARCHIVO_INFO]
    if not all(r.exists() for r in rutas):
        return None
    modelo, info = joblib.load(rutas[0]), json.loads(rutas[2].read_text(encoding="utf-8"))
    if not isinstance(modelo, ModeloZonas):
        modelo = ModeloZonas(modelo, info["umbral"], info["umbral_alto"])
    return modelo, joblib.load(rutas[1]), info


def figura_umbrales(campeon: Campeon):
    """Plotly: precisión y recall del campeón según el umbral (fuera de partición), con las tres zonas."""
    import plotly.graph_objects as go

    candidato = campeon.candidatos[campeon.nombre]
    y = campeon.y
    precision, recall, cortes = precision_recall_curve(y, candidato.promedio)
    fig = go.Figure()
    for valores, nombre, color in ((precision[:-1], "Precisión", "#2a78d6"), (recall[:-1], "Recall", "#eb6834")):
        fig.add_scatter(
            x=cortes,
            y=valores,
            name=nombre,
            mode="lines",
            line={"width": 2, "color": color},
            hovertemplate=f"{nombre}=%{{y:.2f}}<br>umbral=%{{x:.3f}}<extra></extra>",
        )
    bajo, alto = float(cortes.min()), float(cortes.max())
    tope = campeon.umbral_alto if campeon.umbral_alto is not None else alto
    bandas = [("se queda", bajo, campeon.umbral), ("zona gris", campeon.umbral, tope)]
    if campeon.umbral_alto is not None:
        bandas.append(("se retira", campeon.umbral_alto, alto))
    for (texto, x0, x1), opacidad in zip(bandas, (0.0, 0.08, 0.16), strict=False):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#6b7280",
            opacity=opacidad,
            line_width=0,
            annotation_text=texto,
            annotation_position="top left",
        )
    escala = "probabilidad" if campeon.metodo == "predict_proba" else "puntaje (distancia a la frontera)"
    fig.update_layout(
        title=f"Campeón: {TITULOS.get(campeon.nombre, campeon.nombre)} — precisión y recall según el umbral",
        xaxis_title=f"Umbral ({escala})",
        yaxis_title="",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


# ---------------------------------------------------------------- curva ROC y matriz de confusión del campeón

# Paleta de referencia en orden fijo, un color por candidato (hasta 8: los de lista_candidatos y --optuna)
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_GRIS, _TINTA = "#898781", "#0b0b0b"


def matrices_confusion(campeon: Campeon) -> dict[str, np.ndarray]:
    """Matriz de confusión del campeón (filas = real, columnas = predicción) con cada uno de sus umbrales.

    Con los puntajes fuera de partición: cada asociado lo clasificó un modelo que no lo vio, así que cuadra con
    la precisión y el recall de ``metricas_cv``. "alerta" = zona gris o se retira; "se retira" = umbral alto.
    """
    puntaje = campeon.candidatos[campeon.nombre].promedio
    cortes = {"alerta": campeon.umbral}
    if campeon.umbral_alto is not None and campeon.umbral_alto > campeon.umbral:  # si coinciden, una sola matriz
        cortes["se retira"] = campeon.umbral_alto
    return {
        nombre: confusion_matrix(campeon.y, (puntaje >= corte).astype(int), labels=[0, 1])
        for nombre, corte in cortes.items()
    }


def figura_confusion(campeon: Campeon):
    """Plotly: un panel por umbral del campeón con el conteo y el % de su fila (en la fila 'se retira', el recall)."""
    from plotly.subplots import make_subplots

    matrices = matrices_confusion(campeon)
    corte = {"alerta": campeon.umbral, "se retira": campeon.umbral_alto}
    formato = ".1%" if campeon.metodo == "predict_proba" else "+.2f"
    titulos = [f"Umbral {n} ({corte[n]:{formato}})" for n in matrices]
    fig = make_subplots(rows=1, cols=len(matrices), subplot_titles=titulos, horizontal_spacing=0.12)
    reales = ["se queda (0)", "se retira (1)"]
    for k, (nombre, cm) in enumerate(matrices.items(), start=1):
        por_fila = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        predichas = ["sin alerta", "alerta"] if nombre == "alerta" else ["no 'se retira'", "se retira"]
        celdas = [
            [f"<b>{n}</b><br>{p:.1%} de la fila" for n, p in zip(fila, pf, strict=True)]
            for fila, pf in zip(cm, por_fila, strict=True)
        ]
        fig.add_heatmap(
            z=por_fila,
            x=predichas,
            y=reales,
            text=celdas,
            texttemplate="%{text}",
            customdata=cm,
            colorscale=[[0, "#cde2fb"], [1, "#104281"]],  # azul secuencial de la paleta, claro -> oscuro
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="real: %{y}<br>predicción: %{x}<br>%{customdata} asociados (%{z:.1%} de la fila)"
            "<extra></extra>",
            row=1,
            col=k,
        )
        fig.update_xaxes(title_text="Predicción", side="bottom", row=1, col=k)
    fig.update_yaxes(title_text="Real", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Campeón: {TITULOS.get(campeon.nombre, campeon.nombre)} — matriz de confusión "
        f"(fuera de partición, n={len(campeon.y)})<br><sup>El umbral se eligió con estas mismas predicciones: "
        "las cifras son optimistas (la validación anidada está en campeon.json, umbrales_honestos)</sup>",
        template="plotly_white",
        height=420,
        width=440 * len(matrices) + 80,
        font={"color": _TINTA},
    )
    return fig


def figura_roc(campeon: Campeon):
    """Plotly: curva ROC fuera de partición de cada candidato, con los umbrales del campeón marcados.

    Eje x: falsos positivos entre los que se quedan (1 − especificidad); eje y: recall. La diagonal es el azar
    (AUC 0,5). Con pocos retiros el AUC se ve alto aunque muchas alertas sean falsas: por eso el criterio es
    la PR-AUC y esta curva se lee junto con la de precisión y recall (``figura_umbrales``).
    """
    import plotly.graph_objects as go

    y = campeon.y
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="azar (AUC 0,50)", line={"color": _GRIS, "width": 1, "dash": "dot"}
    )
    for i, (nombre, candidato) in enumerate(campeon.candidatos.items()):
        color = _COLORES[i % len(_COLORES)]
        fpr, tpr, cortes = roc_curve(y, candidato.promedio)
        es_campeon = nombre == campeon.nombre
        titulo = TITULOS.get(nombre, nombre)
        fig.add_scatter(
            x=fpr,
            y=tpr,
            customdata=cortes,
            mode="lines",
            name=f"{titulo}{' (campeón)' if es_campeon else ''} · AUC {roc_auc_score(y, candidato.promedio):.3f}",
            line={"color": color, "width": 3 if es_campeon else 2},
            opacity=1 if es_campeon else 0.6,
            hovertemplate=f"{titulo}<br>falsos positivos=%{{x:.1%}}<br>recall=%{{y:.1%}}<br>umbral=%{{customdata:.3f}}"
            "<extra></extra>",
        )
    puntaje = campeon.candidatos[campeon.nombre].promedio
    negativos, positivos = max(int((y == 0).sum()), 1), max(int((y == 1).sum()), 1)
    umbrales = {"alerta (zona gris)": (campeon.umbral, "circle")}
    if campeon.umbral_alto is not None:
        umbrales["se retira"] = (campeon.umbral_alto, "diamond")
    for texto, (corte, simbolo) in umbrales.items():
        alerta = puntaje >= corte
        fpr, tpr = (alerta & (y == 0)).sum() / negativos, (alerta & (y == 1)).sum() / positivos
        fig.add_scatter(
            x=[fpr],
            y=[tpr],
            mode="markers",
            name=f"umbral {texto}",
            marker={"size": 12, "symbol": simbolo, "color": _COLORES[0], "line": {"color": "#fcfcfb", "width": 2}},
            hovertemplate=f"umbral {texto} = {corte:.3f}<br>falsos positivos=%{{x:.1%}}<br>recall=%{{y:.1%}}"
            "<extra></extra>",
        )
    fig.update_layout(
        title="Curva ROC fuera de partición (cada asociado puntuado por un modelo que no lo vio)",
        xaxis={
            "title": "Falsos positivos (de los que se quedan, % con alerta)",
            "tickformat": ".0%",
            "range": [0, 1],
            "gridcolor": "#e1e0d9",
        },
        yaxis={
            "title": "Recall (de los que se retiran, % con alerta)",
            "tickformat": ".0%",
            "range": [0, 1.02],
            "gridcolor": "#e1e0d9",
        },
        template="plotly_white",
        height=620,
        width=720,
        legend={"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
        font={"color": _TINTA},
    )
    return fig


CORTES_CALIBRACION = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1)  # finos cerca de 0: casi todos están ahí


def tabla_calibracion(y, puntaje, cortes=CORTES_CALIBRACION) -> pd.DataFrame:
    """Por grupo de probabilidad: asociados, probabilidad media del modelo y retiro real (sin grupos vacíos).

    Con un 2 % de retiro casi todas las probabilidades están cerca de 0: los deciles quedarían todos ahí y la
    curva tendría dos puntos. Por eso los cortes son fijos y más finos abajo.
    """
    datos = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(puntaje)})
    grupo = pd.cut(datos["p"], bins=list(cortes), include_lowest=True)
    tabla = datos.groupby(grupo, observed=True).agg(asociados=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    return tabla[tabla["asociados"] > 0]


def figura_calibracion(campeon: Campeon):
    """Plotly: curva de calibración del campeón (fuera de partición). None si el puntaje no es una probabilidad.

    Compara la probabilidad media que da el modelo en cada grupo (``tabla_calibracion``) con la tasa de retiro
    real del grupo: sobre la diagonal, el "30 %" del modelo es un 30 % de verdad. Un modelo puede ordenar bien
    (PR-AUC alta) y aun así exagerar o subestimar la probabilidad; el Brier lo resume.
    """
    import plotly.graph_objects as go

    if campeon.metodo != "predict_proba":
        return None
    puntaje, y = campeon.candidatos[campeon.nombre].promedio, campeon.y
    tabla = tabla_calibracion(y, puntaje)
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="calibración perfecta", line={"color": _GRIS, "width": 1, "dash": "dot"}
    )
    fig.add_scatter(
        x=tabla["predicho"],
        y=tabla["real"],
        customdata=np.column_stack([tabla["asociados"], tabla.index.astype(str)]),
        mode="lines+markers",
        name=f"{TITULOS.get(campeon.nombre, campeon.nombre)} (Brier {brier_score_loss(y, puntaje):.4f})",
        line={"color": _COLORES[0], "width": 2},
        # punto más grande = más asociados en el grupo: los grupos chicos (los del medio) son los más ruidosos
        marker={"size": 8 + 16 * np.sqrt(tabla["asociados"] / tabla["asociados"].max()), "line": {"color": "#fcfcfb"}},
        hovertemplate="grupo %{customdata[1]} · %{customdata[0]} asociados<br>probabilidad media del modelo=%{x:.1%}"
        "<br>retiro real del grupo=%{y:.1%}<extra></extra>",
    )
    fig.update_layout(
        title="Calibración del campeón (fuera de partición)",
        xaxis={"title": "Probabilidad que da el modelo", "tickformat": ".0%", "gridcolor": "#e1e0d9"},
        yaxis={"title": "Retiro real en ese grupo", "tickformat": ".0%", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=560,
        width=700,
        legend={"x": 0.02, "y": 0.98},
        font={"color": _TINTA},
    )
    return fig
