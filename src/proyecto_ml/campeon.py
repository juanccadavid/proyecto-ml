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
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from proyecto_ml import abandono as a
from proyecto_ml.datos_campeon import DatosInvalidos as DatosInvalidos
from proyecto_ml.datos_campeon import datos_eleccion as datos_eleccion
from proyecto_ml.datos_campeon import datos_exploracion as datos_exploracion
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.models import get_models

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline

    from proyecto_ml.extractos import Extracto

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


def puntajes_hacia_adelante(modelo: Pipeline, X: pd.DataFrame, y, periodos: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(puntajes, validadas): cada fila de un período se puntúa con un modelo entrenado SOLO con períodos anteriores.

    Es la selección con varias fotos de entrenamiento (``validacion_fuera_de_tiempo`` con 4+ extractos): el mismo
    asociado está en varias fotos, y una partición al azar lo pondría a la vez en entrenamiento y en validación, y
    entrenaría con un período posterior al que valida. Hacia adelante ninguna de las dos cosas pasa. El primer
    período solo entrena: ``validadas`` marca las filas puntuadas (los demás períodos), en el orden de ``X``.
    ``ValueError`` si hay menos de 2 períodos, o si una ventana de entrenamiento o un período validado no tiene las
    dos clases (no hay con qué elegir un umbral): nunca se relaja en silencio.
    """
    y, periodos = np.asarray(y), pd.to_datetime(pd.Series(periodos)).to_numpy()
    orden = np.unique(periodos)
    if len(orden) < 2:
        raise ValueError("La validación hacia adelante necesita al menos 2 períodos de entrenamiento.")
    puntajes, validadas = np.full(len(y), np.nan), periodos > orden[0]
    for periodo in orden[1:]:
        entrena, valida = periodos < periodo, periodos == periodo
        for nombre, filas in (("de entrenamiento antes de", entrena), ("validado", valida)):
            if len(np.unique(y[filas])) < 2:
                raise ValueError(
                    f"El período {nombre} {pd.Timestamp(periodo).date()} no tiene las dos clases (retiros y no "
                    "retiros): no se puede elegir modelo ni umbral con él."
                )
        puntajes[valida] = a.puntaje(clone(modelo).fit(X[entrena], y[entrena]), X[valida])
    return puntajes[validadas], validadas


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
    candidatos: dict[str, Pipeline] | Pipeline,
    df: pd.DataFrame,
    horizonte_meses: int = a.HORIZONTE_MESES,
    meses_prueba: int | None = None,
    folds: int = FOLDS,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
    referencia: str | None = None,
) -> dict:
    """Validación de afuera en el tiempo: elige y entrena con la foto vieja, y mide en la reciente sin tocarla.

    ``df`` es el extracto completo (antes de ``a.foto``). Con ``validacion.particion_temporal``:

    1. foto vieja (``meses_prueba`` antes, por defecto ``horizonte_meses``): predicciones fuera de partición de
       cada candidato -> gana la mayor PR-AUC (el mismo criterio de ``elegir``) -> ``elegir_umbrales``. La
       elección y el umbral salen SOLO de la foto vieja (con un solo ``Pipeline``, no hay elección);
    2. el ganador se entrena con toda la foto vieja;
    3. se mide en la foto reciente con ese umbral fijo: ni la elección, ni el modelo, ni el umbral vieron esas
       etiquetas.

    Con UN solo extracto es una **prueba de estrés, no una estimación fuera de tiempo**: las dos fotos son las
    mismas personas con los mismos datos (solo EDAD y ANTIGUEDAD se corren ``meses_prueba``), y los retiros de la
    foto reciente son asociados que en la vieja cuentan como activos. Mide qué pasa si la etiqueta y los conteos
    sin fecha (``a.CONTEOS_SIN_FECHA``) no están alineados en el tiempo (ver ``config.temporalidad_confirmada``).
    Una validación fuera de tiempo de verdad necesita extractos de fechas distintas. Además:

    - la LISTA de candidatos y, con ``make campeon --optuna``, sus hiperparámetros se armaron mirando datos que
      incluyen la foto reciente: eso no se repite aquí.

    Con ``referencia`` (el nombre del campeón de ``make campeon``), ``"referencia"`` trae además las métricas de ESE
    candidato con el mismo procedimiento (umbral de la foto vieja), para comparar: su elección no es ciega.
    """
    from proyecto_ml.validacion import particion_temporal

    vieja, reciente = particion_temporal(df, horizonte_meses, meses_prueba or horizonte_meses)
    return {
        "t0_entrenamiento": str(vieja[a.FECHA_OBSERVACION].iloc[0].date()),
        "t0_prueba": str(reciente[a.FECHA_OBSERVACION].iloc[0].date()),
        **elegir_y_medir(candidatos, vieja, reciente, folds, beta, precision_min, referencia),
    }


def elegir_y_medir(
    candidatos: dict[str, Pipeline] | Pipeline,
    entrenamiento: pd.DataFrame,
    prueba: pd.DataFrame,
    folds: int = FOLDS,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
    referencia: str | None = None,
) -> dict:
    """El procedimiento ciego de ``validacion_temporal`` y ``validacion_fuera_de_tiempo``.

    Solo con ``entrenamiento``: puntajes de selección de cada candidato -> gana la mayor PR-AUC ->
    ``elegir_umbrales`` con esos mismos puntajes -> se entrena con todo. Después se mide en ``prueba`` (que no
    participó en nada de lo anterior) con ese umbral fijo. Los puntajes de selección dependen de cuántas fotos
    (``FECHA_OBSERVACION``) tiene ``entrenamiento``:

    - **una**: ``StratifiedKFold`` (cada asociado aparece una sola vez; si un identificador se repite, error);
    - **varias**: ``puntajes_hacia_adelante`` (cada período se valida con un modelo entrenado con los anteriores):
      el mismo asociado está en varias fotos y una partición al azar mezclaría sus filas y el futuro.

    Los identificadores (``extractos.LLAVES``) solo sirven para revisar repetidos: no son variables del modelo
    (``abandono.preparar`` usa una lista fija).
    """
    from proyecto_ml.extractos import LLAVES

    if not isinstance(candidatos, dict):
        candidatos = {"modelo": candidatos}
    Xe, ye = a.separar(entrenamiento)
    Xp, yp = a.separar(prueba)
    if int(yp.sum()) < folds or int((yp == 0).sum()) < folds:
        raise ValueError(f"La foto de prueba tiene menos de {folds} asociados en alguna clase: usa más datos.")
    periodos = entrenamiento.get(a.FECHA_OBSERVACION)
    varios = periodos is not None and len(pd.unique(periodos)) > 1
    por_periodo = [entrenamiento] if periodos is None else [g for _, g in entrenamiento.groupby(periodos)]
    for columna in [c for c in LLAVES if c in entrenamiento]:
        if any(g[columna].duplicated().any() for g in por_periodo):
            raise ValueError(f"{columna} se repite dentro de una misma foto de entrenamiento: no se puede validar.")
    if varios:
        estrategia = "hacia adelante (cada período validado con un modelo de los anteriores)"
        hacia_adelante = {n: puntajes_hacia_adelante(m, Xe, ye, periodos) for n, m in candidatos.items()}
        oof = {n: p for n, (p, _) in hacia_adelante.items()}
        y_sel = np.asarray(ye)[next(iter(hacia_adelante.values()))[1]]
    else:
        if int(ye.sum()) < folds or int((ye == 0).sum()) < folds:
            raise ValueError(f"La foto de entrenamiento tiene menos de {folds} asociados en alguna clase.")
        estrategia = f"StratifiedKFold({folds}) en una sola foto"
        oof = {n: puntajes_fuera_de_particion(m, Xe, ye, folds, repeticiones=1)[0][0] for n, m in candidatos.items()}
        y_sel = np.asarray(ye)
    pr_auc_vieja = {n: float(average_precision_score(y_sel, p)) for n, p in oof.items()}
    elegido = max(pr_auc_vieja, key=lambda n: pr_auc_vieja[n])

    def medir(nombre: str) -> dict:
        umbrales = elegir_umbrales(y_sel, oof[nombre], beta, precision_min)
        entrenado = clone(candidatos[nombre]).fit(Xe, ye)
        metodo = a.metodo_puntaje(entrenado)
        return {
            "umbral": umbrales["umbral"],
            **compute_metrics(yp, a.puntaje(entrenado, Xp), umbrales["umbral"], metodo),
        }

    resultado = {
        "elegido": elegido,
        "pr_auc_vieja": pr_auc_vieja,
        "seleccion": estrategia,
        "filas_seleccion": len(y_sel),
        "asociados_entrenamiento": len(entrenamiento),
        "retiros_entrenamiento": int(ye.sum()),
        "asociados_prueba": len(prueba),
        "retiros_prueba": int(yp.sum()),
        **medir(elegido),
    }
    if referencia is not None and referencia in candidatos:
        resultado["referencia"] = {"modelo": referencia, **medir(referencia)}
    return resultado


def validacion_fuera_de_tiempo(
    candidatos: dict[str, Pipeline] | Pipeline,
    extractos: list[Extracto],
    horizonte_meses: int = a.HORIZONTE_MESES,
    folds: int = FOLDS,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
    referencia: str | None = None,
) -> dict:
    """Validación fuera de tiempo con 3 o más extractos de fechas distintas (``extractos.Extracto``).

    Pares consecutivos (E1 -> E2, E2 -> E3, ...): cada uno es una foto con las variables del extracto viejo,
    medidas a SU fecha, y el objetivo del siguiente (``extractos.foto_entre_extractos``). Se entrena y se elige con
    todos los pares menos el último y se mide en el último con ``elegir_y_medir``: la prueba usa variables y
    etiquetas posteriores a todo lo que se usó para elegir. Es la validación que la prueba de estrés de un solo
    extracto no puede dar; necesita extractos reales (docs/datos_reales.md).

    Con 3 extractos hay una foto de entrenamiento (cada asociado una vez) y la selección es estratificada. Con 4 o
    más, el mismo asociado está en varias fotos de entrenamiento y la selección va hacia adelante
    (``puntajes_hacia_adelante``). En los dos casos la prueba incluye a asociados que el modelo vio en fotos
    anteriores: es el uso real (predecir a los asociados actuales en el período siguiente), no una fuga.
    """
    from proyecto_ml import extractos as ex

    ordenados = sorted(extractos, key=lambda e: e.fecha)
    if len(ordenados) < 3:
        raise ValueError(
            f"Hacen falta al menos 3 extractos de fechas distintas (hay {len(ordenados)}): dos para entrenar con un "
            "par y uno más para medir con el par siguiente."
        )
    pares = list(zip(ordenados, ordenados[1:], strict=False))
    fotos = [ex.foto_entre_extractos(v, n, horizonte_meses) for v, n in pares]
    entrenamiento, prueba = pd.concat(fotos[:-1], ignore_index=True), fotos[-1]
    return {
        "extractos": [f"{e.nombre} ({e.fecha.date()})" for e in ordenados],
        "t0_entrenamiento": [str(v.fecha.date()) for v, _ in pares[:-1]],
        "t0_prueba": str(pares[-1][0].fecha.date()),
        "sin_seguimiento": [int(f.attrs.get("sin_seguimiento", 0)) for f in fotos],
        **elegir_y_medir(candidatos, entrenamiento, prueba, folds, beta, precision_min, referencia),
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
        if propio is None:  # campeón cargado de disco: no trae las particiones
            return pd.DataFrame()
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


def firma(carpeta: Path) -> str | None:
    """Huella de los artefactos del campeón en ``carpeta`` (None si falta alguno): cambia cada vez que se guarda.

    Contenido de ``campeon.json`` (trae la fecha de la elección y las métricas) + tamaño y fecha de modificación de
    los dos .joblib. Sirve de llave de caché: el simulador vuelve a cargar el campeón justo cuando cambia, sin leer
    ni deserializar los modelos en cada interacción.
    """
    import hashlib

    rutas = [carpeta / ARCHIVO_MODELO, carpeta / ARCHIVO_GRUPOS, carpeta / ARCHIVO_INFO]
    if not all(r.exists() for r in rutas):
        return None
    h = hashlib.sha256(rutas[2].read_bytes())
    for ruta in rutas[:2]:
        estado = ruta.stat()
        h.update(f"{ruta.name}:{estado.st_size}:{estado.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


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
