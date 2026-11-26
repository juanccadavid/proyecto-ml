"""Regresión: estimar el SALARIO del asociado con sus demás variables (MAE, MSE, RMSE y R²).

El abandono es una clasificación (se retira sí/no); esta es la contraparte de regresión, con un uso real: el
5 % de los asociados no reporta salario (nómina no reportada) y el modelo da una estimación para ellos. Se
entrena solo con quienes sí lo reportan y se valida con **KFold** (en regresión no hay clases que estratificar).

Las métricas, en pesos:

- **MAE** (error absoluto medio): cuánto se equivoca en promedio. Se lee directo: "±X pesos".
- **MSE** (error cuadrático medio): castiga más los errores grandes. Está en pesos², no se lee directo.
- **RMSE** (raíz del MSE): vuelve a pesos; si es mucho mayor que el MAE, hay pocos errores muy grandes.
- **R²**: fracción de la variación del salario que explica el modelo. 0 = igual que predecir el promedio
  (``DummyRegressor``, la referencia), 1 = perfecto, negativo = peor que el promedio.

Las variables son las del abandono (``abandono.preparar``) sin SALARIO.

**Evaluación sin sesgo de selección.** Elegir el mejor por R² en validación cruzada y reportar ese mismo R² lo
infla: es el máximo de varias estimaciones ruidosas sobre las mismas particiones. Por eso ``separar_reserva`` aparta
antes una **reserva** (``FRACCION_RESERVA``) que no se usa para elegir: la validación cruzada elige con el resto y
``medir_en_reserva`` mide una sola vez al elegido y a la referencia, entrenados con ese resto.

**Estimación transversal.** El salario y sus predictores (edad, antigüedades, empresa, profesión...) se miden en el
mismo corte. El modelo describe cómo se asocia el salario con esas variables en esa foto: sirve para estimar el de
quien no lo reporta en el mismo corte, no para pronosticar salarios futuros ni para leer efectos causales (más
antigüedad no "produce" más salario en el modelo; solo van juntos en los datos).

**Error proporcional al salario.** Con el extracto real el error crece con el salario (heterocedasticidad): un MAE
en pesos promedia errores muy distintos. Por eso compiten también dos modelos que aprenden ``log(SALARIO)``
(``_en_logaritmo``), como la ecuación de salarios de Mincer, y la estimación lleva un **intervalo de predicción
multiplicativo** [ŷ ÷ f, ŷ × f] (``calibrar_intervalo``). f sale de los errores relativos fuera de partición de la
validación cruzada (conformal con corrección de muestra finita), **uno por tramo de salario predicho** (conformal
"Mondrian"): con un solo f, en el extracto real el intervalo cubría el 80 % en promedio pero el 67 % en los salarios
más altos. La reserva mide si cubre lo prometido, en total y por tramo (``error_por_tramo`` con ``intervalo``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml.features import build_preprocessor

OBJETIVO = "SALARIO"
FUGA: list[str] = []
NUMERICAS = [c for c in a.NUMERICAS if c != OBJETIVO]
CATEGORICAS = [c for c in a.CATEGORICAS if c not in FUGA]
RECORTE = [c for c in a.COLUMNAS_RECORTE if c != OBJETIVO]
FRACCION_RESERVA = 0.2  # no se usa para elegir el modelo: mide al elegido una sola vez
NIVEL_INTERVALO = 0.8  # 8 de cada 10 salarios reales deberían caer dentro del intervalo de predicción
ERRORES = ["MAE", "MSE", "RMSE"]
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}
TITULOS = {
    "media": "Promedio (referencia)",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "hist_gb": "HistGradientBoosting",
    "ridge_log": "Ridge en logaritmo",
    "hist_gb_log": "HistGradientBoosting en logaritmo",
}


def _en_logaritmo(estimador) -> TransformedTargetRegressor:
    """Aprende log(SALARIO) y devuelve pesos: el error pasa a ser relativo (errar un 10 % pesa igual en cualquier
    salario). Devuelve la media geométrica, algo menor que la aritmética: si eso empeora el R², la validación cruzada
    lo dice."""
    return TransformedTargetRegressor(regressor=estimador, func=np.log, inverse_func=np.exp)


def modelos(random_state: int = 42) -> dict[str, Pipeline]:
    """La referencia (predice siempre el promedio) y tres regresores, con el preprocesamiento del abandono."""
    estimadores = {
        "media": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "hist_gb": HistGradientBoostingRegressor(random_state=random_state),
        "ridge_log": _en_logaritmo(Ridge(alpha=1.0)),
        "hist_gb_log": _en_logaritmo(HistGradientBoostingRegressor(random_state=random_state)),
    }
    # PrepararVariables entrega también SALARIO; el ColumnTransformer lo descarta (remainder="drop").
    return {
        nombre: Pipeline(
            [
                ("preparar", a.PrepararVariables()),
                ("preprocess", build_preprocessor(NUMERICAS, CATEGORICAS, clip_outliers="iqr", clip_columns=RECORTE)),
                ("model", estimador),
            ]
        )
        for nombre, estimador in estimadores.items()
    }


def separar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (filas crudas, sin el abandono) e y (SALARIO), solo de quienes reportan un salario plausible.

    Un salario de relleno (987.654) o de miles de millones no se aprende como si fuera cierto: esa fila queda,
    como la de quien no reporta salario, entre las que el modelo estima (``abandono.SALARIO_PLAUSIBLE``).
    """
    salario = pd.to_numeric(df[OBJETIVO], errors="coerce")
    con_salario = df[salario.between(*a.SALARIO_PLAUSIBLE)]
    return con_salario.drop(columns=[a.TARGET], errors="ignore"), pd.to_numeric(con_salario[OBJETIVO]).astype(float)


def separar_reserva(
    X: pd.DataFrame, y: pd.Series, fraccion: float = FRACCION_RESERVA, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """(X, X_reserva, y, y_reserva): la reserva se aparta ANTES de elegir y no se mira hasta ``medir_en_reserva``."""
    X_eleccion, X_reserva, y_eleccion, y_reserva = train_test_split(X, y, test_size=fraccion, random_state=random_state)
    return X_eleccion, X_reserva, y_eleccion, y_reserva


def predecir_reserva(candidatos: dict[str, Pipeline], nombres: list[str], X, y, X_reserva) -> dict[str, np.ndarray]:
    """Entrena ``nombres`` (el elegido y la referencia) con la parte de elección y predice la reserva."""
    return {n: clone(candidatos[n]).fit(X, y).predict(X_reserva) for n in nombres}


def medir_en_reserva(
    candidatos: dict[str, Pipeline], nombres: list[str], X, y, X_reserva, y_reserva
) -> dict[str, dict[str, float]]:
    """Las métricas de ``predecir_reserva``: cada modelo se mide una sola vez en la reserva."""
    return {n: metricas(y_reserva, p) for n, p in predecir_reserva(candidatos, nombres, X, y, X_reserva).items()}


def metricas(y, prediccion) -> dict[str, float]:
    """MAE, MSE, RMSE y R², más ``MdAPE``: la mediana del error relativo |real − predicho| / real. Se lee como
    "a la mitad de los asociados se les erra por menos de este porcentaje de su salario", sin que los salarios
    altos dominen como en el MAE."""
    y, prediccion = np.asarray(y, dtype=float), np.asarray(prediccion, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, prediccion)),
        "MSE": float(mean_squared_error(y, prediccion)),
        "RMSE": float(root_mean_squared_error(y, prediccion)),
        "R2": float(r2_score(y, prediccion)),
        "MdAPE": float(np.median(np.abs(y - prediccion) / y)),
    }


def _positiva(prediccion) -> np.ndarray:
    """Las predicciones por debajo del salario mínimo plausible se llevan a ese mínimo (Ridge puede dar negativos)."""
    return np.clip(np.asarray(prediccion, dtype=float), a.SALARIO_PLAUSIBLE[0], None)


def amplitud_conformal(y, prediccion_oof, nivel: float = NIVEL_INTERVALO) -> float:
    """q tal que [ŷ·e^−q, ŷ·e^q] cubre ``nivel`` de los salarios: el cuantil conformal de |log(real / predicho)|.

    ``prediccion_oof`` debe venir de modelos que no vieron cada fila (``prediccion_fuera_de_particion``): con
    predicciones de entrenamiento los errores salen chicos y el intervalo, demasiado estrecho. El rango
    ⌈(n + 1)·nivel⌉ es la corrección de muestra finita; si supera n, no hay garantía y se devuelve infinito.
    """
    residuo = np.sort(np.abs(np.log(np.asarray(y, dtype=float) / _positiva(prediccion_oof))))
    rango = int(np.ceil((len(residuo) + 1) * nivel))
    return float(residuo[rango - 1]) if rango <= len(residuo) else float("inf")


@dataclass(frozen=True)
class Intervalo:
    """Intervalo multiplicativo [ŷ ÷ e^q, ŷ × e^q] con un q por tramo de salario predicho.

    ``bordes`` son los cortes entre tramos (cuantiles de las predicciones fuera de partición) y ``q`` la amplitud en
    escala logarítmica de cada tramo. Con un solo tramo es el intervalo conformal de siempre.
    """

    bordes: np.ndarray
    q: np.ndarray
    nivel: float

    def tramo(self, prediccion) -> np.ndarray:
        return np.searchsorted(self.bordes, _positiva(prediccion), side="right")

    def limites(self, prediccion) -> tuple[np.ndarray, np.ndarray]:
        base, q = _positiva(prediccion), self.q[self.tramo(prediccion)]
        return base * np.exp(-q), base * np.exp(q)


def calibrar_intervalo(y, prediccion_oof, nivel: float = NIVEL_INTERVALO, tramos: int = 5) -> Intervalo:
    """Un ``amplitud_conformal`` por tramo del salario predicho (conformal Mondrian): cubre ``nivel`` en cada tramo y
    no solo en promedio, aunque el error relativo crezca con el salario. Necesita predicciones fuera de partición."""
    y, base = np.asarray(y, dtype=float), _positiva(prediccion_oof)
    bordes = np.quantile(base, np.linspace(0, 1, tramos + 1)[1:-1])
    grupo = np.searchsorted(bordes, base, side="right")
    q = np.array([amplitud_conformal(y[grupo == k], base[grupo == k], nivel) for k in range(tramos)])
    return Intervalo(bordes=bordes, q=q, nivel=nivel)


def cobertura(y, prediccion, intervalo: Intervalo) -> dict[str, float]:
    """Qué fracción de los salarios reales cae dentro del intervalo, frente a la prometida, y sus factores."""
    inferior, superior = intervalo.limites(prediccion)
    y = np.asarray(y, dtype=float)
    return {
        "nivel": intervalo.nivel,
        "cobertura": float(np.mean((y >= inferior) & (y <= superior))),
        "factor_min": float(np.exp(intervalo.q.min())),
        "factor_max": float(np.exp(intervalo.q.max())),
    }


def error_por_tramo(y, prediccion, tramos: int = 5, intervalo: Intervalo | None = None) -> pd.DataFrame:
    """El error según el nivel de salario predicho (quintiles por defecto): si crece con el salario, los
    residuos son heterocedásticos y un solo MAE promedia errores muy distintos. Con ``intervalo``
    (``calibrar_intervalo``) agrega la cobertura y el factor del intervalo en cada tramo. Cada fila es un agregado
    de un quinto de la muestra (nunca un asociado): sirve también como figura de los resultados con datos reales."""
    y, prediccion = np.asarray(y, dtype=float), np.asarray(prediccion, dtype=float)
    tramo = pd.qcut(prediccion, tramos, labels=False, duplicates="drop")
    tabla = pd.DataFrame({"tramo": tramo, "real": y, "predicho": prediccion, "error": np.abs(y - prediccion)})
    tabla["error_relativo"] = tabla["error"] / tabla["real"]
    columnas = {
        "asociados": ("real", "size"),
        "predicho_medio": ("predicho", "mean"),
        "real_medio": ("real", "mean"),
        "MAE": ("error", "mean"),
        "MdAPE": ("error_relativo", "median"),
    }
    if intervalo is not None:
        inferior, superior = intervalo.limites(prediccion)
        tabla["dentro"] = (y >= inferior) & (y <= superior)
        tabla["factor"] = superior / _positiva(prediccion)
        columnas["cobertura"] = ("dentro", "mean")
        columnas["factor"] = ("factor", "mean")
    salida = tabla.groupby("tramo").agg(**columnas)
    return salida.rename(index=lambda t: int(t) + 1).rename_axis("tramo_predicho")


def comparar(candidatos: dict[str, Pipeline], X, y, folds: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Validación cruzada KFold: promedio y desviación de cada métrica, mejor R² arriba."""
    cv = KFold(folds, shuffle=True, random_state=random_state)
    filas = {}
    for nombre, modelo in candidatos.items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=SCORING)
        promedio = {m: float(r[f"test_{m}"].mean()) for m in SCORING}
        filas[nombre] = {
            **{m: -promedio[m] for m in ERRORES},  # sklearn maximiza: los errores llegan con signo negativo
            "R2": promedio["R2"],
            "R2_std": float(r["test_R2"].std()),
        }
    return pd.DataFrame.from_dict(filas, orient="index").sort_values("R2", ascending=False)


def prediccion_fuera_de_particion(modelo: Pipeline, X, y, folds: int = 5, random_state: int = 0) -> np.ndarray:
    """La predicción de cada asociado hecha por un modelo que no lo vio (para la figura real vs. predicho)."""
    return cross_val_predict(clone(modelo), X, y, cv=KFold(folds, shuffle=True, random_state=random_state))


def figura_real_vs_predicho(y, prediccion, titulo: str):
    """Plotly: salario real vs. predicho; sobre la diagonal, predicción perfecta."""
    import plotly.graph_objects as go

    y, prediccion = np.asarray(y), np.asarray(prediccion)
    bajo, alto = float(min(y.min(), prediccion.min())), float(max(y.max(), prediccion.max()))
    fig = go.Figure()
    fig.add_scatter(
        x=[bajo, alto],
        y=[bajo, alto],
        mode="lines",
        name="predicción perfecta",
        line={"color": "#898781", "width": 1, "dash": "dot"},
    )
    fig.add_scatter(
        x=y,
        y=prediccion,
        mode="markers",
        name="asociado",
        marker={"size": 8, "color": "#2a78d6", "opacity": 0.5, "line": {"color": "#fcfcfb", "width": 1}},
        hovertemplate="real=$%{x:,.0f}<br>predicho=$%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        title=titulo,
        xaxis={"title": "Salario real", "tickformat": "$,.0f", "gridcolor": "#e1e0d9"},
        yaxis={"title": "Salario predicho (fuera de partición)", "tickformat": "$,.0f", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=560,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def figura_por_tramo(tabla: pd.DataFrame, titulo: str):
    """Plotly con datos AGREGADOS (``error_por_tramo``): salario real medio y predicho medio por tramo, con su MAE.
    Es la figura de los datos reales: no lleva ni un punto por asociado, así que no expone salarios individuales."""
    import plotly.graph_objects as go

    x = [f"Tramo {t}" for t in tabla.index]
    cobertura = tabla["cobertura"] if "cobertura" in tabla else pd.Series(np.nan, index=tabla.index)
    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=tabla["real_medio"],
        name="real (promedio del tramo)",
        marker_color="#2a78d6",
        error_y={"type": "data", "array": tabla["MAE"], "color": "#898781"},
        customdata=np.column_stack([tabla["asociados"], tabla["MAE"], tabla["MdAPE"], cobertura]),
        hovertemplate="%{x}: %{customdata[0]} asociados<br>real medio $%{y:,.0f}<br>MAE $%{customdata[1]:,.0f}"
        "<br>MdAPE %{customdata[2]:.1%}<br>dentro del intervalo %{customdata[3]:.1%}<extra></extra>",
    )
    fig.add_scatter(
        x=x, y=tabla["predicho_medio"], mode="lines+markers", name="predicho (promedio)", line={"color": "#eb6834"}
    )
    fig.update_layout(
        title=titulo,
        xaxis={"title": "Tramo de salario predicho (quintiles; barras de error = MAE del tramo)"},
        yaxis={"title": "Salario", "tickformat": "$,.0f", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=520,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig
