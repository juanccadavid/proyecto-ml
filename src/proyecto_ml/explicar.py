"""SHAP: por qué el modelo le dio a ESTE asociado su puntaje (lo usan el simulador y make campeon).

La importancia por permutación (``evaluate.importancia_variables``) dice qué variables pesan en el modelo en
general; SHAP reparte el puntaje de cada asociado entre sus variables::

    puntaje del asociado = base + contribución de EDAD + contribución de SALARIO + ...

``base`` es el puntaje promedio del ``fondo`` (asociados de referencia). Una contribución positiva sube el
riesgo de este asociado frente a ese promedio; una negativa lo baja. La escala es la del puntaje del modelo:
puntos de probabilidad (logística, árboles) o de distancia a la frontera (Ridge).

Se explican las variables originales de ``abandono.preparar`` (EDAD, SEXO...), no las columnas del OneHot:
así se lee igual con cualquier modelo. Por eso se usa ``shap.PermutationExplainer``, que solo necesita el
puntaje: sirve para los cuatro candidatos y para el ModeloZonas. El masker de SHAP solo admite números, así que
las categóricas viajan como códigos (``_Codigos``) y vuelven a texto antes de llegar al modelo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proyecto_ml import abandono as a

if TYPE_CHECKING:
    import plotly.graph_objects as go

VARIABLES = a.NUMERICAS + a.CATEGORICAS
N_FONDO = 100  # asociados de referencia: más = base más estable, pero cada explicación tarda más
_SUBE, _BAJA = "#e34948", "#2a78d6"  # par divergente de la paleta de referencia: rojo sube el riesgo, azul lo baja
_TINTA, _GRIS = "#0b0b0b", "#898781"


def _preparadas(df: pd.DataFrame) -> pd.DataFrame:
    """Las variables del modelo, venga ``df`` crudo (con FECHANACIMIENTO) o ya preparado."""
    return a.preparar(df) if "FECHANACIMIENTO" in df else df[VARIABLES]


class _Codigos:
    """Categóricas <-> códigos numéricos (vacío = NaN), para pasar por el masker de SHAP y volver."""

    def __init__(self, X: pd.DataFrame):
        self.categorias = {c: pd.Index(X[c].dropna().unique()) for c in a.CATEGORICAS}

    def ampliar(self, X: pd.DataFrame) -> None:
        """Agrega al final las categorías no vistas (p. ej. una ciudad nueva): los códigos viejos no cambian."""
        for c, categorias in self.categorias.items():
            nuevas = pd.Index(X[c].dropna().unique()).difference(categorias, sort=False)
            self.categorias[c] = categorias.append(nuevas)

    def a_numeros(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c, categorias in self.categorias.items():
            X[c] = pd.Series(categorias.get_indexer(X[c]), index=X.index, dtype=float).where(X[c].notna())
        return X.to_numpy(dtype=float)

    def a_tabla(self, valores: np.ndarray) -> pd.DataFrame:
        X = pd.DataFrame(valores, columns=VARIABLES)
        for c, categorias in self.categorias.items():
            codigo = X[c]
            texto = np.asarray(categorias, dtype=object)[codigo.fillna(0).astype(int)]
            X[c] = pd.Series(texto, index=X.index, dtype=object).where(codigo.notna())
        return X


class Explicador:
    """SHAP del ``modelo`` (Pipeline o ModeloZonas) frente a ``fondo`` (``n_fondo`` asociados al azar)."""

    def __init__(self, modelo, fondo: pd.DataFrame, n_fondo: int = N_FONDO, random_state: int = 0):
        import shap

        self.modelo = modelo
        self.random_state = random_state
        self.metodo = a.metodo_puntaje(modelo)
        fondo = _preparadas(fondo)
        self.fondo = fondo.sample(min(n_fondo, len(fondo)), random_state=random_state)
        self.codigos = _Codigos(fondo)
        masker = shap.maskers.Independent(self.codigos.a_numeros(self.fondo), max_samples=len(self.fondo))
        self._shap = shap.PermutationExplainer(self._puntaje, masker, seed=random_state)
        self.base = float(self._puntaje(self.codigos.a_numeros(self.fondo)).mean())

    def _puntaje(self, valores: np.ndarray) -> np.ndarray:
        return a.puntaje(self.modelo, self.codigos.a_tabla(valores))

    def contribuciones(self, filas: pd.DataFrame) -> pd.DataFrame:
        """Filas x variables: cuánto sube (+) o baja (-) cada variable el puntaje de cada asociado."""
        X = _preparadas(filas)
        self.codigos.ampliar(X)
        # PermutationExplainer sortea permutaciones con el generador global de numpy y solo lo siembra al crearse:
        # sin volver a sembrar, el mismo asociado tendría una explicación un poco distinta en cada llamada.
        np.random.seed(self.random_state)  # noqa: NPY002 (es el generador que usa SHAP por dentro)
        resultado = self._shap(self.codigos.a_numeros(X), silent=True)
        return pd.DataFrame(resultado.values, index=X.index, columns=VARIABLES)

    def explicar(self, fila: pd.DataFrame) -> pd.DataFrame:
        """Un asociado (la primera fila): variable, su valor y su contribución, de la mayor a la menor |contribución|.

        ``base`` + la suma de ``contribucion`` da exactamente el puntaje del modelo para ese asociado.
        """
        X = _preparadas(fila).iloc[[0]]
        contribucion = self.contribuciones(X).iloc[0]
        tabla = pd.DataFrame({"variable": VARIABLES, "valor": X.iloc[0].to_numpy(), "contribucion": contribucion})
        orden = tabla["contribucion"].abs().sort_values(ascending=False, kind="stable").index
        return tabla.loc[orden].reset_index(drop=True)


def importancia(contribuciones: pd.DataFrame) -> pd.Series:
    """Importancia global con SHAP: la |contribución| promedio de cada variable, de mayor a menor."""
    return contribuciones.abs().mean().sort_values(ascending=False).rename("shap_medio")


def _texto_valor(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "vacío"
    if isinstance(valor, float):
        return f"{valor:,.0f}" if abs(valor) >= 1000 else f"{valor:.3g}"
    return str(valor)


def _formato(metodo: str) -> tuple[str, str]:
    """(formato d3 para Plotly, nombre de la escala)."""
    return (".1%", "probabilidad de retiro") if metodo == "predict_proba" else (".2f", "puntaje del modelo")


def _texto_puntaje(valor: float, metodo: str, cambio: bool = False) -> str:
    """0.123 -> '12.3%' (o '+1.5 pts' si es un cambio); en Ridge, '-0.80' (o '+0.15')."""
    if metodo == "predict_proba":
        return f"{valor * 100:+.1f} pts" if cambio else f"{valor:.1%}"
    return f"{valor:+.2f}" if cambio else f"{valor:.2f}"


def figura_explicacion(
    tabla: pd.DataFrame, base: float, metodo: str, top: int = 8, etiquetas: dict | None = None
) -> go.Figure:
    """Cascada de SHAP de un asociado: desde la base, cada variable sube (rojo) o baja (azul) el puntaje.

    Muestra las ``top`` variables de mayor |contribución|; el resto se suma en "otras N variables".
    """
    import plotly.graph_objects as go

    formato, escala = _formato(metodo)
    principales, resto = tabla.head(top), tabla.iloc[top:]
    nombres = [
        f"{(etiquetas or {}).get(v, v)} = {_texto_valor(x)}"
        for v, x in zip(principales["variable"], principales["valor"], strict=True)
    ]
    valores = principales["contribucion"].tolist()
    if len(resto):
        nombres.append(f"otras {len(resto)} variables")
        valores.append(float(resto["contribucion"].sum()))
    puntaje = base + sum(valores)
    # El valor va en la etiqueta de cada fila (no junto a la barra): así no choca con el eje ni se corta.
    filas = [
        f"promedio del fondo: {_texto_puntaje(base, metodo)}",
        *[f"{n}  ({_texto_puntaje(v, metodo, cambio=True)})" for n, v in zip(nombres, valores, strict=True)],
        f"<b>este asociado: {_texto_puntaje(puntaje, metodo)}</b>",
    ]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=filas,
            x=[base, *valores, puntaje],
            measure=["absolute", *["relative"] * len(valores), "total"],
            increasing={"marker": {"color": _SUBE}},
            decreasing={"marker": {"color": _BAJA}},
            totals={"marker": {"color": _GRIS}},
            connector={"line": {"color": "#c3c2b7", "width": 1}},
            textposition="none",
            hovertemplate="%{y}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")  # se lee de arriba (base) hacia abajo (este asociado)
    fig.update_xaxes(tickformat=formato, title=escala, gridcolor="#e1e0d9")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=120 + 34 * (len(nombres) + 2),
        margin={"t": 30, "b": 10, "l": 10, "r": 20},
        font={"color": _TINTA},
    )
    return fig


def figura_importancia(importancia_shap: pd.Series, metodo: str, titulo: str, top: int = 15) -> go.Figure:
    """Barras horizontales: |contribución SHAP| promedio de cada variable (la más importante arriba)."""
    import plotly.graph_objects as go

    formato, escala = _formato(metodo)
    datos = importancia_shap.head(top)[::-1]
    fig = go.Figure(
        go.Bar(
            x=datos.to_numpy(),
            y=datos.index,
            orientation="h",
            marker={"color": _BAJA, "cornerradius": 4},
            hovertemplate=f"%{{y}}<br>|contribución| media = %{{x:{formato}}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis={"title": f"|contribución SHAP| promedio ({escala})", "tickformat": formato},
        template="plotly_white",
        height=140 + 28 * len(datos),
        margin={"l": 10},
    )
    return fig
