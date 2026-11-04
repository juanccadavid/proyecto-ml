"""Segmentación de asociados (no supervisada): KMeans y DBSCAN sobre el comportamiento, y su mapa interactivo.

Aprendizaje **no supervisado**: los algoritmos nunca ven ``#RETIROS``; agrupan a los asociados por cómo usan el
fondo (``abandono.COMPORTAMIENTO``: solicitudes, ahorros, créditos, intención de retiro). El retiro real solo se
mira después, para validar si los segmentos sirven. Los supervisados (``abandono.modelos_supervisados``, el
campeón) aprenden directamente del retiro.

- **KMeans** parte a todos en ``k`` grupos alrededor de ``k`` centros. ``k`` se elige con la silueta (qué tan
  separado queda cada asociado de los otros grupos frente al suyo); ``tabla_k`` da también la inercia (codo).
- **DBSCAN** busca zonas densas: un grupo es un montón de asociados con al menos ``min_samples`` vecinos a
  menos de ``eps``. Quien no cae en ninguna zona densa es **atípico** (etiqueta -1). No hay que darle ``k``.
  ``eps`` sale de la curva de distancias al vecino ``min_samples`` (cuantil ``CUANTIL_EPS``).

El mapa pone a cada asociado en dos ejes que se leen directo (promedios del comportamiento estandarizado):
**señales de salida** (``SALIDA``: cancelaciones, disminuciones, retiros parciales, intención de retiro) y
**vínculo con el fondo** (``VINCULO``: ahorros, créditos, servicios), y dibuja el contorno de cada segmento.
Con PCA, dos componentes solo explican el 28 % de la variación y los segmentos se enciman.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_ml import abandono as a

if TYPE_CHECKING:
    import plotly.graph_objects as go

VARIABLES = a.COMPORTAMIENTO
RANGO_K = range(2, 9)
SALIDA = [v for v, peso in a.SENAL_RIESGO.items() if peso > 0]
VINCULO = [v for v in VARIABLES if v not in SALIDA]
CUANTIL_EPS = 0.9  # eps = distancia al vecino min_samples que solo supera el 10 % de los asociados
ATIPICO = -1
# Paleta de referencia en orden fijo; el gris queda para los atípicos de DBSCAN.
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_GRIS, _TINTA = "#898781", "#0b0b0b"


def escalador() -> Pipeline:
    """Fila cruda o preparada -> comportamiento imputado (mediana) y estandarizado: la entrada de los dos."""
    return Pipeline(
        [
            ("preparar", a.PrepararVariables()),
            ("comportamiento", ColumnTransformer([("c", "passthrough", VARIABLES)])),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def tabla_k(Z: np.ndarray, rango=RANGO_K, random_state: int = 42) -> pd.DataFrame:
    """Silueta (más alta = grupos más separados) e inercia (para el codo) de KMeans con cada ``k``."""
    filas = []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        filas.append({"k": k, "silueta": float(silhouette_score(Z, km.labels_)), "inercia": float(km.inertia_)})
    return pd.DataFrame(filas).set_index("k")


def eps_por_vecinos(Z: np.ndarray, min_samples: int, cuantil: float = CUANTIL_EPS) -> float:
    """``eps`` de DBSCAN: el cuantil de la distancia de cada asociado a su vecino número ``min_samples``."""
    distancias = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1]
    return float(np.quantile(distancias, cuantil))


class Segmentacion:
    """KMeans y DBSCAN sobre el mismo comportamiento escalado, más los dos ejes del mapa.

    Los segmentos de KMeans se numeran por su señal de salida (``abandono.SENAL_RIESGO`` en el centro): el
    segmento 1 es el de más solicitudes de cancelación e intención de retiro. Sin mirar el retiro real.
    """

    def __init__(
        self,
        k: int | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
        random_state: int = 42,
    ):
        self.k = k
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> Segmentacion:
        self.escalador_ = escalador().fit(df)
        Z = self.escalador_.transform(df)
        if self.k is None:  # k por silueta
            self.tabla_k_ = tabla_k(Z, random_state=self.random_state)
            self.k_ = int(self.tabla_k_["silueta"].idxmax())
        else:
            self.tabla_k_ = None
            self.k_ = self.k
        kmeans = KMeans(n_clusters=self.k_, n_init=20, random_state=self.random_state).fit(Z)
        centros = pd.DataFrame(kmeans.cluster_centers_, columns=VARIABLES)
        senal = sum(centros[c] * peso for c, peso in a.SENAL_RIESGO.items())
        orden = senal.sort_values(ascending=False).index  # segmento 1 = más señales de salida
        self.renombrar_ = {int(grupo): i + 1 for i, grupo in enumerate(orden)}
        self.kmeans_ = kmeans

        self.min_samples_ = self.min_samples or 2 * len(VARIABLES)  # regla común: el doble de dimensiones
        self.eps_ = self.eps or eps_por_vecinos(Z, self.min_samples_)
        self.dbscan_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples_).fit(Z)
        nucleos = self.dbscan_.core_sample_indices_
        self.nucleos_ = NearestNeighbors(n_neighbors=1).fit(Z[nucleos])
        self.etiquetas_nucleos_ = self.dbscan_.labels_[nucleos]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.escalador_.transform(df)

    def kmeans(self, df: pd.DataFrame) -> np.ndarray:
        """Segmento de KMeans (1 = más señales de salida) de cada asociado."""
        return np.vectorize(self.renombrar_.get)(self.kmeans_.predict(self.transform(df)))

    def dbscan(self, df: pd.DataFrame) -> np.ndarray:
        """Grupo de DBSCAN (0, 1, ...) o -1 si es atípico. DBSCAN no predice: se asigna el grupo del asociado
        núcleo más cercano si está a menos de ``eps`` (así queda igual para los datos de entrenamiento)."""
        distancia, cercano = self.nucleos_.kneighbors(self.transform(df))
        etiqueta = self.etiquetas_nucleos_[cercano[:, 0]]
        return np.where(distancia[:, 0] <= self.eps_, etiqueta, ATIPICO)

    def plano(self, df: pd.DataFrame) -> np.ndarray:
        """Coordenadas del mapa: (señales de salida, vínculo con el fondo), en desviaciones estándar."""
        Z = self.transform(df)
        columnas = {v: i for i, v in enumerate(VARIABLES)}
        return np.column_stack(
            [Z[:, [columnas[v] for v in SALIDA]].mean(axis=1), Z[:, [columnas[v] for v in VINCULO]].mean(axis=1)]
        )


def perfil(df: pd.DataFrame, segmento: np.ndarray) -> pd.DataFrame:
    """Por segmento: asociados, % del total, tasa de retiro real (que el algoritmo no vio) y comportamiento medio."""
    X = a.preparar(df) if "FECHANACIMIENTO" in df else df
    tabla = X[VARIABLES].groupby(segmento).mean()
    tabla.insert(0, "tasa_retiro", df[a.TARGET].groupby(segmento).mean() if a.TARGET in df else np.nan)
    tabla.insert(0, "pct", pd.Series(segmento).value_counts(normalize=True))
    tabla.insert(0, "asociados", pd.Series(segmento).value_counts())
    return tabla.rename_axis("segmento")


def comparar_con_retiro(df: pd.DataFrame, segmentacion: Segmentacion) -> pd.DataFrame:
    """No supervisado vs. retiro real: si marcar "en riesgo" al segmento 1 de KMeans o a los atípicos de DBSCAN
    encuentra a los que se van (recall), cuántas alertas aciertan (precisión) y el ARI con el retiro."""
    y = df[a.TARGET].to_numpy()
    marcas = {
        "KMeans: segmento 1": segmentacion.kmeans(df) == 1,
        "DBSCAN: atípicos": segmentacion.dbscan(df) == ATIPICO,
    }
    filas = {}
    for nombre, riesgo in marcas.items():
        tp = int((riesgo & (y == 1)).sum())
        filas[nombre] = {
            "marcados": int(riesgo.sum()),
            "tasa_retiro": float(y[riesgo].mean()) if riesgo.any() else float("nan"),
            "precision": tp / riesgo.sum() if riesgo.any() else 0.0,
            "recall": tp / max(int(y.sum()), 1),
            "ari": float(adjusted_rand_score(y, riesgo.astype(int))),
        }
    return pd.DataFrame(filas).T


def _contorno(puntos: np.ndarray, nucleo: float = 0.85) -> np.ndarray:
    """Envolvente convexa del ``nucleo`` (85 %) más central del segmento, cerrada y con un poco de holgura.

    Sin recortar, unos pocos asociados lejanos estiran el contorno sobre todo el mapa.
    """
    from scipy.spatial import ConvexHull, QhullError

    distancia = np.linalg.norm(puntos - np.median(puntos, axis=0), axis=1)
    p = np.unique(puntos[distancia <= np.quantile(distancia, nucleo)], axis=0)
    try:
        casco = p[ConvexHull(p).vertices]  # en orden alrededor del contorno
    except (QhullError, ValueError):  # menos de 3 puntos distintos o todos en línea
        return np.vstack([p, p[:1]])
    centro = casco.mean(axis=0)
    casco = centro + (casco - centro) * 1.06  # que la línea no pase encima de los puntos del borde
    return np.vstack([casco, casco[:1]])


def _nombre_segmento(metodo: str, segmento: int, k: int) -> str:
    if metodo == "DBSCAN":
        return "atípicos" if segmento == ATIPICO else f"grupo denso {segmento + 1}"
    if k == 2:
        return {1: "segmento 1 (riesgo)", 2: "segmento 2 (estable)"}[segmento]
    return f"segmento {segmento}" + (" (más señales de salida)" if segmento == 1 else "")


def figura_mapa(
    segmentacion: Segmentacion,
    df: pd.DataFrame,
    asociado: pd.DataFrame | None = None,
    titulo: str | None = None,
    detalle_individual: bool = True,
) -> go.Figure:
    """Mapa interactivo (señales de salida x vínculo) con un botón para ver KMeans o DBSCAN.

    Cada segmento es un color con su contorno; al pasar el mouse sale el segmento, su tasa de retiro real y el
    comportamiento del asociado. Con ``asociado`` (una fila) se marca dónde cae, con una estrella.

    ``detalle_individual=False`` (datos reales): el mouse muestra solo el segmento y su tasa (agregados), no los
    conteos ni si esa persona se retiró.
    """
    import plotly.graph_objects as go

    # Los conteos dejan a muchos asociados en el mismo punto: se separan un poco solo para dibujarlos.
    plano = segmentacion.plano(df) + np.random.default_rng(0).normal(0, 0.03, (len(df), 2))
    X = a.preparar(df) if "FECHANACIMIENTO" in df else df
    retiro = df[a.TARGET].to_numpy() if a.TARGET in df else None
    etiquetas = {"KMeans": segmentacion.kmeans(df), "DBSCAN": segmentacion.dbscan(df)}
    detalle = np.column_stack(
        [
            X["#INTENCIONRETIROS_ULT3MESES"].fillna(-1),
            X["#SOLICITUD_CANCELACIONAHORROS"].fillna(-1),
            X["#SOLICITUD_DISMINUCIONCUOTAS"].fillna(-1),
            retiro if retiro is not None else np.full(len(X), np.nan),
        ]
    )
    fig = go.Figure()
    visibles: dict[str, list[bool]] = {"KMeans": [], "DBSCAN": []}
    for metodo, segmento in etiquetas.items():
        orden = sorted(set(segmento), key=lambda s: (s == ATIPICO, s))
        for i, s in enumerate(orden):
            en = segmento == s
            nombre = _nombre_segmento(metodo, int(s), segmentacion.k_)
            color = _GRIS if s == ATIPICO else _COLORES[i % len(_COLORES)]
            tasa = f" · se retira el {retiro[en].mean():.1%}" if retiro is not None else ""
            visible = metodo == "KMeans"
            if s != ATIPICO and en.sum() >= 3:  # contorno del segmento (los atípicos no forman grupo)
                borde = _contorno(plano[en])
                fig.add_scatter(
                    x=borde[:, 0],
                    y=borde[:, 1],
                    mode="lines",
                    line={"color": color, "width": 3, "shape": "spline", "smoothing": 0.6},
                    hoverinfo="skip",
                    showlegend=False,
                    visible=visible,
                )
                for m in visibles:
                    visibles[m].append(m == metodo)
            fig.add_scatter(
                x=plano[en, 0],
                y=plano[en, 1],
                mode="markers",
                name=f"{nombre} ({int(en.sum())}{tasa})",
                marker={
                    "size": 7 if s != ATIPICO else 8,
                    "color": color,
                    "opacity": 0.75,
                    "symbol": "x" if s == ATIPICO else "circle",
                    "line": {"color": "#fcfcfb", "width": 0.5},
                },
                customdata=detalle[en] if detalle_individual else None,
                hovertemplate=(
                    f"<b>{nombre}</b>{tasa}<br>intenciones de retiro: %{{customdata[0]}}"
                    "<br>cancelaciones de ahorros: %{customdata[1]}<br>disminuciones de cuotas: %{customdata[2]}"
                    + ("<br>se retiró: %{customdata[3]}" if retiro is not None else "")
                    if detalle_individual
                    else f"<b>{nombre}</b>{tasa}"
                )
                + "<extra></extra>",
                visible=visible,
            )
            for m in visibles:
                visibles[m].append(m == metodo)
    if asociado is not None:
        punto = segmentacion.plano(asociado)
        fig.add_scatter(
            x=punto[:, 0],
            y=punto[:, 1],
            mode="markers",
            name="este asociado",
            marker={"size": 20, "symbol": "star", "color": _TINTA, "line": {"color": "#fcfcfb", "width": 2}},
            hovertemplate="este asociado<extra></extra>",
        )
        for m in visibles:
            visibles[m].append(True)
    botones = [
        {
            "label": f"KMeans (k = {segmentacion.k_})",
            "method": "update",
            "args": [{"visible": visibles["KMeans"]}],
        },
        {
            "label": f"DBSCAN (eps = {segmentacion.eps_:.2f})",
            "method": "update",
            "args": [{"visible": visibles["DBSCAN"]}],
        },
    ]
    fig.update_layout(
        title=titulo or "Segmentos de asociados por comportamiento (sin ver el retiro)",
        xaxis={
            "title": "señales de salida → (cancelaciones, disminuciones, intención de retiro; en desv. estándar)",
            "zeroline": False,
            "gridcolor": "#e1e0d9",
        },
        yaxis={
            "title": "vínculo con el fondo → (ahorros, créditos, servicios)",
            "zeroline": False,
            "gridcolor": "#e1e0d9",
        },
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "y": -0.18},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.12,
                "xanchor": "left",
                "buttons": botones,
                "showactive": True,
            }
        ],
        margin={"t": 110},
        font={"color": _TINTA},
    )
    return fig


def figura_k(tabla: pd.DataFrame, elegido: int) -> go.Figure:
    """Silueta según ``k`` (el elegido, marcado): por qué KMeans usa ese número de segmentos."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=tabla.index,
            y=tabla["silueta"],
            mode="lines+markers",
            line={"color": _COLORES[0], "width": 2},
            marker={"size": 9},
            hovertemplate="k=%{x}<br>silueta=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_scatter(
        x=[elegido],
        y=[tabla.loc[elegido, "silueta"]],
        mode="markers",
        marker={"size": 16, "color": "rgba(0,0,0,0)", "line": {"color": _TINTA, "width": 2}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.update_layout(
        title=f"KMeans: silueta según el número de segmentos (elegido k = {elegido})",
        xaxis={"title": "k (número de segmentos)", "dtick": 1},
        yaxis={"title": "silueta (más alto = más separados)", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    return fig
