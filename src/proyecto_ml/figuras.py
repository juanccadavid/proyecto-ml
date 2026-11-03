"""Marca de origen para las figuras: una figura suelta (HTML) debe decir si sus datos son sintéticos."""

from __future__ import annotations

from typing import TYPE_CHECKING

from proyecto_ml.config import es_experimental

if TYPE_CHECKING:
    import plotly.graph_objects as go

MARCA = "DATOS SINTÉTICOS · resultado experimental"


def marcar_origen(fig: go.Figure, origen: str | None = None) -> go.Figure:
    """Con datos sintéticos, agrega la marca arriba a la derecha (con datos reales, no cambia nada)."""
    if es_experimental(origen):
        fig.add_annotation(
            text=MARCA,
            xref="paper",
            yref="paper",
            x=1,
            y=1.08,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font={"size": 11, "color": "#d03b3b"},
        )
    return fig
