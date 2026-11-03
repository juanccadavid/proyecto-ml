"""Registro de los experimentos de abandono en MLflow (experimento ``abandono``).

Reúsa la configuración de ``train.py`` (``MLFLOW_TRACKING_URI`` del .env, por defecto ``mlflow.db``).
Cada script de abandono abre una corrida "padre" con los parámetros de los datos y los gráficos
adjuntos, y una corrida hija por modelo con:

- parámetros: datos, filas, limpieza y los hiperparámetros del modelo (``model__...``);
- métricas con prefijo: ``cv_`` (validación cruzada), ``nuevos_`` (asociados generados que no vio),
  ``entrenamiento_``, ``esc_...`` (escenarios);
- el modelo entrenado (skops si se puede, si no cloudpickle).

El campeón (``make campeon``) además queda en el registro de modelos como ``abandono`` con el alias
``campeon``: ``mlflow.sklearn.load_model("models:/abandono@campeon")``.
Los modelos ya no se guardan sueltos en ``models/``: ahí solo queda el campeón.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

from proyecto_ml.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    import pandas as pd

EXPERIMENTO = "abandono"
MODELO_REGISTRADO = "abandono"
ALIAS_CAMPEON = "campeon"
# Tipos propios que skops puede cargar sin riesgo: el código es del repositorio.
PREFIJOS_CONFIABLES = ("proyecto_ml.", "sklearn.", "numpy.", "builtins.", "collections.")


def configurar(experimento: str = EXPERIMENTO) -> None:
    """Apunta a ``MLFLOW_TRACKING_URI`` (como train.py) y crea o elige el experimento."""
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(experimento)


def nombre_metrica(texto: str) -> str:
    """'Prec.' -> 'prec', 'TT (Sec)' -> 'tt_sec', 'roc_auc' -> 'roc_auc': nombres simples para MLflow."""
    return re.sub(r"[^0-9a-z]+", "_", texto.lower()).strip("_")


def con_prefijo(prefijo: str, metricas: dict) -> dict[str, float]:
    """``{prefijo_metrica: valor}``, solo números finitos (MLflow no acepta NaN de forma útil)."""
    salida = {}
    for clave, valor in metricas.items():
        if isinstance(valor, bool) or not isinstance(valor, int | float):
            continue
        if math.isfinite(valor):
            salida[f"{prefijo}_{nombre_metrica(str(clave))}"] = float(valor)
    return salida


def _simple(valor) -> bool:
    return valor is None or isinstance(valor, str | int | float | bool)


def parametros_modelo(pipeline) -> dict:
    """Hiperparámetros del estimador final (``model__...``), solo los de tipo simple."""
    final = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    return {f"model__{k}": v for k, v in final.get_params(deep=False).items() if _simple(v)}


def _tipos_confiables(pipeline) -> list[str] | None:
    """Tipos que skops marca como no confiables, si todos son del repo/sklearn/numpy; si no, None."""
    try:
        import skops.io as sio
    except ImportError:
        return None
    try:
        tipos = sio.get_untrusted_types(data=sio.dumps(pipeline))
    except Exception:  # skops no sabe serializar algún objeto
        return None
    return tipos if all(t.startswith(PREFIJOS_CONFIABLES) for t in tipos) else None


def registrar_modelo(pipeline, registrar_como: str | None = None):
    """Guarda el modelo en la corrida activa (y en el registro, si ``registrar_como``). Devuelve ModelInfo."""
    import mlflow
    import mlflow.sklearn

    tipos = _tipos_confiables(pipeline)
    if tipos is not None:
        try:
            return mlflow.sklearn.log_model(
                pipeline, name="model", skops_trusted_types=tipos, registered_model_name=registrar_como
            )
        except mlflow.exceptions.MlflowException:
            pass  # cae a cloudpickle
    return mlflow.sklearn.log_model(
        pipeline, name="model", serialization_format="cloudpickle", registered_model_name=registrar_como
    )


def asignar_alias(version, alias: str = ALIAS_CAMPEON, modelo: str = MODELO_REGISTRADO) -> None:
    """``models:/abandono@campeon`` pasa a apuntar a ``version``."""
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(modelo, alias, str(version))


def _registrar(params: dict | None, metricas: dict | None, artefactos: Iterable[Path] | None, tags: dict | None):
    import mlflow

    if params:
        mlflow.log_params(params)  # None queda como "None", igual que en train.py
    if metricas:
        mlflow.log_metrics(metricas)
    for ruta in artefactos or ():
        if ruta is not None and ruta.exists():
            mlflow.log_artifact(str(ruta))
    if tags:
        mlflow.set_tags(tags)


@contextmanager
def corrida(
    nombre: str,
    params: dict | None = None,
    metricas: dict | None = None,
    artefactos: Iterable[Path] | None = None,
    tags: dict | None = None,
    anidada: bool = False,
) -> Iterator[str]:
    """Abre una corrida, registra lo que se le pase y devuelve su ``run_id`` (con ``with``)."""
    import mlflow

    with mlflow.start_run(run_name=nombre, nested=anidada) as run:
        _registrar(params, metricas, artefactos, tags)
        yield run.info.run_id


def registrar_modelos(
    modelos: dict,
    params: dict,
    metricas: dict[str, dict[str, float]],
    titulos: dict[str, str] | None = None,
    guardar_modelo: bool = True,
) -> dict[str, str]:
    """Una corrida hija por modelo (dentro de la corrida activa): params comunes + hiperparámetros,
    ``metricas[nombre]`` (ya con prefijo) y el modelo. Devuelve ``{nombre: run_id}``."""
    ids = {}
    for nombre, modelo in modelos.items():
        propios = {"modelo": nombre, **params}
        estimador = getattr(modelo, "pipeline", modelo)  # AgrupadorAbandono guarda su Pipeline en .pipeline
        if hasattr(estimador, "get_params"):
            propios |= parametros_modelo(estimador)
        tags = {"titulo": (titulos or {}).get(nombre, nombre)}
        with corrida(nombre, propios, metricas.get(nombre), tags=tags, anidada=True) as rid:
            if guardar_modelo:
                registrar_modelo(modelo)
            ids[nombre] = rid
    return ids


def metricas_tabla(tabla: pd.DataFrame, prefijo: str) -> dict[str, dict[str, float]]:
    """Una tabla modelo x métrica (comparativa de CV, métricas en nuevos) -> ``{modelo: {prefijo_metrica: v}}``."""
    return {str(nombre): con_prefijo(prefijo, fila.to_dict()) for nombre, fila in tabla.iterrows()}


def unir(*diccionarios: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Une varios ``{modelo: {metrica: valor}}`` en uno solo."""
    salida: dict[str, dict[str, float]] = {}
    for d in diccionarios:
        for nombre, metricas in d.items():
            salida.setdefault(nombre, {}).update(metricas)
    return salida
