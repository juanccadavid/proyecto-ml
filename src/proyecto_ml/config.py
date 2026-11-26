"""Configuración central: rutas del proyecto y variables de entorno (.env)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
_DOTENV_CARGADO: dict[str, str] = {}
_DOTENV_FIRMA: str | None = None


def _recargar_dotenv() -> None:
    """Sincroniza ``ENV_PATH`` con ``os.environ`` si su contenido cambió, sin pisar variables del entorno externo."""
    global _DOTENV_FIRMA
    firma = hashlib.sha256(ENV_PATH.read_bytes()).hexdigest() if ENV_PATH.exists() else ""
    if firma == _DOTENV_FIRMA:
        return
    nuevos = {k: v for k, v in dotenv_values(ENV_PATH).items() if v is not None} if firma else {}
    for clave, anterior in list(_DOTENV_CARGADO.items()):
        if os.environ.get(clave) == anterior:  # nadie la sobrescribió desde la terminal ni con monkeypatch
            if clave in nuevos:
                os.environ[clave] = nuevos[clave]
                _DOTENV_CARGADO[clave] = nuevos[clave]
            else:
                os.environ.pop(clave, None)
                del _DOTENV_CARGADO[clave]
    for clave, valor in nuevos.items():
        if clave not in os.environ:
            os.environ[clave] = valor
            _DOTENV_CARGADO[clave] = valor
    _DOTENV_FIRMA = firma


_recargar_dotenv()

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"


def _resolve(path_str: str) -> Path:
    """Convierte rutas relativas del .env en rutas absolutas respecto a la raíz."""
    path = Path(path_str)
    return path if path.is_absolute() else ROOT_DIR / path


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    data_path: Path
    target_column: str
    id_column: str | None
    time_column: str | None
    group_column: str | None
    random_state: int
    test_size: float
    mlflow_tracking_uri: str
    mlflow_experiment: str
    datos_asociados: Path


# Origen de los datos del abandono (ORIGEN_DATOS en el .env). Los resultados de cada origen se etiquetan y se
# registran por separado (experimentos y modelo de MLflow distintos): no se comparan como si fueran lo mismo.
ORIGEN_SINTETICO = "sintetico"  # GeneradorAsociados o archivos del equipo: resultados EXPERIMENTALES
ORIGEN_REAL = "real_anonimizado"  # el extracto del fondo, ya anonimizado por quien lo custodia
ORIGENES = (ORIGEN_SINTETICO, ORIGEN_REAL)


def origen_datos(origen: str | None = None) -> str:
    """El origen de los datos: ``origen``, o ``ORIGEN_DATOS`` del .env, o "sintetico" si no se dice nada."""
    _recargar_dotenv()
    valor = (origen or os.getenv("ORIGEN_DATOS", "") or ORIGEN_SINTETICO).strip()
    if valor not in ORIGENES:
        raise ValueError(f"ORIGEN_DATOS debe ser uno de {ORIGENES}, no {valor!r}")
    return valor


def es_experimental(origen: str | None = None) -> bool:
    """True con datos sintéticos: sus métricas no dicen cómo le irá al modelo con el fondo real."""
    return origen_datos(origen) == ORIGEN_SINTETICO


FECHA_CORTE_DEFECTO = "2026-03-31"  # la del extracto real (DataAnonimizada.xlsx); los sintéticos usan la misma


def fecha_corte() -> str:
    """Fecha del corte C (``FECHA_CORTE`` del .env): las variables se miden en t0 = C y el evento cae en (C, C + 3m].
    Con el extracto real debe coincidir con su fecha de corte."""
    _recargar_dotenv()
    return os.getenv("FECHA_CORTE", "").strip() or FECHA_CORTE_DEFECTO


def temporalidad_confirmada() -> bool:
    """``TEMPORALIDAD_CONFIRMADA=si`` en el .env: el fondo confirmó que las variables y conteos del extracto
    (ULT3MES y demás de comportamiento) están medidos al corte t0 = C y no incluyen hechos posteriores al evento."""
    _recargar_dotenv()
    return os.getenv("TEMPORALIDAD_CONFIRMADA", "").strip().lower() in ("si", "sí", "true", "1")


def aviso_origen(origen: str | None = None) -> str:
    """Una línea para la consola, los reportes y el simulador: en qué modo se está trabajando."""
    if es_experimental(origen):
        return "MODO SINTÉTICO: resultados EXPERIMENTALES; no dicen cómo le irá al modelo con el fondo real."
    return "DATOS REALES ANONIMIZADOS: resultados del fondo (se registran aparte de los sintéticos)."


def get_settings() -> Settings:
    _recargar_dotenv()
    default_mlflow = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
    return Settings(
        data_path=_resolve(os.getenv("DATA_PATH", "data/processed/dataset.parquet")),
        target_column=os.getenv("TARGET_COLUMN", "target"),
        id_column=_optional("ID_COLUMN"),
        time_column=_optional("TIME_COLUMN"),
        group_column=_optional("GROUP_COLUMN"),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", default_mlflow),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT_NAME", "proyecto-1"),
        # Interruptor del .env: datos de ejemplo (por defecto) o la última copia bajada de Supabase.
        datos_asociados=_resolve(os.getenv("DATOS_ASOCIADOS", "data/ejemplo/asociados_ejemplo.parquet")),
    )
