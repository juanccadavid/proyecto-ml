"""Configuración central: rutas del proyecto y variables de entorno (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)

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


def get_settings() -> Settings:
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
