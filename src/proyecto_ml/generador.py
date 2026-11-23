"""Generador reproducible de asociados ficticios con el esquema anonimizado definitivo de 44 columnas."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from proyecto_ml.config import DATA_DIR

CARPETA_EJEMPLO = DATA_DIR / "ejemplo"
FECHA_CORTE = pd.Timestamp("2026-10-31")


class Desarrollo(TypedDict):
    n: int
    seed: int
    tasa_retiro: float


DESARROLLO: Desarrollo = {"n": 5000, "seed": 7, "tasa_retiro": 0.1}
NOMBRE_DESARROLLO = "asociados_desarrollo"

APELLIDOS = ["PRUEBA", "FICTICIO", "EJEMPLO", "DEMO", "SINTETICO", "MUESTRA"]
NOMBRES_F = ["ANA", "LUISA", "SOFIA", "CAMILA", "PAULA"]
NOMBRES_M = ["CARLOS", "JORGE", "FELIPE", "DAVID", "MIGUEL"]
CARGOS = ["ANALISTA", "AUXILIAR", "OPERARIO", "DESARROLLADOR", "DOCENTE", "CONTADOR"]
ESTUDIOS = ["BACHILLER", "TECNICO", "TECNOLOGO", "PROFESIONAL", "POSGRADO"]

GRUPOS = {
    "SOCIODEMOGRAFICO": [
        "TIPOIDENTIFICACION",
        "NROIDENTIFICACION",
        "CODPAISCEDULA",
        "PRIMERAPELLIDO",
        "SEGUNDOAPELLIDO",
        "NOMBRES",
        "SEGUNDONOMBRE",
        "NOMBREINTEGRADO",
        "DIRECCION",
        "CODPAISRESIDENCIA",
        "CODDEPARTAMENTORESIDENCIA",
        "CODCIUDADRESIDENCIA",
        "RELACION",
        "SEXO",
        "EMAIL",
        "FECHANACIMIENTO",
        "EDAD",
        "PAISNACE",
        "DPTONACE",
        "CIUDADNACE",
        "FECHAULTACTUALIZACION",
        "ESTADO",
        "ESTRATO",
        "CELULAR",
        "ESTUDIOS",
        "CODPROFESION",
        "ESTADOCIVIL",
        "FECHAINGRESO",
        "ANTIGUEDAD_F_MESES",
    ],
    "LABORAL": [
        "CODEMPRESA",
        "CODDEPENDENCIA",
        "SALARIO",
        "FECHAINGRESOLABORA",
        "ANTIGUEDAD_LA_MESES",
        "TIPOCONTRATO",
        "CARGO",
    ],
    "PRODUCTOS": [
        "#SOLICITUDCREDITOS_ULT3MES",
        "#CANTIDAD_AHORROSALAVISTA",
        "#CANTIDAD_AHORROSCONTRACTUALES",
        "#SOLICITUD_RETIROPARCIALAHORROS",
        "#SOLICITUD_CANCELACIONAHORROS",
        "#SOLICITUD_DISMINUCIONCUOTAS",
        "#CANTIDAD_SERVICIOS",
    ],
    "RETIRO": ["#RETIROS"],
}


def _fechas(rng: np.random.Generator, inicio: str, fin: str, n: int) -> pd.Series:
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    dias = rng.integers(0, (b - a).days + 1, n)
    return pd.Series(a + pd.to_timedelta(dias, unit="D")).dt.normalize()


def _meses(desde: pd.Series) -> pd.Series:
    return ((FECHA_CORTE - desde).dt.days / 30.4375).clip(lower=0).round().astype(int)


def _seudonimo(seed: int, numero: int) -> str:
    return hashlib.sha256(f"ficticio-{seed}-{numero}".encode()).hexdigest()


def _generar(n: int, seed: int, tasa_retiro: float, fuerza_senal: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n + 1)
    pick = lambda opciones, **kw: rng.choice(opciones, n, **kw)  # noqa: E731
    retirado = rng.random(n) < tasa_retiro
    sexo = pick(["F", "M"])
    nombres = np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M))
    segundo = np.where(rng.random(n) < 0.5, np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M)), "SIN_DATO")
    ap1, ap2 = pick(APELLIDOS), pick(APELLIDOS)
    nacimiento = _fechas(rng, "1962-01-01", "2004-10-31", n)
    edad = ((FECHA_CORTE - nacimiento).dt.days / 365.25).astype(int)
    ingreso_fondo = _fechas(rng, "1995-01-01", "2025-12-31", n)
    ingreso_laboral = _fechas(rng, "1995-01-01", "2025-12-31", n)
    mayoria = nacimiento + pd.DateOffset(years=18)
    ingreso_fondo = ingreso_fondo.where(ingreso_fondo > mayoria, mayoria + pd.Timedelta(days=1))
    ingreso_laboral = ingreso_laboral.where(ingreso_laboral > mayoria, mayoria + pd.Timedelta(days=1))
    f = max(float(fuerza_senal), 0.0)
    senal = np.where(retirado, f, 0.0)
    salario = np.maximum((rng.lognormal(15.0 + 0.08 * senal, 0.4, n) / 10_000).round() * 10_000, 1_423_500).astype(int)
    solicitudes_cancelacion = rng.poisson(0.1 + 1.1 * senal)
    solicitudes_parciales = rng.poisson(0.5 + 1.3 * senal)
    solicitudes_credito = rng.poisson(np.maximum(0.8 - 0.5 * senal, 0.1))
    servicios = np.maximum(rng.integers(1, 7, n) - retirado.astype(int), 1)
    ciudad = pick([5001, 5266, 5360, 5088, 5631])

    data = {
        "TIPOIDENTIFICACION": pick(["CC", "CE"]),
        "NROIDENTIFICACION": [_seudonimo(seed, int(i)) for i in idx],
        "CODPAISCEDULA": 169,
        "PRIMERAPELLIDO": ap1,
        "SEGUNDOAPELLIDO": ap2,
        "NOMBRES": nombres,
        "SEGUNDONOMBRE": segundo,
        "NOMBREINTEGRADO": [" ".join(parts) for parts in zip(ap1, ap2, nombres, segundo, strict=True)],
        "DIRECCION": "DIRECCION_ANONIMIZADA",
        "CODPAISRESIDENCIA": 169,
        "CODDEPARTAMENTORESIDENCIA": 5,
        "CODCIUDADRESIDENCIA": ciudad,
        "RELACION": "ASOCIADO",
        "SEXO": sexo,
        "EMAIL": "correo_anonimizado@example.invalid",
        "FECHANACIMIENTO": nacimiento,
        "EDAD": edad,
        "PAISNACE": 169,
        "DPTONACE": 5,
        "CIUDADNACE": ciudad,
        "FECHAULTACTUALIZACION": _fechas(rng, "2025-01-01", "2026-10-31", n),
        "ESTADO": "ACTIVO",
        "ESTRATO": pick(["1", "2", "3", "4", "5", "6"]),
        "CELULAR": 300_000_0000 + idx,
        "ESTUDIOS": pick(ESTUDIOS),
        "CODPROFESION": [f"P{v:03d}" for v in rng.integers(1, 80, n)],
        "ESTADOCIVIL": pick(["SOLTERO", "CASADO", "UNION_LIBRE", "OTRO"]),
        "FECHAINGRESO": ingreso_fondo,
        "ANTIGUEDAD_F_MESES": _meses(ingreso_fondo),
        "CODEMPRESA": [f"E{v:03d}" for v in rng.integers(1, 30, n)],
        "CODDEPENDENCIA": [f"D{v:03d}" for v in rng.integers(1, 60, n)],
        "SALARIO": salario,
        "FECHAINGRESOLABORA": ingreso_laboral,
        "ANTIGUEDAD_LA_MESES": _meses(ingreso_laboral),
        "TIPOCONTRATO": pick(["FIJO", "INDEFINIDO", "OTRO"]),
        "CARGO": pick(CARGOS),
        "#SOLICITUDCREDITOS_ULT3MES": solicitudes_credito,
        "#CANTIDAD_AHORROSALAVISTA": rng.integers(1, 4, n),
        "#CANTIDAD_AHORROSCONTRACTUALES": rng.poisson(np.maximum(1.0 - 0.5 * senal, 0.1)),
        "#SOLICITUD_RETIROPARCIALAHORROS": solicitudes_parciales,
        "#SOLICITUD_CANCELACIONAHORROS": solicitudes_cancelacion,
        "#SOLICITUD_DISMINUCIONCUOTAS": (rng.poisson(0.2 + 1.0 * senal)).astype(str),
        "#CANTIDAD_SERVICIOS": servicios,
        "#RETIROS": retirado.astype(int),
    }
    columnas = [col for grupo in GRUPOS.values() for col in grupo]
    return pd.DataFrame(data)[columnas]


def diccionario() -> pd.DataFrame:
    return pd.DataFrame([(col, grupo) for grupo, cols in GRUPOS.items() for col in cols], columns=["COLUMNA", "GRUPO"])


FALTANTES = {
    "SALARIO": 0.05,
    "ESTRATO": 0.03,
    "CARGO": 0.02,
    "EMAIL": 0.03,
    "ANTIGUEDAD_LA_MESES": 0.05,
    "#CANTIDAD_SERVICIOS": 0.05,
}


class GeneradorAsociados:
    def __init__(
        self,
        n: int = 50,
        seed: int = 42,
        tasa_retiro: float = 0.3,
        faltantes: dict[str, float] | None = None,
        fuerza_senal: float = 1.0,
    ):
        if fuerza_senal < 0:
            raise ValueError("fuerza_senal debe ser >= 0")
        self.n = n
        self.seed = seed
        self.tasa_retiro = tasa_retiro
        self.faltantes = faltantes
        self.fuerza_senal = fuerza_senal
        self.reporte_faltantes_: dict[str, int] = {}

    def generar(self) -> pd.DataFrame:
        df = _generar(self.n, self.seed, self.tasa_retiro, self.fuerza_senal)
        return self.agregar_faltantes(df) if self.faltantes else df

    def agregar_faltantes(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1)
        df = df.copy()
        self.reporte_faltantes_ = {}
        for columna, probabilidad in (self.faltantes or {}).items():
            if columna not in df:
                raise KeyError(f"La columna {columna!r} no existe en el esquema de asociados")
            vacias = rng.random(len(df)) < probabilidad
            if pd.api.types.is_integer_dtype(df[columna]):
                df[columna] = df[columna].astype("Int64")
            df.loc[vacias, columna] = None
            self.reporte_faltantes_[columna] = int(vacias.sum())
        return df

    def guardar(self, carpeta: Path = CARPETA_EJEMPLO, nombre: str = "asociados_ejemplo") -> tuple[Path, Path]:
        df = self.generar()
        carpeta.mkdir(parents=True, exist_ok=True)
        parquet, xlsx = carpeta / f"{nombre}.parquet", carpeta / f"{nombre}.xlsx"
        df.to_parquet(parquet, index=False)
        with pd.ExcelWriter(xlsx, engine="openpyxl", datetime_format="YYYY-MM-DD", date_format="YYYY-MM-DD") as writer:
            df.to_excel(writer, sheet_name="DATOS", index=False)
            diccionario().to_excel(writer, sheet_name="DICCIONARIO", index=False)
        return parquet, xlsx


def datos_desarrollo() -> pd.DataFrame:
    return GeneradorAsociados(**DESARROLLO).generar()
