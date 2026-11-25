"""Generador reproducible del extracto ficticio y de una etiqueta futura exclusivamente sintética.

Habla los mismos campos que el extracto real (``DataAnonimizada.xlsx``): los códigos del diccionario de datos
(``ESTUDIOS`` O/U/G..., ``TIPOCONTRATO`` 2/1/15/O/0, ``#SOLICITUD_DISMINUCIONCUOTAS`` SI/NO...), la misma fecha
de corte y una forma parecida: la mayoría de los conteos de productos en 0, un tercio sin estrato, edad alrededor
de 40 años y antigüedades sesgadas a la derecha. Las proporciones son aproximadas y redondeadas a propósito: dan
la forma del extracto sin copiar sus cifras. Así un modelo entrenado aquí recibe datos reales con los mismos
códigos que vio al entrenar.

La etiqueta ``EVENTO_RETIRO_3M_SINTETICO`` la inventa el generador: los "retirados" tienen más probabilidad de
pedir cancelaciones, retiros parciales y disminución de cuotas, y menos de tener ahorro contractual, créditos y
servicios. ``fuerza_senal`` escala esa diferencia (0 = ninguna).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from proyecto_ml.config import DATA_DIR, FECHA_CORTE_DEFECTO

CARPETA_EJEMPLO = DATA_DIR / "ejemplo"
FECHA_CORTE = pd.Timestamp(FECHA_CORTE_DEFECTO)  # la del extracto real
TARGET_SINTETICO = "EVENTO_RETIRO_3M_SINTETICO"


class Desarrollo(TypedDict):
    n: int
    seed: int
    tasa_retiro: float


DESARROLLO: Desarrollo = {"n": 5000, "seed": 7, "tasa_retiro": 0.1}
NOMBRE_DESARROLLO = "asociados_desarrollo"

APELLIDOS = ["PRUEBA", "FICTICIO", "EJEMPLO", "DEMO", "SINTETICO", "MUESTRA"]
NOMBRES_F = ["ANA", "LUISA", "SOFIA", "CAMILA", "PAULA"]
NOMBRES_M = ["CARLOS", "JORGE", "FELIPE", "DAVID", "MIGUEL"]
# Profesiones genéricas. En el extracto real CARGO llega igual a CODPROFESION (pregunta abierta con el fondo):
# el generador lo reproduce para que los dos orígenes tengan la misma estructura.
PROFESIONES = [
    "ADMINISTRACION DE EMPRESAS",
    "SIN PROFESION",
    "TECNICO /TECNOLOGO",
    "CONTADURIA PUBLICA",
    "INGENIERIA DE SISTEMAS",
    "INGENIERIA INDUSTRIAL",
    "ADMINISTRACION FINANCIERA",
    "ECONOMIA",
    "DERECHO",
    "PSICOLOGIA",
    "EXEMPLEADO",
    "JUBILADOS",
]
PESOS_PROFESIONES = [0.22, 0.13, 0.11, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03]

# Códigos del diccionario de datos (docs/diccionario_datos.md) con proporciones aproximadas del extracto real.
ESTUDIOS = {"O": 0.96, "U": 0.015, "G": 0.01, "E": 0.005, "T": 0.005, "S": 0.003, "P": 0.002}
ESTADOCIVIL = {"S": 0.455, "C": 0.35, "V": 0.155, "T": 0.02, "U": 0.01, "D": 0.004, "F": 0.004, "P": 0.002}
TIPOCONTRATO = {"2": 0.87, "1": 0.085, "15": 0.04, "O": 0.003, "0": 0.002}
# Un tercio sin estrato, como en el extracto (allí viene en blanco). Vacío como "", igual que el formulario y la BD.
ESTRATO = {"1": 0.45, "2": 0.01, "3": 0.17, "4": 0.005, "5": 0.003, "6": 0.002, "": 0.36}
SEXO = {"F": 0.7, "M": 0.3}
EMPRESAS = 17  # una empresa grande y varias pequeñas
PESO_EMPRESA_PRINCIPAL = 0.75
SIN_DEPENDENCIA = "9999"  # código del extracto para "sin dependencia"

# Productos y solicitudes: probabilidad de tener al menos uno (activos -> "retirados" sintéticos) y cuántos de más
# en promedio cuando se tiene (Poisson). Las marginales quedan cerca de las del extracto con un 10 % de retiro.
PRODUCTOS: dict[str, tuple[float, float, float]] = {
    "#SOLICITUDCREDITOS_ULT3MES": (0.15, 0.03, 0.4),
    "#CANTIDAD_AHORROSALAVISTA": (0.21, 0.21, 0.3),
    "#CANTIDAD_AHORROSCONTRACTUALES": (0.43, 0.10, 0.25),
    "#SOLICITUD_RETIROPARCIALAHORROS": (0.02, 0.30, 0.9),
    "#SOLICITUD_CANCELACIONAHORROS": (0.005, 0.17, 1.2),
    "#CANTIDAD_SERVICIOS": (0.35, 0.10, 0.45),
    "#RETIROS": (0.01, 0.03, 0.5),
}
DISMINUCION_CUOTAS = (0.005, 0.10)  # probabilidad de "SI"

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


def _fechas(rng: np.random.Generator, inicio: str, fin: str | pd.Timestamp, n: int) -> pd.Series:
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    dias = rng.integers(0, (b - a).days + 1, n)
    return pd.Series(a + pd.to_timedelta(dias, unit="D")).dt.normalize()


def _hace_meses(rng: np.random.Generator, meses: np.ndarray) -> pd.Series:
    """Una fecha ``meses`` meses calendario antes del corte (día al azar del 1 al 28). La antigüedad del extracto
    es esa diferencia de meses calendario: (año del corte - año) * 12 + (mes del corte - mes)."""
    total = FECHA_CORTE.year * 12 + FECHA_CORTE.month - 1 - meses.astype(int)
    partes = {"year": total // 12, "month": total % 12 + 1, "day": rng.integers(1, 29, len(meses))}
    return pd.to_datetime(pd.DataFrame(partes))


def _seudonimo(seed: int, numero: int) -> str:
    """10 caracteres hexadecimales, como el identificador anonimizado del extracto."""
    return hashlib.sha256(f"ficticio-{seed}-{numero}".encode()).hexdigest()[:10]


def _elegir(rng: np.random.Generator, opciones: dict, n: int) -> np.ndarray:
    """``n`` valores de ``opciones`` ({valor: probabilidad}); las probabilidades se normalizan."""
    p = np.array(list(opciones.values()), dtype=float)
    return np.array(list(opciones), dtype=object)[rng.choice(len(p), n, p=p / p.sum())]


def _conteo(rng: np.random.Generator, retirado: np.ndarray, f: float, base: float, senal: float, extra: float):
    """Cero la mayoría de las veces; con probabilidad ``base`` (``senal`` en los "retirados", escalada por la
    fuerza de la señal ``f``) vale 1 + Poisson(``extra``)."""
    p = np.where(retirado, np.clip(base + f * (senal - base), 0, 1), base)
    return np.where(rng.random(len(retirado)) < p, 1 + rng.poisson(extra, len(retirado)), 0).astype(int)


def _generar(n: int, seed: int, tasa_retiro: float, fuerza_senal: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n + 1)
    pick = lambda opciones, **kw: rng.choice(opciones, n, **kw)  # noqa: E731
    retirado = rng.random(n) < tasa_retiro
    f = max(float(fuerza_senal), 0.0)
    sexo = _elegir(rng, SEXO, n)
    nombres = np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M))
    segundo = np.where(rng.random(n) < 0.5, np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M)), "SIN_DATO")
    ap1, ap2 = pick(APELLIDOS), pick(APELLIDOS)
    # Edad alrededor de 40 años; la fecha de nacimiento sale de ella y EDAD se recalcula como en el extracto.
    anos = np.clip(rng.normal(39, 10, n), 19, 80)
    nacimiento = pd.Series(FECHA_CORTE - pd.to_timedelta(anos * 365.25, unit="D")).dt.normalize()
    edad = ((FECHA_CORTE - nacimiento).dt.days / 365.25).astype(int).to_numpy()
    # Antigüedades sesgadas a la derecha (muchos recientes, pocos de décadas), sin pasar de la mayoría de edad.
    tope = np.maximum((edad - 18) * 12, 0)
    meses_fondo = np.minimum(rng.exponential(80, n).round(), tope)
    meses_laboral = np.minimum(meses_fondo + rng.exponential(75, n).round(), tope)
    ingreso_fondo, ingreso_laboral = _hace_meses(rng, meses_fondo), _hace_meses(rng, meses_laboral)
    salario = rng.lognormal(np.log(3_000_000) + 0.08 * f * retirado, 0.33, n)
    salario = np.maximum((salario / 1_000).round() * 1_000, 1_423_500).astype(int)
    empresa = np.where(
        rng.random(n) < PESO_EMPRESA_PRINCIPAL, "E01", [f"E{v:02d}" for v in rng.integers(2, EMPRESAS + 1, n)]
    )
    dependencia = np.where(rng.random(n) < 0.6, SIN_DEPENDENCIA, [str(v) for v in rng.integers(1000, 1200, n)])
    profesion = rng.choice(PROFESIONES, n, p=PESOS_PROFESIONES)
    departamento = pick([11001, 22002])
    base, con_senal = DISMINUCION_CUOTAS
    disminucion = np.where(retirado, np.clip(base + f * (con_senal - base), 0, 1), base)
    conteos = {col: _conteo(rng, retirado, f, *params) for col, params in PRODUCTOS.items()}

    data = {
        "TIPOIDENTIFICACION": _elegir(rng, {"DNI": 0.99, "CI": 0.01}, n),
        "NROIDENTIFICACION": [_seudonimo(seed, int(i)) for i in idx],
        "CODPAISCEDULA": 169,
        "PRIMERAPELLIDO": ap1,
        "SEGUNDOAPELLIDO": ap2,
        "NOMBRES": nombres,
        "SEGUNDONOMBRE": segundo,
        "NOMBREINTEGRADO": [" ".join(parts) for parts in zip(ap1, ap2, nombres, segundo, strict=True)],
        "DIRECCION": "DIRECCION_ANONIMIZADA",
        "CODPAISRESIDENCIA": 169,
        "CODDEPARTAMENTORESIDENCIA": departamento,
        "CODCIUDADRESIDENCIA": departamento * 1000 + 10,
        "RELACION": "A",
        "SEXO": sexo,
        "EMAIL": [f"asociado{i}@example.invalid" for i in idx],
        "FECHANACIMIENTO": nacimiento,
        "EDAD": edad,
        "PAISNACE": 169,
        "DPTONACE": departamento,
        "CIUDADNACE": departamento * 1000 + 10,
        "FECHAULTACTUALIZACION": _fechas(rng, "2024-01-01", FECHA_CORTE, n),
        "ESTADO": "A",
        "ESTRATO": _elegir(rng, ESTRATO, n),
        "CELULAR": 30_000_000 + idx,
        "ESTUDIOS": _elegir(rng, ESTUDIOS, n),
        "CODPROFESION": profesion,
        "ESTADOCIVIL": _elegir(rng, ESTADOCIVIL, n),
        "FECHAINGRESO": ingreso_fondo,
        "ANTIGUEDAD_F_MESES": meses_fondo.astype(int),
        "CODEMPRESA": empresa,
        "CODDEPENDENCIA": dependencia,
        "SALARIO": salario,
        "FECHAINGRESOLABORA": ingreso_laboral,
        "ANTIGUEDAD_LA_MESES": meses_laboral.astype(int),
        "TIPOCONTRATO": _elegir(rng, TIPOCONTRATO, n),
        "CARGO": profesion,
        **{col: v for col, v in conteos.items() if col != "#RETIROS"},
        "#SOLICITUD_DISMINUCIONCUOTAS": np.where(rng.random(n) < disminucion, "SI", "NO"),
        "#RETIROS": conteos["#RETIROS"],
        TARGET_SINTETICO: retirado.astype(int),
    }
    columnas = [col for grupo in GRUPOS.values() for col in grupo]
    return pd.DataFrame(data)[[*columnas, TARGET_SINTETICO]]


def diccionario(incluir_objetivo_sintetico: bool = False) -> pd.DataFrame:
    filas = [(col, grupo) for grupo, cols in GRUPOS.items() for col in cols]
    if incluir_objetivo_sintetico:
        filas.append((TARGET_SINTETICO, "SOLO_SINTETICO"))
    return pd.DataFrame(filas, columns=["COLUMNA", "GRUPO"])


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
        """44 columnas del extracto más la etiqueta futura creada por el generador."""
        df = _generar(self.n, self.seed, self.tasa_retiro, self.fuerza_senal)
        return self.agregar_faltantes(df) if self.faltantes else df

    def generar_extracto(self) -> pd.DataFrame:
        """Solo las 44 columnas que existen en el extracto real; no inventa una etiqueta futura."""
        return self.generar().drop(columns=TARGET_SINTETICO)

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

    def guardar(
        self,
        carpeta: Path = CARPETA_EJEMPLO,
        nombre: str = "asociados_ejemplo",
        incluir_objetivo_sintetico: bool = False,
    ) -> tuple[Path, Path]:
        df = self.generar() if incluir_objetivo_sintetico else self.generar_extracto()
        carpeta.mkdir(parents=True, exist_ok=True)
        parquet, xlsx = carpeta / f"{nombre}.parquet", carpeta / f"{nombre}.xlsx"
        df.to_parquet(parquet, index=False)
        with pd.ExcelWriter(xlsx, engine="openpyxl", datetime_format="YYYY-MM-DD", date_format="YYYY-MM-DD") as writer:
            df.to_excel(writer, sheet_name="DATOS", index=False)
            diccionario(incluir_objetivo_sintetico).to_excel(writer, sheet_name="DICCIONARIO", index=False)
        return parquet, xlsx


def datos_desarrollo() -> pd.DataFrame:
    return GeneradorAsociados(**DESARROLLO).generar()
