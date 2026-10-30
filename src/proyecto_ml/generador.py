"""Generador de asociados FICTICIOS con el esquema real (SOCIODEMOGRAFICO, LABORAL, PRODUCTOS, RETIRO).

Nombres, cédulas, teléfonos, cuentas y correos son inventados: no corresponden a ninguna persona.
Los retirados se comportan distinto (más solicitudes de cancelación, más intención de retiro), así que
los modelos tienen algo que aprender. Cumple el contrato contratos/asociados.yaml.

Uso:
    from proyecto_ml.generador import GeneradorAsociados

    df = GeneradorAsociados(n=1000, seed=7).generar()       # asociados nuevos para validar
    GeneradorAsociados().guardar()                          # los 50 de data/ejemplo/ (seed 42)

La misma semilla produce siempre el mismo dataset; otra semilla, asociados distintos que siguen
el mismo comportamiento (datos que un modelo nunca vio).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from proyecto_ml.config import DATA_DIR

CARPETA_EJEMPLO = DATA_DIR / "ejemplo"

# Apellidos evidentemente ficticios para que nadie confunda un registro con una persona real.
APELLIDOS = ["PRUEBA", "FICTICIO", "EJEMPLO", "DEMO", "SINTETICO", "MUESTRA", "SIMULADO", "MODELO", "PILOTO", "ENSAYO"]
NOMBRES_F = ["ANA", "LUISA", "MARIA", "SOFIA", "CAMILA", "VALENTINA", "PAULA", "LAURA", "DIANA", "ISABEL"]
NOMBRES_M = ["CARLOS", "JORGE", "ANDRES", "FELIPE", "DAVID", "JUAN", "MIGUEL", "SANTIAGO", "PABLO", "DANIEL"]
# Códigos DANE de Antioquia: Medellín, Envigado, Itagüí, Bello, Sabaneta
CIUDADES = [5001, 5266, 5360, 5088, 5631]
CARGOS = [
    "ANALISTA",
    "AUXILIAR ADMINISTRATIVO",
    "MEDICO",
    "OPERARIO",
    "DESARROLLADOR",
    "DOCENTE",
    "ENFERMERO",
    "CONTADOR",
]
CIIU = {8411: 84, 8512: 85, 8610: 86, 6201: 62, 4711: 47}
MOTIVOS = [
    "INSATISFACCION CON TASAS",
    "TERMINACION CONTRATO LABORAL",
    "NECESIDAD DE LIQUIDEZ",
    "TRASLADO A OTRA ENTIDAD",
    "PENSION",
]

# Columnas por grupo, en el orden del esquema. AGENCIA y ESTADO se repiten en LABORAL y llevan sufijo.
GRUPOS = {
    "SOCIODEMOGRAFICO": [
        "CEDULA",
        "PRIMERAPELLIDO",
        "SEGUNDOAPELLIDO",
        "NOMBRES",
        "SEGUNDONOMBRE",
        "NOMBREINTEGRADO",
        "DIRECCION",
        "TIPOIDENTIFICACION",
        "RELACION",
        "AGENCIA",
        "TELEFONO1",
        "EXTENSION1",
        "TELEFONO2",
        "EXTENSION2",
        "APARTADOAEREO",
        "SEXO",
        "CODPAISCEDULA",
        "CODCIUDADCEDULA",
        "EMAIL",
        "FECHARETIRO",
        "FECHAULTACTUALIZACION",
        "FECHANACIMIENTO",
        "CODPAIS",
        "CODDEPARTAMENTO",
        "CODCIUDAD",
        "CODZONA",
        "CODCOMUNA",
        "CODBARRIO",
        "CODCIIU",
        "CODDIVISION",
        "ESTADO",
        "ESTRATO",
    ],
    "LABORAL": [
        "CODEMPRESA",
        "CODDEPENDENCIA",
        "PERIODODEDUCE",
        "DEDUCEOCACIONAL",
        "SALARIO",
        "CODEMPRESALABORA",
        "FECHAINGRESOLABORA",
        "TIPOCONTRATO",
        "FECHATERMINACONTRATO",
        "CARGO",
        "CODBANCO",
        "NUMEROCUENTA",
        "AGENCIA_LABORAL",
        "CODANALISTA",
        "RUTAANALISTA",
        "TIPOCUENTA",
        "ESTADO_LABORAL",
        "ULTIMAACTUALIZACION",
        "OPERADORACTUALIZO",
        "SZIN",
        "IDTIPOSALARIO",
        "TIPOPACTO",
        "IDASOCIADOS",
        "CODIGOTIPO",
        "CODPAISOFI",
        "CODDEPARTAMENTOOFI",
        "CODCIUDADOFI",
        "DIRECCIONEMPRESA",
        "TELOFICINA",
        "EXTENSION",
        "MODALIDADSALARIAL",
    ],
    "PRODUCTOS": [
        "#SOLICITUD_DISMINUCIONCUOTAS",
        "#SOLICITUD_CANCELACIONAHORROS",
        "#SOLICITUD_RETIROPARCIALAHORROS",
        "#CANTIDAD_AHORROSALAVISTA",
        "#CANTIDAD_AHORROSCONTRACTUALES",
        "CANCELACIONCREDITOS_ULT3MESES",
        "#SOLICITUDCREDITOS_ULT3MESES",
        "#CANTIDAD_SERVICIOS",
    ],
    "RETIRO": ["#INTENCIONRETIROS_ULT3MESES", "#RETIROS", "MOTIVORETIRO"],
}


def _fechas(rng: np.random.Generator, inicio: str, fin: str, n: int) -> pd.Series:
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    dias = rng.integers(0, (b - a).days + 1, n)
    return pd.Series(a + pd.to_timedelta(dias, unit="D")).dt.normalize()


def _generar(n: int, seed: int, tasa_retiro: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n + 1)
    pick = lambda opciones, **kw: rng.choice(opciones, n, **kw)  # noqa: E731

    # Retiro: la propensión sube con solicitudes de cancelación/disminución e intención declarada.
    retirado = rng.random(n) < tasa_retiro
    contrato_fijo = rng.random(n) < np.where(retirado, 0.6, 0.25)
    sexo = pick(["F", "M"])
    nombres = np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M))
    segundo = np.where(rng.random(n) < 0.5, np.where(sexo == "F", pick(NOMBRES_F), pick(NOMBRES_M)), "")
    ap1, ap2 = pick(APELLIDOS), pick(APELLIDOS)
    ciudad = pick(CIUDADES)
    en_medellin = ciudad == 5001
    ciiu = pick(list(CIIU))
    agencia = rng.integers(1, 6, n)
    empresa = rng.integers(101, 111, n)

    fecha_retiro = _fechas(rng, "2025-11-01", "2026-10-15", n).where(retirado)
    ingreso = _fechas(rng, "1995-01-01", "2024-12-31", n)
    fin_contrato = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    fin_contrato[contrato_fijo] = (ingreso + pd.DateOffset(years=1))[contrato_fijo]
    # Para retirados con contrato fijo, el contrato termina el mismo día del retiro.
    termina_con_retiro = contrato_fijo & retirado
    fin_contrato[termina_con_retiro] = fecha_retiro[termina_con_retiro]
    fin_contrato[contrato_fijo & ~retirado] = _fechas(rng, "2026-11-01", "2027-12-31", n)[contrato_fijo & ~retirado]
    ult_act = _fechas(rng, "2025-06-01", "2026-10-15", n)
    ult_act = ult_act.where(~retirado, fecha_retiro)

    motivo = np.where(retirado, pick(MOTIVOS), "")
    motivo = np.where(termina_con_retiro, "TERMINACION CONTRATO LABORAL", motivo)

    salario = (np.round(rng.lognormal(15.0, 0.45, n) / 10_000) * 10_000).astype(int)
    salario = np.maximum(salario, 1_423_500)  # no por debajo del mínimo 2025
    integral = salario >= 10_000_000

    data = {
        # ---- SOCIODEMOGRAFICO ----
        "CEDULA": 900_000_000 + idx,
        "PRIMERAPELLIDO": ap1,
        "SEGUNDOAPELLIDO": ap2,
        "NOMBRES": nombres,
        "SEGUNDONOMBRE": segundo,
        "NOMBREINTEGRADO": [" ".join(p for p in parts if p) for parts in zip(ap1, ap2, nombres, segundo, strict=True)],
        "DIRECCION": [
            f"{t} {rng.integers(1, 100)} # {rng.integers(1, 100)}-{rng.integers(1, 99):02d}"
            for t in pick(["CL", "CR", "TV", "DG", "AV"])
        ],
        "TIPOIDENTIFICACION": np.where(rng.random(n) < 0.95, "CC", "CE"),
        "RELACION": "ASOCIADO",
        "AGENCIA": agencia,
        "TELEFONO1": [f"300{i:07d}" for i in idx],
        "EXTENSION1": "",
        "TELEFONO2": np.where(rng.random(n) < 0.5, [f"604555{i:04d}" for i in idx], ""),
        "EXTENSION2": "",
        "APARTADOAEREO": "",
        "SEXO": sexo,
        "CODPAISCEDULA": 169,
        "CODCIUDADCEDULA": ciudad,
        "EMAIL": [f"asociado{i:03d}@example.com" for i in idx],
        "FECHARETIRO": fecha_retiro,
        "FECHAULTACTUALIZACION": ult_act,
        "FECHANACIMIENTO": _fechas(rng, "1962-01-01", "2004-12-31", n),
        "CODPAIS": 169,
        "CODDEPARTAMENTO": 5,
        "CODCIUDAD": ciudad,
        "CODZONA": rng.integers(1, 4, n),
        "CODCOMUNA": np.where(en_medellin, rng.integers(1, 17, n), 0),
        "CODBARRIO": 0,  # se completa abajo a partir de la comuna
        "CODCIIU": ciiu,
        "CODDIVISION": [CIIU[c] for c in ciiu],
        "ESTADO": np.where(retirado, "RETIRADO", "ACTIVO"),
        "ESTRATO": rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.08, 0.22, 0.32, 0.2, 0.12, 0.06]),
        # ---- LABORAL ----
        "CODEMPRESA": empresa,
        "CODDEPENDENCIA": rng.integers(10, 50, n),
        "PERIODODEDUCE": pick(["MENSUAL", "QUINCENAL"]),
        "DEDUCEOCACIONAL": np.where(rng.random(n) < 0.2, "S", "N"),
        "SALARIO": salario,
        "CODEMPRESALABORA": empresa,
        "FECHAINGRESOLABORA": ingreso,
        "TIPOCONTRATO": np.where(contrato_fijo, "FIJO", "INDEFINIDO"),
        "FECHATERMINACONTRATO": fin_contrato,
        "CARGO": pick(CARGOS),
        "CODBANCO": pick([1, 7, 13, 51, 23]),
        "NUMEROCUENTA": [f"{i:010d}" for i in idx],
        "AGENCIA_LABORAL": agencia,
        "CODANALISTA": rng.integers(501, 506, n),
        "RUTAANALISTA": [f"R{a}" for a in agencia],
        "TIPOCUENTA": np.where(rng.random(n) < 0.8, "AHORROS", "CORRIENTE"),
        "ESTADO_LABORAL": np.where(retirado, "RETIRADO", "VIGENTE"),
        "ULTIMAACTUALIZACION": ult_act,
        "OPERADORACTUALIZO": [f"OPER{o:02d}" for o in rng.integers(1, 6, n)],
        "SZIN": rng.integers(1, 3, n),
        "IDTIPOSALARIO": np.where(integral, 2, 1),
        "TIPOPACTO": np.where(integral, "INTEGRAL", "CONVENCIONAL"),
        "IDASOCIADOS": 10_000 + idx,
        "CODIGOTIPO": np.where(contrato_fijo, 2, 1),
        "CODPAISOFI": 169,
        "CODDEPARTAMENTOOFI": 5,
        "CODCIUDADOFI": np.where(rng.random(n) < 0.7, 5001, ciudad),
        "DIRECCIONEMPRESA": [f"CL {e - 60} # {e - 50}-{e % 100:02d}" for e in empresa],
        "TELOFICINA": [f"6045550{e:03d}" for e in empresa],
        "EXTENSION": [str(x) if x else "" for x in np.where(rng.random(n) < 0.8, rng.integers(100, 500, n), 0)],
        "MODALIDADSALARIAL": np.where(integral, "INTEGRAL", "FIJO"),
        # ---- PRODUCTOS ----
        "#SOLICITUD_DISMINUCIONCUOTAS": rng.poisson(np.where(retirado, 1.5, 0.3)),
        "#SOLICITUD_CANCELACIONAHORROS": rng.poisson(np.where(retirado, 1.2, 0.1)),
        "#SOLICITUD_RETIROPARCIALAHORROS": rng.poisson(np.where(retirado, 2.0, 0.6)),
        "#CANTIDAD_AHORROSALAVISTA": rng.integers(1, 4, n),
        "#CANTIDAD_AHORROSCONTRACTUALES": rng.poisson(np.where(retirado, 0.3, 1.0)),
        "CANCELACIONCREDITOS_ULT3MESES": rng.poisson(np.where(retirado, 0.8, 0.1)),
        "#SOLICITUDCREDITOS_ULT3MESES": rng.poisson(np.where(retirado, 0.2, 0.9)),
        "#CANTIDAD_SERVICIOS": np.maximum(rng.integers(1, 7, n) - retirado, 1),
        # ---- RETIRO ----
        "#INTENCIONRETIROS_ULT3MESES": rng.poisson(np.where(retirado, 1.5, 0.2)),
        "#RETIROS": retirado.astype(int),
        "MOTIVORETIRO": motivo,
    }
    df = pd.DataFrame(data)
    df["CODBARRIO"] = np.where(en_medellin, df["CODCOMUNA"] * 100 + rng.integers(1, 20, n), 0)
    # Contrato (mayor_de_edad_al_ingreso): nadie ingresa a laborar antes de los 18 años.
    mayoria = df["FECHANACIMIENTO"] + pd.DateOffset(years=18)
    df["FECHAINGRESOLABORA"] = df["FECHAINGRESOLABORA"].where(df["FECHAINGRESOLABORA"] >= mayoria, mayoria)
    columnas = [c for cols in GRUPOS.values() for c in cols]
    return df[columnas]


def diccionario() -> pd.DataFrame:
    """Columna -> grupo (SOCIODEMOGRAFICO, LABORAL, PRODUCTOS, RETIRO)."""
    return pd.DataFrame([(c, g) for g, cols in GRUPOS.items() for c in cols], columns=["COLUMNA", "GRUPO"])


class GeneradorAsociados:
    """Asociados ficticios reproducibles: misma semilla, mismo dataset."""

    def __init__(self, n: int = 50, seed: int = 42, tasa_retiro: float = 0.3):
        self.n = n
        self.seed = seed
        self.tasa_retiro = tasa_retiro

    def generar(self) -> pd.DataFrame:
        return _generar(self.n, self.seed, self.tasa_retiro)

    def guardar(self, carpeta: Path = CARPETA_EJEMPLO, nombre: str = "asociados_ejemplo") -> tuple[Path, Path]:
        """Escribe <nombre>.parquet y <nombre>.xlsx (hojas DATOS y DICCIONARIO)."""
        df = self.generar()
        carpeta.mkdir(parents=True, exist_ok=True)
        parquet, xlsx = carpeta / f"{nombre}.parquet", carpeta / f"{nombre}.xlsx"
        df.to_parquet(parquet, index=False)
        with pd.ExcelWriter(xlsx, engine="openpyxl", datetime_format="YYYY-MM-DD", date_format="YYYY-MM-DD") as xw:
            df.to_excel(xw, sheet_name="DATOS", index=False)
            diccionario().to_excel(xw, sheet_name="DICCIONARIO", index=False)
        return parquet, xlsx
