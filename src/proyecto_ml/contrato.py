"""Contrato de datos: carga el YAML de contratos/ y lo aplica.

Del mismo contrato salen la validación de DataFrames en tres niveles (pandera), el SQL de
Supabase (tabla + CHECK + RLS), los registros listos para la API y la configuración del formulario web.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.errors import SchemaErrors

from proyecto_ml.config import ROOT_DIR

CONTRATO_ASOCIADOS = ROOT_DIR / "contratos" / "asociados.yaml"
TIPOS_SQL = {"entero": "bigint", "texto": "text", "fecha": "date"}


@dataclass(frozen=True)
class Columna:
    nombre: str
    grupo: str
    tipo: str
    requerido: bool = False
    unico: bool = False
    min: Any = None
    max: Any = None
    valores: list | None = None
    patron: str | None = None
    descripcion: str | None = None

    @property
    def db(self) -> str:
        """Nombre en la BD: minúsculas y '#' -> 'num_' (#RETIROS -> num_retiros)."""
        return ("num_" + self.nombre[1:] if self.nombre.startswith("#") else self.nombre).lower()


@dataclass(frozen=True)
class Contrato:
    nombre: str
    version: str
    tabla_db: str
    objetivo: str
    columnas: list[Columna]
    reglas: list[dict] = field(default_factory=list)
    distribucion: list[dict] = field(default_factory=list)

    @property
    def nombres(self) -> list[str]:
        return [c.nombre for c in self.columnas]


def cargar(path: str | Path = CONTRATO_ASOCIADOS) -> Contrato:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    columnas = [Columna(grupo=grupo, **col) for grupo, cols in raw["grupos"].items() for col in cols]
    if invalidos := {c.tipo for c in columnas} - TIPOS_SQL.keys():
        raise ValueError(f"Tipos no soportados en el contrato: {invalidos}")
    meta = (raw["nombre"], str(raw["version"]), raw["tabla_db"], raw["objetivo"])
    return Contrato(*meta, columnas, raw.get("reglas", []), raw.get("distribucion", []))


# ---------------------------------------------------------------------------- validación (pandera)
#
# Nivel 1 (por fila): pa.Column con Check.ge/le/isin/str_matches, generado desde 'grupos' del YAML.
#   Es lo mismo que escribir pa.Field(ge=..., le=...) en un DataFrameModel, sin repetir 74 columnas a mano.
# Nivel 2 (distribución) y nivel 3 (entre columnas): checks sobre el DataFrame completo, el equivalente
#   de @pa.dataframe_check. La lógica vive en DISTRIBUCION y REGLAS; los parámetros, en el YAML.


def _vacio(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.strip() == "")


def _fecha(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.where(~_vacio(s)), errors="coerce")


def _texto(*cols: pd.Series) -> pd.Series:
    """Une columnas de texto con espacio saltando vacíos (como concat_ws en SQL)."""
    partes = pd.concat(cols, axis=1).astype("string")
    return partes.apply(lambda fila: " ".join(x.strip() for x in fila.dropna() if x.strip()), axis=1)


# Nivel 2: (medida del lote, condición sobre la medida). Los parámetros p vienen del YAML.
DISTRIBUCION: dict[str, tuple[Callable[[pd.DataFrame, dict], float], Callable[[float, dict], bool]]] = {
    "volumen_minimo": (lambda d, p: len(d), lambda v, p: v >= p["min_filas"]),
    "tasa_retiro_plausible": (lambda d, p: d["#RETIROS"].mean(), lambda v, p: p["min"] <= v <= p["max"]),
    "salario_en_pesos": (lambda d, p: d["SALARIO"].median(), lambda v, p: p["min"] <= v <= p["max"]),
    "salarios_atipicos_marginales": (
        lambda d, p: (d["SALARIO"] > p["factor"] * d["SALARIO"].median()).mean(),
        lambda v, p: v <= p["max_fraccion"],
    ),
    "edad_mediana_plausible": (
        lambda d, p: ((pd.Timestamp(p["fecha_corte"]) - _fecha(d["FECHANACIMIENTO"])).dt.days / 365.25).median(),
        lambda v, p: p["min"] <= v <= p["max"],
    ),
}

# Nivel 3: versión pandas de las reglas del YAML (mismo id). True en las filas que cumplen.
REGLAS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "retiro_posterior_ingreso": lambda d: (
        _vacio(d["FECHARETIRO"]) | (_fecha(d["FECHARETIRO"]) > _fecha(d["FECHAINGRESOLABORA"]))
    ),
    "nombreintegrado_coherente": lambda d: (
        d["NOMBREINTEGRADO"].astype("string").str.strip()
        == _texto(d["PRIMERAPELLIDO"], d["SEGUNDOAPELLIDO"], d["NOMBRES"], d["SEGUNDONOMBRE"])
    ),
    "retiro_coherente": lambda d: (
        ((d["#RETIROS"] == 1) == ~_vacio(d["FECHARETIRO"]))
        & ((d["#RETIROS"] == 1) == (d["ESTADO"] == "RETIRADO"))
        & ((d["#RETIROS"] == 1) == ~_vacio(d["MOTIVORETIRO"]))
    ),
    "estado_laboral_coherente": lambda d: (d["ESTADO"] == "RETIRADO") == (d["ESTADO_LABORAL"] == "RETIRADO"),
    "contrato_fijo_con_fecha_fin": lambda d: (d["TIPOCONTRATO"] == "FIJO") == ~_vacio(d["FECHATERMINACONTRATO"]),
    "fin_contrato_posterior_ingreso": lambda d: (
        _vacio(d["FECHATERMINACONTRATO"]) | (_fecha(d["FECHATERMINACONTRATO"]) > _fecha(d["FECHAINGRESOLABORA"]))
    ),
    "mayor_de_edad_al_ingreso": lambda d: (
        _fecha(d["FECHAINGRESOLABORA"]) >= _fecha(d["FECHANACIMIENTO"]) + pd.DateOffset(years=18)
    ),
    "pacto_integral_coherente": lambda d: (
        ((d["TIPOPACTO"] == "INTEGRAL") == (d["MODALIDADSALARIAL"] == "INTEGRAL"))
        & ((d["TIPOPACTO"] == "INTEGRAL") == (d["IDTIPOSALARIO"] == 2))
    ),
}


DTYPES_PA = {"entero": "Int64", "fecha": "datetime64[ns]", "texto": None}


def _columna_pa(c: Columna) -> pa.Column:
    """Nivel 1: una columna del contrato como pa.Column (tipo con coerce, nulos, único y checks)."""
    limite = pd.Timestamp if c.tipo == "fecha" else (lambda x: x)
    checks = []
    if c.valores is not None:
        checks.append(pa.Check.isin(c.valores))
    if c.min is not None:
        checks.append(pa.Check.ge(limite(c.min)))
    if c.max is not None:
        checks.append(pa.Check.le(limite(c.max)))
    if c.patron is not None:
        checks.append(pa.Check.str_matches(c.patron))
    return pa.Column(DTYPES_PA[c.tipo], checks, nullable=not c.requerido, unique=c.unico, coerce=c.tipo != "texto")


def _check_lote(check: Callable[[pd.DataFrame], Any]) -> Callable[[pd.DataFrame], Any]:
    """Si falta una columna que usa el check o vino corrupta, ya lo reporta el nivel 1: no se duplica."""

    def envuelto(d: pd.DataFrame) -> Any:
        try:
            return check(d)
        except (KeyError, TypeError, ValueError):
            return True

    return envuelto


def esquema(contrato: Contrato | None = None) -> pa.DataFrameSchema:
    """Esquema pandera con los tres niveles del contrato."""
    contrato = contrato or cargar()
    checks = []
    for d in contrato.distribucion:
        medida, condicion = DISTRIBUCION[d["id"]]
        lote = lambda df, d=d, m=medida, ok=condicion: bool(ok(m(df, d), d))  # noqa: E731
        checks.append(pa.Check(_check_lote(lote), name=d["id"]))
    for r in contrato.reglas:
        filas = lambda df, f=REGLAS[r["id"]]: f(df).fillna(False).astype(bool)  # noqa: E731
        checks.append(pa.Check(_check_lote(filas), name=r["id"]))
    columnas = {c.nombre: _columna_pa(c) for c in contrato.columnas}
    return pa.DataFrameSchema(columnas, checks=checks, strict=True, name=f"{contrato.nombre} v{contrato.version}")


def _normalizar(df: pd.DataFrame, contrato: Contrato) -> pd.DataFrame:
    """El texto vacío cuenta como dato faltante (igual que en la BD, donde se guarda como null)."""
    df = df.copy()
    for c in contrato.columnas:
        if c.tipo == "texto" and c.nombre in df:
            df[c.nombre] = df[c.nombre].astype(object).where(~_vacio(df[c.nombre]), None)
    return df


def _mensajes(fallas: pd.DataFrame, df: pd.DataFrame, contrato: Contrato) -> list[str]:
    """Traduce los failure_cases de pandera a mensajes ordenados por nivel."""
    distribucion = {d["id"]: d for d in contrato.distribucion}
    reglas = {r["id"]: r for r in contrato.reglas}
    tipos = {c.nombre: c.tipo for c in contrato.columnas}
    traduccion = {"not_nullable": "vacías en columna requerida", "field_uniqueness": "duplicadas en columna única"}
    check = fallas["check"].astype(str)
    errores = []

    # Nivel 1. Si una columna no pudo convertirse a su tipo, solo se reporta eso: sus demás checks
    # fallan en cadena (comparar texto con números) y serían ruido.
    por_columna = fallas[~check.isin([*distribucion, *reglas])]
    sin_tipo = set(fallas.loc[check.str.startswith("coerce_dtype"), "column"])
    es_tipo = por_columna["check"].astype(str).str.startswith("coerce_dtype")
    por_columna = por_columna[~por_columna["column"].isin(sin_tipo) | es_tipo]
    for (columna, nombre), g in por_columna.groupby(["column", "check"], dropna=False, sort=False):
        if nombre == "column_in_dataframe":
            errores.append(f"Nivel 1 · Faltan columnas del contrato: {sorted(g['failure_case'])}")
        elif nombre == "column_in_schema":
            errores.append(f"Nivel 1 · Columnas fuera del contrato: {sorted(g['failure_case'])}")
        else:
            problema = traduccion.get(nombre, f"no cumplen {nombre}")
            if str(nombre).startswith("coerce_dtype"):
                problema = f"no son de tipo {tipos.get(columna)}"
            ej = g.iloc[0]
            errores.append(
                f"Nivel 1 · {columna}: {len(g)} fila(s) {problema} (ej. fila {ej['index']}: {ej['failure_case']!r})"
            )

    # Nivel 2: se informa el valor observado del lote para saber qué tan lejos está del rango.
    for d in contrato.distribucion:
        if (check == d["id"]).any():
            observado = DISTRIBUCION[d["id"]][0](df, d)
            limites = {k: v for k, v in d.items() if k not in ("id", "descripcion")}
            errores.append(f"Nivel 2 · {d['id']}: {d['descripcion']} (observado: {observado:.4g}; límites: {limites})")

    # Nivel 3: pandera repite cada fila fallida una vez por columna; se cuentan filas únicas.
    for r in contrato.reglas:
        g = fallas[check == r["id"]]
        if len(g):
            filas = [int(i) for i in pd.unique(g["index"].dropna())]
            errores.append(
                f"Nivel 3 · {r['id']}: {len(filas)} fila(s) no cumplen ({r['descripcion']}); ej. {filas[:5]}"
            )
    return errores


def validar(df: pd.DataFrame, contrato: Contrato | None = None) -> list[str]:
    """Devuelve los incumplimientos del contrato en los tres niveles (lista vacía si cumple)."""
    contrato = contrato or cargar()
    normalizado = _normalizar(df, contrato)
    try:
        esquema(contrato).validate(normalizado, lazy=True)
    except SchemaErrors as e:
        return _mensajes(e.failure_cases, normalizado, contrato)
    return []


# ---------------------------------------------------------------------------- Supabase / web


def _literal(v: Any) -> str:
    return f"'{str(v).replace(chr(39), chr(39) * 2)}'" if isinstance(v, str) else str(v)


def _checks_sql(c: Columna) -> list[str]:
    col = c.db
    conv = (lambda v: f"date {_literal(str(v))}") if c.tipo == "fecha" else _literal
    checks = []
    if c.tipo == "texto":
        checks.append(f"{col} <> ''")  # el vacío se guarda como null
    if c.valores is not None:
        checks.append(f"{col} in ({', '.join(_literal(v) for v in c.valores)})")
    if c.min is not None:
        checks.append(f"{col} >= {conv(c.min)}")
    if c.max is not None:
        checks.append(f"{col} <= {conv(c.max)}")
    if c.patron is not None:
        checks.append(f"{col} ~ {_literal(c.patron)}")
    return checks


def sql(contrato: Contrato | None = None) -> str:
    """SQL para Supabase: tabla con restricciones del contrato y RLS que solo permite insertar."""
    contrato = contrato or cargar()
    t = f"public.{contrato.tabla_db}"
    cols = [
        "id bigint generated always as identity primary key",
        "creado_en timestamptz not null default now()",
        "origen text not null default 'formulario' check (origen in ('formulario', 'script'))",
    ]
    for c in contrato.columnas:
        partes = [c.db, TIPOS_SQL[c.tipo]]
        partes += ["not null"] if c.requerido else []
        partes += ["unique"] if c.unico else []
        if checks := _checks_sql(c):
            partes.append(f"check ({' and '.join(checks)})")
        cols.append(" ".join(partes))
    cols += [f"constraint {r['id']} check ({' '.join(r['sql'].split())})" for r in contrato.reglas]

    comentarios = [
        f"comment on column {t}.{c.db} is {_literal(' — '.join(filter(None, [c.nombre, c.descripcion])))};"
        for c in contrato.columnas
    ]
    cuerpo = ",\n  ".join(cols)
    origen = f"contratos/{contrato.nombre}.yaml v{contrato.version}"
    return f"""-- Generado desde {origen}. No editar a mano: make contrato-sql
-- Pegar en Supabase > SQL Editor > Run. Para aplicar un contrato nuevo en desarrollo: drop table {t};

create table if not exists {t} (
  {cuerpo}
);

comment on table {t} is {_literal(f"Contrato {contrato.nombre} v{contrato.version}")};
{chr(10).join(comentarios)}

-- Seguridad: la clave pública (anon/publishable) solo puede INSERTAR. Leer, editar y borrar
-- queda para el panel de Supabase o la clave service_role (que nunca va en el HTML ni en Git).
alter table {t} enable row level security;
grant insert on {t} to anon;
grant usage on sequence {t}_id_seq to anon;
drop policy if exists anon_solo_inserta on {t};
create policy anon_solo_inserta on {t} for insert to anon with check (true);
"""


def a_registros_db(df: pd.DataFrame, contrato: Contrato | None = None, origen: str = "script") -> list[dict]:
    """Convierte el DataFrame en dicts con nombres de la BD, vacíos como None y fechas ISO."""
    contrato = contrato or cargar()
    registros = []
    for _, fila in df[contrato.nombres].iterrows():
        reg = {"origen": origen}
        for c in contrato.columnas:
            v = fila[c.nombre]
            if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                reg[c.db] = None
            elif c.tipo == "entero":
                reg[c.db] = int(v)
            elif c.tipo == "fecha":
                reg[c.db] = pd.Timestamp(v).date().isoformat()
            else:
                reg[c.db] = str(v)
        registros.append(reg)
    return registros


def js(contrato: Contrato | None = None) -> str:
    """Contrato como window.CONTRATO para el formulario (funciona abriendo el HTML con doble clic)."""
    contrato = contrato or cargar()
    grupos: dict[str, list] = {}
    for c in contrato.columnas:
        campo = {k: v for k, v in vars(c).items() if v not in (None, False) and k != "grupo"}
        grupos.setdefault(c.grupo, []).append({**campo, "db": c.db})
    data = {"nombre": contrato.nombre, "version": contrato.version, "tabla_db": contrato.tabla_db, "grupos": grupos}
    return (
        f"// Generado desde contratos/{contrato.nombre}.yaml. No editar a mano: make contrato-web\n"
        f"window.CONTRATO = {json.dumps(data, ensure_ascii=False, indent=2)};\n"
    )
