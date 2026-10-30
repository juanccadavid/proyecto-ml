"""Contrato de datos: carga el YAML de contratos/ y lo aplica.

Del mismo contrato salen la validación de DataFrames, el SQL de Supabase (tabla + CHECK + RLS),
los registros listos para la API y la configuración del formulario web.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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

    @property
    def nombres(self) -> list[str]:
        return [c.nombre for c in self.columnas]


def cargar(path: str | Path = CONTRATO_ASOCIADOS) -> Contrato:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    columnas = [Columna(grupo=grupo, **col) for grupo, cols in raw["grupos"].items() for col in cols]
    if invalidos := {c.tipo for c in columnas} - TIPOS_SQL.keys():
        raise ValueError(f"Tipos no soportados en el contrato: {invalidos}")
    return Contrato(raw["nombre"], str(raw["version"]), raw["tabla_db"], raw["objetivo"], columnas, raw["reglas"])


# ---------------------------------------------------------------------------- validación


def _vacio(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.strip() == "")


def _fecha(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.where(~_vacio(s)), errors="coerce")


# Versión pandas de las reglas del YAML (mismo id). Cada una devuelve True en las filas que cumplen.
REGLAS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
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


def _error(col: str, malas: pd.Series, problema: str, valores: pd.Series) -> str:
    idx = malas[malas].index
    return f"{col}: {len(idx)} fila(s) {problema} (ej. fila {idx[0]}: {valores.loc[idx[0]]!r})"


def _validar_columna(c: Columna, s: pd.Series) -> list[str]:
    errores = []
    vacio = _vacio(s)
    if c.requerido and vacio.any():
        errores.append(_error(c.nombre, vacio, "vacías en columna requerida", s))
    s = s[~vacio]
    falso = pd.Series(False, index=s.index)

    if c.tipo == "entero":
        v = pd.to_numeric(s, errors="coerce")
        malos = v.isna() | (v % 1 != 0)
    elif c.tipo == "fecha":
        v = pd.to_datetime(s, errors="coerce")
        malos = v.isna()
    else:
        v = s.astype(str)
        malos = falso
    if malos.any():
        errores.append(_error(c.nombre, malos, f"no son de tipo {c.tipo}", s))
    v, s = v[~malos], s[~malos]

    limite = pd.Timestamp if c.tipo == "fecha" else (lambda x: x)
    checks = [
        (c.valores is not None, lambda: ~v.isin(c.valores), f"fuera de {c.valores}"),
        (c.min is not None, lambda: v < limite(c.min), f"menores que {c.min}"),
        (c.max is not None, lambda: v > limite(c.max), f"mayores que {c.max}"),
        (c.patron is not None, lambda: ~v.str.fullmatch(c.patron), f"no cumplen el patrón {c.patron}"),
        (c.unico, lambda: v.duplicated(keep=False), "duplicadas en columna única"),
    ]
    for aplica, condicion, problema in checks:
        if aplica and (malas := condicion()).any():
            errores.append(_error(c.nombre, malas, problema, s))
    return errores


def validar(df: pd.DataFrame, contrato: Contrato | None = None) -> list[str]:
    """Devuelve la lista de incumplimientos del contrato (vacía si el DataFrame cumple)."""
    contrato = contrato or cargar()
    errores = []
    if faltan := [c for c in contrato.nombres if c not in df.columns]:
        errores.append(f"Faltan columnas del contrato: {faltan}")
    if sobran := [c for c in df.columns if c not in contrato.nombres]:
        errores.append(f"Columnas fuera del contrato: {sobran}")
    for c in contrato.columnas:
        if c.nombre in df.columns:
            errores += _validar_columna(c, df[c.nombre])
    if faltan:
        return errores
    for regla in contrato.reglas:
        ok = REGLAS[regla["id"]](df).fillna(False)
        if not ok.all():
            filas = list(ok[~ok].index[:5])
            errores.append(
                f"Regla {regla['id']}: {(~ok).sum()} fila(s) no cumplen ({regla['descripcion']}); ej. {filas}"
            )
    return errores


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
