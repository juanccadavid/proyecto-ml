"""Contrato de datos: carga el YAML de contratos/ y lo aplica.

Del mismo contrato salen la validación de DataFrames en tres niveles (pandera), el SQL de
Supabase (tabla + CHECK + RLS), los registros listos para la API y la configuración del formulario web.

Dos contratos, según la etapa:

- **De origen** (``cargar()``): el extracto completo, con sus datos personales. Lo usa quien custodia los datos
  (antes de anonimizar), los datos sintéticos (que imitan ese extracto) y Supabase.
- **Del dataset anonimizado para ML** (``para_ml()``): el mismo contrato sin las columnas ``pii: eliminar``, con
  las ``pii: seudonimizar`` como ``<COLUMNA>_SEUDO`` (HMAC de 64 caracteres hex) y sin las reglas que usan
  columnas eliminadas. Exige igual todas las variables del modelo y del objetivo, y como es estricto, rechaza
  cualquier columna personal que haya quedado en el archivo. Lo usa el modo ``real_anonimizado``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.errors import SchemaErrors

from proyecto_ml.cleaning import clean_dataset
from proyecto_ml.config import ROOT_DIR, fecha_corte

CONTRATO_ASOCIADOS = ROOT_DIR / "contratos" / "asociados.yaml"
PII = ("eliminar", "seudonimizar")
SUFIJO_SEUDONIMO = "_SEUDO"
PATRON_SEUDONIMO = "^[0-9a-f]{64}$"  # HMAC-SHA256 completo en hexadecimal (texto, nunca un número)
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
    pii: str | None = None  # "eliminar" o "seudonimizar" (ver contratos/asociados.yaml)

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
    if invalidos := {c.pii for c in columnas if c.pii} - set(PII):
        raise ValueError(f"pii debe ser uno de {PII}: {invalidos}")
    meta = (raw["nombre"], str(raw["version"]), raw["tabla_db"], raw["objetivo"])
    return Contrato(*meta, columnas, raw.get("reglas", []), raw.get("distribucion", []))


def columnas_pii(contrato: Contrato | None = None, tratamiento: str | None = None) -> list[str]:
    """Las columnas con datos personales del contrato de origen (todas, o solo las de ``tratamiento``)."""
    contrato = contrato or cargar()
    return [c.nombre for c in contrato.columnas if c.pii and (tratamiento is None or c.pii == tratamiento)]


def columnas_de_regla(regla: dict, contrato: Contrato) -> list[str]:
    """Las columnas que usa una regla de nivel 3, leídas de su SQL (nombres de la BD)."""
    return [c.nombre for c in contrato.columnas if re.search(rf"\b{re.escape(c.db)}\b", regla["sql"])]


def para_ml(contrato: Contrato | None = None) -> Contrato:
    """El contrato del dataset anonimizado para ML (ver el docstring del módulo).

    No relaja nada de lo que usa el modelo: las variables, el objetivo, sus tipos, rangos, la distribución y las
    reglas que no tocan PII siguen exigidas. Lo único que cambia son las columnas personales.
    """
    contrato = contrato or cargar()
    columnas = []
    for c in contrato.columnas:
        if c.pii == "eliminar":
            continue
        if c.pii == "seudonimizar":
            columnas.append(
                Columna(
                    nombre=c.nombre + SUFIJO_SEUDONIMO,
                    grupo=c.grupo,
                    tipo="texto",
                    requerido=c.requerido,
                    unico=c.unico,
                    patron=PATRON_SEUDONIMO,
                    descripcion=f"Seudónimo (HMAC-SHA256) de {c.nombre}",
                )
            )
        else:
            columnas.append(c)
    quedan = {c.nombre for c in columnas}
    reglas = [r for r in contrato.reglas if set(columnas_de_regla(r, contrato)) <= quedan]
    return Contrato(
        f"{contrato.nombre}_ml",
        contrato.version,
        contrato.tabla_db,
        contrato.objetivo,
        columnas,
        reglas,
        contrato.distribucion,
    )


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


def _o_vacio(condicion: pd.Series, *columnas: pd.Series) -> pd.Series:
    """Como un CHECK de SQL: si falta alguno de los datos, la regla no aplica (el vacío lo reporta el nivel 1)."""
    return condicion | pd.concat([_vacio(c) for c in columnas], axis=1).any(axis=1)


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
        # Edad a FECHA_CORTE (config.fecha_corte): la misma fecha que usan la foto y EDAD en abandono.preparar.
        lambda d, p: ((pd.Timestamp(fecha_corte()) - _fecha(d["FECHANACIMIENTO"])).dt.days / 365.25).median(),
        lambda v, p: p["min"] <= v <= p["max"],
    ),
}

# Nivel 3: versión pandas de las reglas del YAML (mismo id). True en las filas que cumplen.
REGLAS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "retiro_posterior_ingreso": lambda d: _o_vacio(
        _fecha(d["FECHARETIRO"]) > _fecha(d["FECHAINGRESOLABORA"]), d["FECHARETIRO"], d["FECHAINGRESOLABORA"]
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
    "fin_contrato_posterior_ingreso": lambda d: _o_vacio(
        _fecha(d["FECHATERMINACONTRATO"]) > _fecha(d["FECHAINGRESOLABORA"]),
        d["FECHATERMINACONTRATO"],
        d["FECHAINGRESOLABORA"],
    ),
    "mayor_de_edad_al_ingreso": lambda d: _o_vacio(
        _fecha(d["FECHAINGRESOLABORA"]) >= _fecha(d["FECHANACIMIENTO"]) + pd.DateOffset(years=18),
        d["FECHAINGRESOLABORA"],
        d["FECHANACIMIENTO"],
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


def sin_valores(errores: list[str]) -> list[str]:
    """Los mensajes de ``validar`` sin los valores de ejemplo (``ej. fila 3: '...'``) ni números de fila: para
    mostrarlos con datos reales sin que un dato personal termine en la consola o en un log."""
    return [re.sub(r"\s*[(;]\s*ej\.[^)]*\)?$", "", e).strip() for e in errores]


def limpiar_extracto(df: pd.DataFrame, contrato: Contrato | None = None, *, verbose: bool = False) -> pd.DataFrame:
    """``clean_dataset`` con la configuración que no rompe el contrato de asociados.

    Para el extracto crudo del fondo, ANTES de anonimizar (scripts/anonimizar.py --limpiar). Con los
    valores por defecto de ``clean_dataset`` el contrato falla: renombra todas las columnas (#RETIROS ->
    retiros), borra columnas opcionales que vienen vacías y vuelve números los teléfonos y la cuenta
    (pierden los ceros a la izquierda). Aquí:

    - ``normalize_names=False``: los nombres del contrato no se tocan.
    - ``numeric_columns``: solo las que el contrato declara ``entero`` (los textos de dígitos siguen como texto).
    - ``drop_empty=False``: una columna opcional vacía sigue siendo parte del contrato.
    """
    contrato = contrato or cargar()
    enteras = [c.nombre for c in contrato.columnas if c.tipo == "entero" and c.nombre in df]
    return clean_dataset(df, normalize_names=False, numeric_columns=enteras, drop_empty=False, verbose=verbose)


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


def _columnas_sql(contrato: Contrato) -> list[str]:
    """Columnas del contrato con sus restricciones, y las reglas entre columnas."""
    cols = []
    for c in contrato.columnas:
        partes = [c.db, TIPOS_SQL[c.tipo]]
        partes += ["not null"] if c.requerido else []
        partes += ["unique"] if c.unico else []
        if checks := _checks_sql(c):
            partes.append(f"check ({' and '.join(checks)})")
        cols.append(" ".join(partes))
    return cols + [f"constraint {r['id']} check ({' '.join(r['sql'].split())})" for r in contrato.reglas]


def sql_formulario(contrato: Contrato | None = None) -> str:
    """Solo el equipo inserta desde el formulario: ``equipo`` + ``es_del_equipo()`` + política en ``<tabla>``.

    - La clave pública (anon) sola no puede insertar: hay que iniciar sesión (Supabase Auth, correo y
      contraseña) con un correo de ``public.equipo``. Lo enviado queda de una vez en ``<tabla>``, con
      todas las restricciones del contrato (y sin repetir cédula ni IDASOCIADOS).
    """
    contrato = contrato or cargar()
    t = f"public.{contrato.tabla_db}"
    correo = "lower(coalesce(auth.jwt() ->> 'email', ''))"
    return f"""-- Formulario: solo el equipo puede insertar (con sesión), directo en {t}.
-- 1) La clave pública sola (anon, sin sesión) no inserta.
drop policy if exists anon_solo_inserta on {t};
revoke insert on {t} from anon;
revoke usage on sequence {t}_id_seq from anon;

-- 2) El equipo: correos que pueden enviar el formulario. Se administra en el panel de Supabase
--    (Table Editor) o con la clave secreta; con RLS y sin políticas, anon y authenticated no la ven.
create table if not exists public.equipo (
  email text primary key check (email = lower(email)),
  nombre text,
  agregado_en timestamptz not null default now()
);
alter table public.equipo enable row level security;

create or replace function public.es_del_equipo() returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (select 1 from public.equipo where email = {correo})
$$;
revoke all on function public.es_del_equipo() from public;
grant execute on function public.es_del_equipo() to authenticated;

-- 3) Con sesión y correo del equipo: insertar (no leer, editar ni borrar) en {t}.
grant insert on {t} to authenticated;
grant usage on sequence {t}_id_seq to authenticated;
drop policy if exists equipo_inserta on {t};
create policy equipo_inserta on {t} for insert to authenticated
  with check (public.es_del_equipo() and origen = 'formulario');
"""


def migracion_formulario(contrato: Contrato | None = None) -> str:
    """supabase/migraciones/003_formulario_directo.sql: para una base que ya corrió la 002."""
    contrato = contrato or cargar()
    t = contrato.tabla_db
    return f"""-- Migración 003: el equipo inserta directo en public.{t}, sin revisión (contrato v{contrato.version}).
-- Para una base que ya corrió la 002. Pegar en Supabase > SQL Editor > Run.
-- Generado con contrato.migracion_formulario(); una base nueva lo trae en supabase/asociados.sql.
-- Ojo: borra public.{t}_entrada (y lo que hubiera pendiente: son datos de prueba del formulario).

alter table public.{t} enable row level security;
drop table if exists public.{t}_entrada;

{sql_formulario(contrato)}"""


def sql(contrato: Contrato | None = None) -> str:
    """SQL para Supabase: tabla con las restricciones del contrato y el formulario solo para el equipo."""
    contrato = contrato or cargar()
    t = f"public.{contrato.tabla_db}"
    cols = [
        "id bigint generated always as identity primary key",
        "creado_en timestamptz not null default now()",
        "origen text not null default 'formulario' check (origen in ('formulario', 'script'))",
        *_columnas_sql(contrato),
    ]
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

-- Seguridad: con RLS, la clave pública sola no lee, inserta, edita ni borra {t}. Inserta solo el equipo con
-- sesión (abajo); leer, la clave secreta (scripts subir/bajar, nunca en el HTML ni en Git) y el panel de Supabase.
alter table {t} enable row level security;

{sql_formulario(contrato)}"""


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


def desde_registros_db(registros: list[dict], contrato: Contrato | None = None) -> pd.DataFrame:
    """Inverso de a_registros_db: filas de la BD -> DataFrame con los nombres y tipos del contrato.

    Descarta las columnas propias de la BD (id, creado_en, origen): no son parte del contrato.
    """
    contrato = contrato or cargar()
    db = pd.DataFrame(registros, columns=[c.db for c in contrato.columnas])
    df = pd.DataFrame(index=db.index)
    for c in contrato.columnas:
        if c.tipo == "entero":
            v = pd.to_numeric(db[c.db]).astype("Int64")
            df[c.nombre] = v.astype("int64") if v.notna().all() else v
        elif c.tipo == "fecha":
            df[c.nombre] = pd.to_datetime(db[c.db])
        else:
            df[c.nombre] = db[c.db].fillna("")  # vacío como "" (igual que el generador y el formulario)
    return df


def js(contrato: Contrato | None = None) -> str:
    """Contrato como window.CONTRATO para el formulario (funciona abriendo el HTML con doble clic)."""
    contrato = contrato or cargar()
    grupos: dict[str, list] = {}
    for c in contrato.columnas:
        campo = {k: v for k, v in vars(c).items() if v not in (None, False) and k not in ("grupo", "pii")}
        grupos.setdefault(c.grupo, []).append({**campo, "db": c.db})
    data = {"nombre": contrato.nombre, "version": contrato.version, "tabla_db": contrato.tabla_db}
    data |= {"grupos": grupos}
    return (
        f"// Generado desde contratos/{contrato.nombre}.yaml. No editar a mano: make contrato-web\n"
        f"window.CONTRATO = {json.dumps(data, ensure_ascii=False, indent=2)};\n"
    )
