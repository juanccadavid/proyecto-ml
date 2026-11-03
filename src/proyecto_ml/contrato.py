"""Contrato de datos: carga el YAML de contratos/ y lo aplica.

Del mismo contrato salen la validación de DataFrames en tres niveles (pandera), el SQL de
Supabase (tabla + CHECK + RLS), los registros listos para la API y la configuración del formulario web.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.errors import SchemaErrors

from proyecto_ml.cleaning import clean_dataset
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
        lambda d, p: ((pd.Timestamp(p["fecha_corte"]) - _fecha(d["FECHANACIMIENTO"])).dt.days / 365.25).median(),
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


REVISION = ("pendiente", "aprobada", "rechazada")  # columna revision de <tabla>_entrada
# Columnas propias de <tabla>_entrada (no pueden llamarse como una del contrato: ESTADO ya es "estado").
COLUMNAS_ENTRADA = ("id", "creado_en", "enviado_por", "revision", "revisado_en", "motivo_revision")


def _columnas_sql(contrato: Contrato, unicos: bool = True) -> list[str]:
    """Columnas del contrato con sus restricciones (y las reglas entre columnas). Sin ``unicos``, sin UNIQUE."""
    cols = []
    for c in contrato.columnas:
        partes = [c.db, TIPOS_SQL[c.tipo]]
        partes += ["not null"] if c.requerido else []
        partes += ["unique"] if c.unico and unicos else []
        if checks := _checks_sql(c):
            partes.append(f"check ({' and '.join(checks)})")
        cols.append(" ".join(partes))
    return cols + [f"constraint {r['id']} check ({' '.join(r['sql'].split())})" for r in contrato.reglas]


def sql_formulario(contrato: Contrato | None = None) -> str:
    """Solo el equipo inserta, y en una tabla de entrada: ``<tabla>_entrada`` + ``equipo`` + RLS.

    - La clave pública (anon) ya no puede insertar en ``<tabla>``: la basura del formulario no bloquea
      ``supabase-bajar`` ni ocupa cédulas reales.
    - El formulario inicia sesión con un código por correo (Supabase Auth) e inserta en ``<tabla>_entrada``
      solo si el correo está en ``public.equipo``. ``make supabase-revisar`` pasa a ``<tabla>`` lo válido.
    """
    contrato = contrato or cargar()
    t, e = f"public.{contrato.tabla_db}", f"public.{contrato.tabla_db}_entrada"
    correo = "lower(coalesce(auth.jwt() ->> 'email', ''))"
    revision = ", ".join(_literal(x) for x in REVISION)
    cols = [
        "id bigint generated always as identity primary key",
        "creado_en timestamptz not null default now()",
        f"enviado_por text not null default {correo}",
        f"revision text not null default 'pendiente' check (revision in ({revision}))",
        "revisado_en timestamptz",
        "motivo_revision text",
        *_columnas_sql(contrato, unicos=False),  # los repetidos se resuelven al revisar
    ]
    cuerpo = ",\n  ".join(cols)
    return f"""-- Formulario: solo el equipo puede insertar, y en una tabla de entrada que se revisa.
-- 1) La clave pública (anon) ya no inserta en {t}.
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

-- 3) Tabla de entrada: mismas columnas y reglas que {t}, sin UNIQUE. make supabase-revisar pasa lo
--    válido a {t} y marca cada fila (revision) como aprobada o rechazada, con motivo_revision.
create table if not exists {e} (
  {cuerpo}
);
comment on table {e} is {_literal(f"Entrada del formulario (contrato {contrato.nombre} v{contrato.version})")};
alter table {e} enable row level security;
grant insert on {e} to authenticated;
grant usage on sequence {e}_id_seq to authenticated;
drop policy if exists equipo_inserta on {e};
create policy equipo_inserta on {e} for insert to authenticated
  with check (public.es_del_equipo() and revision = 'pendiente' and enviado_por = {correo});
"""


def migracion_formulario(contrato: Contrato | None = None) -> str:
    """supabase/migraciones/002_formulario_equipo.sql: lo mismo que ``sql_formulario``, para una base ya creada."""
    contrato = contrato or cargar()
    return f"""-- Migración 002: formulario solo para el equipo, en una tabla de entrada (contrato v{contrato.version}).
-- Para una base ya creada con supabase/asociados.sql anterior. Pegar en Supabase > SQL Editor > Run.
-- Generado con contrato.migracion_formulario(); una base nueva lo trae en supabase/asociados.sql.
-- Después: agregar los correos del equipo (Table Editor > equipo, o en SQL):
--   insert into public.equipo (email, nombre) values ('correo@dominio.com', 'Nombre');

alter table public.{contrato.tabla_db} enable row level security;

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

-- Seguridad: con RLS y sin políticas, nadie con la clave pública lee, inserta, edita ni borra {t}.
-- Solo la clave secreta (scripts subir/bajar/revisar, nunca en el HTML ni en Git) y el panel de Supabase.
alter table {t} enable row level security;

{sql_formulario(contrato)}"""


def revisar_entradas(entradas: list[dict], existentes: list[dict], contrato: Contrato | None = None) -> list[dict]:
    """Decide qué filas de ``<tabla>_entrada`` pasan a ``<tabla>``: una decisión por entrada, en orden de id.

    Aprueba si la fila cumple el contrato por fila y entre columnas (niveles 1 y 3; la distribución, nivel 2,
    es de lotes y no aplica a una fila) y si sus columnas únicas (CEDULA, IDASOCIADOS) no están ni en
    ``existentes`` (filas de ``<tabla>``) ni en una entrada aprobada antes. Devuelve, por entrada:
    ``id``, ``enviado_por``, ``aprobado``, ``motivo`` y ``registro`` (la fila lista para ``<tabla>``).
    """
    contrato = contrato or cargar()
    por_fila = replace(contrato, distribucion=[])
    unicas = [c for c in contrato.columnas if c.unico]
    usados = {c.db: {fila.get(c.db) for fila in existentes} for c in unicas}
    decisiones = []
    for entrada in sorted(entradas, key=lambda f: f["id"]):
        motivos = validar(desde_registros_db([entrada], contrato), por_fila)
        motivos += [f"{c.nombre} {entrada.get(c.db)} ya existe" for c in unicas if entrada.get(c.db) in usados[c.db]]
        aprobado = not motivos
        if aprobado:
            for c in unicas:
                usados[c.db].add(entrada.get(c.db))
        registro = {"origen": "formulario", **{c.db: entrada.get(c.db) for c in contrato.columnas}}
        decisiones.append(
            {
                "id": entrada["id"],
                "enviado_por": entrada.get("enviado_por"),
                "aprobado": aprobado,
                "motivo": "; ".join(motivos) or None,
                "registro": registro,
            }
        )
    return decisiones


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
        campo = {k: v for k, v in vars(c).items() if v not in (None, False) and k != "grupo"}
        grupos.setdefault(c.grupo, []).append({**campo, "db": c.db})
    data = {"nombre": contrato.nombre, "version": contrato.version, "tabla_db": contrato.tabla_db}
    data |= {"tabla_entrada": f"{contrato.tabla_db}_entrada", "grupos": grupos}
    return (
        f"// Generado desde contratos/{contrato.nombre}.yaml. No editar a mano: make contrato-web\n"
        f"window.CONTRATO = {json.dumps(data, ensure_ascii=False, indent=2)};\n"
    )
