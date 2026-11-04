"""Anonimización ANTES de que los datos salgan de la organización origen (la usa scripts/anonimizar.py).

- Datos personales: el contrato de origen marca cada columna con ``pii: eliminar`` o ``pii: seudonimizar``
  (contratos/asociados.yaml). Las primeras se eliminan; las segundas salen como ``<COLUMNA>_SEUDO``.
- Seudónimo: HMAC-SHA256 con una clave secreta (``ANON_SALT``), los 64 caracteres hexadecimales completos (texto,
  nunca un número: un entero recortaría el hash y podría chocar). El mismo identificador da siempre el mismo
  seudónimo, así que los registros se pueden unir entre extractos. Sin la clave no se recalcula por fuerza
  bruta sobre cédulas.
- Cuasi-identificadores (edad, salario) se pueden agrupar en rangos (``bins``).

**Seudonimizar no es anonimizar de forma irreversible:** quien tenga la clave puede recalcular el seudónimo de
una cédula conocida y reconocer su fila. La clave la guarda solo quien custodia los datos; nunca va al
repositorio, a MLflow ni a los logs. Una tabla de correspondencia (seudónimo -> cédula), si se necesita para
contactar asociados, vive fuera del pipeline de ML, con el custodio.

``anonimizar_extracto`` no devuelve nada si algo queda mal (``AnonimizacionInsegura``): columnas sin decisión,
columnas personales que siguen ahí, seudónimos inválidos o valores personales copiados en otra columna. Sus
mensajes dicen columnas y cantidades, nunca valores.

Detección de PII residual (``verificar``, configurable con ``Deteccion``). Busca en la salida los valores de
las columnas personales DE ESTE extracto, no "cualquier número" (eso daría falsos positivos en masa):

1. **Exacto:** una celda igual a un valor personal de al menos ``largo_min_valor`` caracteres (cualquier columna).
2. **Identificador incrustado:** en las columnas de texto, cada secuencia de dígitos, también escrita con puntos,
   guiones o espacios (``cc 1.037.000.000``, ``300-123-4567``), se compara con las cédulas, teléfonos y cuentas
   del extracto (``largo_min_digitos`` dígitos o más). Solo cuenta si es una de ellas, no por ser un número.
3. **Texto incrustado:** nombres completos, direcciones y correos (valores de varias palabras o con ``@``, de
   ``largo_min_texto`` caracteres o más) dentro de un texto más largo, sin distinguir mayúsculas.
4. **Forma de correo:** una celda con forma de correo electrónico en una columna de texto que se conserva.
5. **Nombre de columna:** ``--conservar`` una columna cuyo nombre sugiere un dato personal (``nombres_sensibles``)
   se rechaza. Si de verdad no es personal, se agrega al contrato (con revisión), no se fuerza aquí.

Límites (no es una garantía de anonimización): un nombre o apellido suelto solo se busca exacto (buscarlo dentro
de textos marcaría "Santa María" o "Calle Gómez"); los valores cortos (menos de ``largo_min_valor`` caracteres,
p. ej. una extensión o un IDASOCIADOS de 4 dígitos) no se buscan; tampoco un dato personal transformado (con
errores, abreviado, en otro orden) ni uno que no esté en las columnas personales del extracto. Los
cuasi-identificadores (edad, barrio, cargo) siguen en la salida: ver docs/privacidad.md.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from proyecto_ml import contrato as k

LARGO_MIN_SAL = 16
LARGO_MIN_VALOR = 6  # los valores personales más cortos (p. ej. una extensión) chocarían con códigos legítimos
SEPARADORES = r"[.\- ]"  # cómo se escribe a mano una cédula o un teléfono: 1.037.000.000, 300-123-4567, 300 123 4567
_RE_CORREO = re.compile(r"[^\s@]+@[^\s@]+\.[a-z]{2,}", re.IGNORECASE)
_PUNTUACION = ".,;:()[]{}<>\"'¡!¿?"


@dataclass(frozen=True)
class Deteccion:
    """Qué busca ``verificar`` en la salida (ver el docstring del módulo). ``None`` en ``nombres_sensibles`` lo
    apaga; los largos se pueden subir si un fondo tiene códigos legítimos que chocan con sus identificadores."""

    largo_min_valor: int = LARGO_MIN_VALOR
    largo_min_digitos: int = 6
    largo_min_texto: int = 8
    buscar_correos: bool = True
    nombres_sensibles: str | None = (
        r"CEDULA|DOCUMENTO|IDENTIFICACION|E_?MAIL|CORREO|TELEFONO|CELULAR|MOVIL|DIRECCION|APELLIDO|NOMBRES"
    )


DETECCION = Deteccion()


class AnonimizacionInsegura(ValueError):
    """La salida conservaría datos personales o no se pudo seudonimizar: no se escribe."""


def _texto_id(value: object) -> str:
    """El identificador como texto estable: 1037000000, 1037000000.0 y ' 1037000000 ' dan lo mismo."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        raise ValueError("identificador vacío")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    texto = str(value).strip()
    if not texto:
        raise ValueError("identificador vacío")
    return texto


def _clave(salt: str | None) -> str:
    clave = salt or os.environ.get("ANON_SALT") or ""
    if len(clave) < LARGO_MIN_SAL:
        raise AnonimizacionInsegura(f"ANON_SALT debe tener al menos {LARGO_MIN_SAL} caracteres (make salt)")
    return clave


def hash_value(value: object, salt: str) -> str:
    """Seudónimo de un identificador: HMAC-SHA256 completo, 64 caracteres hexadecimales."""
    if not salt or len(salt) < LARGO_MIN_SAL:
        raise ValueError(f"ANON_SALT debe tener al menos {LARGO_MIN_SAL} caracteres")
    return hmac.new(salt.encode(), _texto_id(value).encode(), hashlib.sha256).hexdigest()


def seudonimizar(serie: pd.Series, salt: str) -> pd.Series:
    """Seudónimo de cada fila. Un identificador vacío no se puede seudonimizar: error (sin mostrar valores)."""
    vacios = serie.isna() | serie.astype("string").str.strip().eq("")
    if vacios.any():
        raise AnonimizacionInsegura(f"{serie.name}: {int(vacios.sum())} identificador(es) vacío(s)")
    return serie.map(lambda v: hash_value(v, salt)).astype("string")


def anonymize(
    df: pd.DataFrame,
    *,
    id_columns: Iterable[str] = (),
    drop_columns: Iterable[str] = (),
    bins: dict[str, list[float]] | None = None,
    salt: str | None = None,
) -> pd.DataFrame:
    """Versión genérica (sin contrato): seudonimiza ``id_columns`` en su lugar y elimina ``drop_columns``.

    Para el extracto de asociados usa ``anonimizar_extracto``, que además verifica la salida.
    """
    salt = _clave(salt)
    out = df.copy()
    for col in id_columns:
        out[col] = seudonimizar(out[col], salt)
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    for col, edges in (bins or {}).items():
        out[col] = pd.cut(out[col], bins=edges, include_lowest=True).astype(str)
    return out


def _como_texto(v: object) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()


def _normalizar(texto: str) -> str:
    """Minúsculas, sin puntuación en los bordes de cada palabra y con un solo espacio entre palabras."""
    return " ".join(t.strip(_PUNTUACION) for t in texto.lower().split() if t.strip(_PUNTUACION))


def _es_texto(serie: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
        or isinstance(serie.dtype, pd.CategoricalDtype)
    )


@dataclass(frozen=True)
class _Personales:
    """Los valores de las columnas personales del extracto, preparados para cada tipo de búsqueda."""

    exactos: set[str]
    digitos: set[str]  # identificadores numéricos (cédula, teléfono, cuenta) solo con sus dígitos
    textos: set[str]  # nombres completos, direcciones y correos, normalizados
    palabras_max: int

    @classmethod
    def de(cls, original: pd.DataFrame, columnas: set[str], d: Deteccion) -> _Personales:
        valores = {_como_texto(v) for c in sorted(columnas & set(original.columns)) for v in original[c].dropna()}
        valores = {v for v in valores if v}
        numerico = re.compile(rf"\+?\d(?:{SEPARADORES}?\d)*")
        digitos = {re.sub(r"\D", "", v) for v in valores if numerico.fullmatch(v)}
        textos = {_normalizar(v) for v in valores if not numerico.fullmatch(v)}
        textos = {t for t in textos if len(t) >= d.largo_min_texto and (" " in t or "@" in t)}
        return cls(
            exactos={v for v in valores if len(v) >= d.largo_min_valor},
            digitos={v for v in digitos if len(v) >= d.largo_min_digitos},
            textos=textos,
            palabras_max=max((t.count(" ") + 1 for t in textos), default=0),
        )

    def numero_incrustado(self, texto: str) -> bool:
        if not self.digitos:
            return False
        corridas = set(re.findall(r"\d+", texto))  # 1037000000 tal cual
        corridas |= {re.sub(r"\D", "", m) for m in re.findall(rf"\d(?:{SEPARADORES}?\d)+", texto)}  # 1.037.000.000
        return not corridas.isdisjoint(self.digitos)

    def texto_incrustado(self, texto: str) -> bool:
        if not self.textos:
            return False
        palabras = _normalizar(texto).split()
        for n in range(1, min(self.palabras_max, len(palabras)) + 1):  # ventanas de n palabras seguidas
            if any(" ".join(palabras[i : i + n]) in self.textos for i in range(len(palabras) - n + 1)):
                return True
        return False


def columnas_sensibles(columnas: Iterable[str], deteccion: Deteccion = DETECCION) -> list[str]:
    """Las ``columnas`` cuyo nombre sugiere un dato personal (``Deteccion.nombres_sensibles``)."""
    if not deteccion.nombres_sensibles:
        return []
    patron = re.compile(deteccion.nombres_sensibles, re.IGNORECASE)
    return [c for c in columnas if patron.search(str(c).replace(" ", "_"))]


def verificar(
    original: pd.DataFrame,
    salida: pd.DataFrame,
    eliminadas: set[str],
    seudonimizadas: set[str],
    salt: str,
    deteccion: Deteccion = DETECCION,
) -> list[str]:
    """Problemas de la salida (lista vacía si está bien). Solo nombres de columnas, reglas y cantidades, nunca
    valores. Qué busca y qué no: ver el docstring del módulo. Una lista vacía no garantiza que no quede PII."""
    problemas = []
    if len(salida) != len(original):
        problemas.append(f"la salida tiene {len(salida)} filas y el extracto {len(original)}")
    if quedan := sorted((eliminadas | seudonimizadas) & set(salida.columns)):
        problemas.append(f"columnas personales que siguen en la salida: {quedan}")
    patron = re.compile(k.PATRON_SEUDONIMO)
    for c in sorted(seudonimizadas & set(original.columns)):
        destino = c + k.SUFIJO_SEUDONIMO
        if destino not in salida:
            problemas.append(f"falta {destino} (seudónimo de {c})")
        elif not salida[destino].astype(str).map(lambda v: bool(patron.fullmatch(v))).all():
            problemas.append(f"{destino}: hay valores que no son un seudónimo de 64 caracteres hexadecimales")
    personales = _Personales.de(original, eliminadas | seudonimizadas, deteccion)
    for col in salida.columns:
        textos = salida[col].dropna().map(_como_texto)
        exacto = textos.map(personales.exactos.__contains__).astype(bool)  # (isin con texto Arrow es lentísimo)
        if (n := int(exacto.sum())) > 0:
            problemas.append(f"{col}: {n} fila(s) con valores de columnas personales del extracto")
        if (n := int(textos.str.contains(salt, regex=False).sum())) > 0:
            problemas.append(f"{col}: {n} fila(s) contienen la clave ANON_SALT")
        if str(col).endswith(k.SUFIJO_SEUDONIMO) or not _es_texto(salida[col]):
            continue  # el seudónimo ya se revisó arriba; números y fechas, con la comparación exacta
        resto = textos[~exacto]  # lo exacto ya se contó: no se cuenta dos veces
        # cada valor distinto se revisa una vez (barrio, cargo... se repiten)
        distintos = pd.Series(resto.unique(), dtype=object)
        busquedas: list[tuple[Callable[[str], bool], str]] = [
            (personales.numero_incrustado, "con un número de una columna personal (cédula, teléfono) dentro"),
            (personales.texto_incrustado, "con un nombre, dirección o correo del extracto dentro del texto"),
        ]
        if deteccion.buscar_correos:
            busquedas.append((lambda v: bool(_RE_CORREO.search(v)), "con forma de correo electrónico"))
        for buscar, problema in busquedas:
            marcados = set(distintos[distintos.map(buscar).astype(bool)])
            if marcados and (n := int(resto.map(marcados.__contains__).astype(bool).sum())) > 0:
                problemas.append(f"{col}: {n} fila(s) {problema}")
    return problemas


def anonimizar_extracto(
    df: pd.DataFrame,
    contrato: k.Contrato | None = None,
    *,
    salt: str | None = None,
    conservar: Iterable[str] = (),
    eliminar: Iterable[str] = (),
    seudonimizar_columnas: Iterable[str] = (),
    bins: dict[str, list[float]] | None = None,
    deteccion: Deteccion = DETECCION,
) -> pd.DataFrame:
    """El extracto listo para compartir, según el contrato de origen, o ``AnonimizacionInsegura``.

    Toda columna del extracto necesita una decisión: estar en el contrato (y entonces su ``pii`` dice qué hacer),
    o venir en ``conservar``, ``eliminar`` o ``seudonimizar_columnas``. Una columna desconocida no pasa en silencio:
    podría ser un dato personal que el contrato no conoce.
    """
    contrato = contrato or k.cargar()
    salt = _clave(salt)
    conservar, eliminar, a_seudo = set(conservar), set(eliminar), set(seudonimizar_columnas)
    a_eliminar = set(k.columnas_pii(contrato, "eliminar")) | eliminar
    a_seudo |= set(k.columnas_pii(contrato, "seudonimizar"))
    if choque := sorted(conservar & (a_eliminar | a_seudo)):
        raise AnonimizacionInsegura(f"Columnas personales marcadas para conservar: {choque}")
    if sensibles := columnas_sensibles(sorted(conservar - set(contrato.nombres)), deteccion):
        raise AnonimizacionInsegura(
            f"Por su nombre, {sensibles} parecen datos personales y se marcaron para conservar: usa --eliminar o "
            "--seudonimizar. Si de verdad no son personales, agrégalas al contrato (con revisión del equipo)."
        )
    decididas = set(contrato.nombres) | conservar | a_eliminar | a_seudo
    if sin_decision := [c for c in df.columns if c not in decididas]:
        raise AnonimizacionInsegura(
            f"Columnas que el contrato no conoce y no tienen decisión: {sin_decision}. Indica para cada una "
            "--conservar, --eliminar o --seudonimizar."
        )
    salida = df.drop(columns=[c for c in df.columns if c in a_eliminar])
    for c in [c for c in df.columns if c in a_seudo]:  # el seudónimo queda en la posición de la columna original
        posicion = salida.columns.get_loc(c)
        seudonimos = seudonimizar(df[c], salt)
        salida = salida.drop(columns=c)
        salida.insert(posicion, c + k.SUFIJO_SEUDONIMO, seudonimos)
    for col, edges in (bins or {}).items():
        salida[col] = pd.cut(salida[col], bins=edges, include_lowest=True).astype(str)
    eliminadas, seudonimizadas = a_eliminar & set(df.columns), a_seudo & set(df.columns)
    if problemas := verificar(df, salida, eliminadas, seudonimizadas, salt, deteccion):
        raise AnonimizacionInsegura("La salida no es segura:\n  - " + "\n  - ".join(problemas))
    return salida
