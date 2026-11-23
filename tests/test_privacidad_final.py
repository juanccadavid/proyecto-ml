"""Controles finales de publicación y predicción."""

from pathlib import Path

from proyecto_ml import abandono as a
from proyecto_ml import contrato as k


def test_contrato_publico_no_contiene_valores_del_extracto():
    texto = Path("contratos/asociados.yaml").read_text(encoding="utf-8")
    assert "DataAnonimizada" not in texto
    assert "ANON_SALT" not in texto


def test_contrato_tiene_solo_los_44_nombres_aprobados():
    contrato = k.cargar()
    assert len(contrato.nombres) == 44
    assert len(set(contrato.nombres)) == 44


def test_fechas_crudas_no_entran_al_modelo():
    fechas = {"FECHANACIMIENTO", "FECHAINGRESO", "FECHAINGRESOLABORA", "FECHAULTACTUALIZACION"}
    assert fechas.isdisjoint(a.NUMERICAS + a.CATEGORICAS)


def test_identificador_anonimizado_solo_sirve_para_cruce():
    contrato = k.cargar()
    columna = next(c for c in contrato.columnas if c.nombre == "NROIDENTIFICACION")
    assert columna.unico and columna.requerido
    assert columna.nombre not in a.NUMERICAS + a.CATEGORICAS
