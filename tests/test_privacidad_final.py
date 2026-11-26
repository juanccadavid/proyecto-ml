"""Controles finales de publicación y predicción."""

import re
from pathlib import Path

import yaml

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


def test_los_contenedores_no_reciben_los_secretos_del_env():
    """Antes app y campeon cargaban el .env completo (env_file): SUPABASE_SECRET_KEY, ANON_SALT, tokens... Ahora cada
    contenedor recibe solo las variables de datos que lee, y el .env tampoco entra a la imagen."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    secreto = re.compile(r"SECRET|TOKEN|PAT$|SALT|DB_URL|KEY|PASSWORD", re.IGNORECASE)
    for nombre, servicio in compose["services"].items():
        assert "env_file" not in servicio, nombre
        variables = list(servicio.get("environment") or {})
        assert not [v for v in variables if secreto.search(v)], nombre
    assert {"ORIGEN_DATOS", "FECHA_CORTE"} <= set(compose["services"]["app"]["environment"])
    assert ".env" in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
