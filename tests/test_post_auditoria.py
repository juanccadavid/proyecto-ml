"""Regresiones posteriores al cambio de esquema."""

from proyecto_ml import abandono as a


def test_nombres_de_predictores_son_unicos():
    predictores = a.NUMERICAS + a.CATEGORICAS
    assert len(predictores) == len(set(predictores))


def test_objetivo_no_es_predictor():
    assert a.TARGET not in a.NUMERICAS + a.CATEGORICAS
