"""Técnicas de preprocesamiento (imputadores, escaladores, encoders, binning, PCA, RFE), variables nuevas,
correlación y el comparador de make tecnicas."""

import numpy as np
import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import correlacion as k
from proyecto_ml import tecnicas as t
from proyecto_ml.features import ESCALADORES, IMPUTADORES, build_preprocessor
from proyecto_ml.generador import GeneradorAsociados


@pytest.fixture(scope="module")
def datos():
    df = GeneradorAsociados(n=400, seed=3, tasa_retiro=0.25).generar()
    return t.con_vacios(a.preparar(df), 0.15, seed=1), df[a.TARGET]


def _pre(**kw):
    return build_preprocessor(a.NUMERICAS, a.CATEGORICAS, **kw)


@pytest.mark.parametrize("estrategia", IMPUTADORES)
def test_imputadores_no_dejan_vacios(datos, estrategia):
    X, y = datos
    salida = _pre(numeric_strategy=estrategia).fit(X, y).transform(X)
    assert not np.isnan(salida).any()


def test_indicador_de_vacios_agrega_una_columna_por_variable_con_vacios(datos):
    X, y = datos
    sin, con = _pre().fit(X, y), _pre(missing_indicator=True).fit(X, y)
    assert con.transform(X).shape[1] - sin.transform(X).shape[1] == len(t.COLUMNAS_VACIOS)
    nombres = list(con.get_feature_names_out())
    assert all(f"missingindicator_{c}" in nombres for c in t.COLUMNAS_VACIOS)


@pytest.mark.parametrize("escalador", sorted(ESCALADORES))
def test_escaladores(datos, escalador):
    X, y = datos
    pre = _pre(scaler=escalador).fit(X, y)
    num = pre.transform(X)[:, : len(a.NUMERICAS)]
    if escalador == "minmax":
        assert num.min() >= -1e-9 and num.max() <= 1 + 1e-9
    if escalador == "normalizer":
        assert np.allclose(np.linalg.norm(num, axis=1)[np.linalg.norm(num, axis=1) > 0], 1)


def test_escalador_desconocido():
    with pytest.raises(ValueError, match="scaler"):
        _pre(scaler="zscore")


def test_target_encoder_una_columna_por_categorica_entre_0_y_1(datos):
    X, y = datos
    salida = _pre(encoding="target").fit(X, y).transform(X)
    assert salida.shape[1] == len(a.NUMERICAS) + len(a.CATEGORICAS)
    cat = salida[:, len(a.NUMERICAS) :]
    assert cat.min() >= 0 and cat.max() <= 1  # tasa de retiro por categoría


def test_binning_cambia_las_columnas_por_rangos_y_no_recorta_los_conteos(datos):
    X, y = datos
    pre = _pre(bins=4, bin_columns=a.COLUMNAS_RECORTE, clip_outliers="iqr", clip_columns=a.COLUMNAS_RECORTE).fit(X, y)
    ct = pre.named_steps["columns"] if hasattr(pre, "named_steps") else pre
    assert list(ct.named_transformers_["bins"].feature_names_in_) == a.COLUMNAS_RECORTE
    assert "outliers" not in ct.named_transformers_["num"].named_steps  # los conteos siguen sin recortar
    rangos = ct.named_transformers_["bins"].transform(X[a.COLUMNAS_RECORTE])
    assert rangos.shape[1] <= 4 * len(a.COLUMNAS_RECORTE) and set(np.unique(rangos)) <= {0.0, 1.0}


def test_pca_y_rfe_reducen_columnas(datos):
    X, y = datos
    completo = _pre().fit(X, y).transform(X).shape[1]
    assert _pre(pca_components=5).fit(X, y).transform(X).shape[1] == 5
    rfe = _pre(rfe_features=8).fit(X, y)
    assert rfe.transform(X).shape[1] == 8 < completo


def test_crear_variables(datos):
    X, _ = datos
    nuevo = a.CrearVariables().fit_transform(X.head(30))
    assert list(nuevo.columns[-len(a.VARIABLES_NUEVAS) :]) == a.VARIABLES_NUEVAS
    fila = nuevo.iloc[0]
    assert fila["SENALES_SALIDA"] == sum(fila[c] for c in a.SALIDA)
    assert fila["SALIDA_POR_PRODUCTO"] == pytest.approx(fila["SENALES_SALIDA"] / (1 + fila["PRODUCTOS"]))
    assert nuevo["SALARIO_LOG"].dropna().equals(np.log1p(nuevo["SALARIO"].dropna()))
    assert len(X.columns) == len(a.NUMERICAS) + len(a.CATEGORICAS)  # no modifica la entrada


# ---------------------------------------------------------------- correlación


def test_correlacion_detecta_redundancia_y_senal():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 500))
    X = pd.DataFrame(
        {
            "senal": y * 2 + rng.normal(0, 0.5, 500),
            "copia": None,
            "ruido": rng.normal(size=500),
            "cat_senal": np.where(y == 1, "a", "b"),
            "cat_ruido": rng.choice(["x", "y", "z"], 500),
        }
    )
    X["copia"] = X["senal"] * 3 + 1
    pares = k.pares_redundantes(k.matriz(X))
    assert pares.iloc[0][["variable_a", "variable_b"]].tolist() == ["senal", "copia"]
    assert "ruido" not in set(pares["variable_a"]) | set(pares["variable_b"])
    tabla = k.con_objetivo(X, y)
    assert set(tabla.head(3)["variable"]) == {"senal", "copia", "cat_senal"}
    assert tabla.set_index("variable").loc["cat_senal", "valor"] > 0.9
    assert tabla.set_index("variable").loc["cat_ruido", "valor"] < 0.1


def test_figura_matriz():
    fig = k.figura_matriz(k.matriz(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})))
    assert fig.data[0].zmin == -1 and fig.data[0].zmax == 1


# ---------------------------------------------------------------- comparador


def test_tecnicas_cubren_lo_pedido():
    nombres = {x.nombre for x in t.TECNICAS}
    assert {"SMOTE", "ADASYN", "KNN", "iterativa", "mediana + indicador", "ordinal", "target"} <= nombres
    assert {"min-max", "max-abs", "robusto", "normalizer"} <= nombres
    assert {"variables nuevas", "binning", "PCA 95 %", "RFE 15"} <= nombres
    assert len(nombres) == len(t.TECNICAS)


def test_comparar_una_fila_por_tecnica_con_diferencia_contra_la_base(datos):
    X, y = datos
    elegidas = [x for x in t.TECNICAS if x.nombre in ("base", "SMOTE", "variables nuevas", "binning")]
    tabla = t.comparar(X, y, elegidas, folds=2, repeticiones=1)
    assert list(tabla["tecnica"]) == ["base", "SMOTE", "variables nuevas", "binning"]
    assert tabla.loc[0, "dif_pr_auc"] == 0
    assert tabla["pr_auc"].between(0, 1).all() and tabla["roc_auc"].between(0, 1).all()
    assert len(t.figura(tabla).data) == tabla["grupo"].nunique()
    # diferencia con la base partición a partición: su intervalo contiene la diferencia media
    assert (tabla["dif_ic_inf"] <= tabla["dif_pr_auc"] + 1e-12).all()
    assert (tabla["dif_pr_auc"] <= tabla["dif_ic_sup"] + 1e-12).all()
    assert not tabla.loc[0, "concluyente"]  # la base contra sí misma


def test_muestreo_solo_con_el_pipeline_de_imblearn():
    smote = t.modelo(next(x for x in t.TECNICAS if x.nombre == "SMOTE"))
    assert type(smote).__module__.startswith("imblearn") and smote.named_steps["model"].class_weight is None
    base = t.modelo(t.TECNICAS[0])
    assert "sampler" not in base.named_steps and base.named_steps["model"].class_weight == "balanced"


def test_clasificador_desconocido():
    with pytest.raises(ValueError, match="clasificador"):
        t.clasificador("svm")
