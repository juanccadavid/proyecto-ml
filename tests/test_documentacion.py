"""La metodología y sus evidencias: cada cifra citada debe estar en el JSON que cita, y ese JSON en Git.

Si alguien vuelve a correr un experimento y cambia la cita sin actualizar el texto (o transcribe mal una cifra), estas
pruebas fallan. No vuelven a correr los experimentos: comprueban que el documento dice lo que dicen sus evidencias.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOC = (RAIZ / "docs" / "metodologia.md").read_text(encoding="utf-8")
CITADOS = sorted(set(re.findall(r"reports/metrics/([a-z_]+_\d{8}_\d{6}\.json)", DOC)))


def cifra(valor: float, decimales: int = 3) -> str:
    """Como se escribe en el documento: coma decimal (0,568)."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def cargar(tipo: str) -> dict:
    """El JSON citado más reciente de ``tipo`` (el de Optuna de antes de la corrección también se cita)."""
    nombre = max(n for n in CITADOS if n.startswith(f"{tipo}_"))
    return json.loads((RAIZ / "reports" / "metrics" / nombre).read_text(encoding="utf-8"))


def faltan(valores) -> list[str]:
    return [cifra(v) for v in valores if cifra(v) not in DOC]


def test_los_json_citados_estan_en_git():
    try:
        salida = subprocess.run(
            ["git", "ls-files", "reports/metrics"], cwd=RAIZ, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("sin git: no se puede saber qué está versionado")
    versionados = set(salida.split())
    assert {"campeon", "tecnicas", "optuna", "afinamiento", "pycaret", "abandono", "regresion"} <= {
        n.split("_")[0] for n in CITADOS
    }
    assert [n for n in CITADOS if f"reports/metrics/{n}" not in versionados] == []


def test_cifras_del_campeon_coinciden_con_su_json():
    info = cargar("campeon")
    cv = info["metricas_cv"]
    valores = [cv[k] for k in ("pr_auc", "pr_auc_ic_inf", "pr_auc_ic_sup", "pr_auc_std", "precision", "recall")]
    valores += [info["umbral"], info["umbral_alto"]]
    valores += [info["umbrales_honestos"][k] for k in ("precision", "recall", "precision_alto", "recall_alto")]
    valores += [info["nuevos"][k] for k in ("pr_auc", "roc_auc", "precision", "recall")]
    for escenario in (e for e in info["escenarios"] if e["modelo"] == info["modelo"]):
        valores += [escenario["pr_auc"], escenario["precision"], escenario["recall"]]
    for fila in info["prevalencia"]:
        valores += [fila["precision_alerta"], fila["precision_alto"], fila["umbral_alto_recalculado"]]
    comprobacion = info["prevalencia_comprobacion"]
    valores += [comprobacion["precision_esperada"], comprobacion["precision_medida"]]
    valores += [v for v in info["shap_nuevos"].values() if v >= 0.01]  # la tabla de la lógica sintética
    assert faltan(valores) == []
    assert f"{info['tasa_retiro'] * 100:.1f} %".replace(".", ",") in DOC
    assert f"{round(info['tasa_retiro'] * info['filas'])} eventos" in DOC


def test_cifras_de_optuna_afinar_y_pycaret_coinciden_con_sus_json():
    optuna = cargar("optuna")
    valores = [optuna["comparativa"][m]["AUC"] for m in ("hist_gb_optuna", "random_forest_optuna", "hist_gb")]
    nuevos = optuna["nuevos (n=1000)"]
    valores += [nuevos[m][k] for m in nuevos for k in ("roc_auc", "pr_auc")]
    valores += [nuevos[m][k] for m in ("random_forest", "random_forest_optuna") for k in ("precision", "recall")]
    valores += [nuevos["hist_gb_optuna"]["precision"], nuevos["hist_gb_optuna"]["recall"]]

    afinar = cargar("afinamiento")
    nuevos = afinar["nuevos (n=1000)"]
    valores += [nuevos[m]["roc_auc"] for m in nuevos] + [nuevos["ridge"]["recall"], nuevos["ridge_afinado"]["recall"]]
    for pipeline in ("LogisticRegression_median_onehot_iqr", "Ridge_median_ordinal_iqr"):
        valores += [afinar["preprocesamientos"][pipeline][k] for k in ("AUC_mean", "AUC_std", "AUC_nuevos")]

    pycaret = cargar("pycaret")
    valores += [pycaret["comparativa"][m]["AUC"] for m in ("pycaret_gbc", "pycaret_ada")]
    valores += [pycaret["nuevos (n=1000)"][m]["pr_auc"] for m in ("pycaret_lightgbm", "pycaret_gbc", "hist_gb")]
    assert faltan(valores) == []


def test_importancias_y_regresion_coinciden_con_sus_json():
    abandono = cargar("abandono")
    mdi = {f["variable"].split("__")[-1]: f["importancia"] for f in abandono["importancia_rf"]}
    permutacion = {f["variable"]: f["importancia"] for f in abandono["importancia"] if f["modelo"] == "random_forest"}
    continuas = [mdi[v] for v in ("EDAD", "ANTIGUEDAD_F_MESES", "ANTIGUEDAD_LA_MESES")]
    assert f"{cifra(min(continuas))}–{cifra(max(continuas))}" in DOC
    assert faltan([mdi["SALARIO"], permutacion["SALARIO"]]) == []

    intervalo = cargar("regresion")["intervalo"]
    assert f"{intervalo['cobertura'] * 100:.1f} %".replace(".", ",") in DOC
    assert all(cifra(intervalo[k], 2) in DOC for k in ("factor_min", "factor_max"))


def test_sensibilidad_de_las_tecnicas_a_la_semilla_coincide_con_sus_json():
    """Semillas 7/42/2026 de los datos de desarrollo: un JSON por semilla, en ese orden (por fecha)."""
    nombres = [n for n in CITADOS if n.startswith("tecnicas_")]
    assert len(nombres) == 3
    pr_auc: dict[str, list[float]] = {}
    for nombre in nombres:
        datos = json.loads((RAIZ / "reports" / "metrics" / nombre).read_text(encoding="utf-8"))
        for fila in datos["tecnicas"]:
            pr_auc.setdefault(fila["tecnica"], []).append(fila["pr_auc"])
    for tecnica in ("base", "sin balancear", "target", "normalizer"):
        assert "/".join(cifra(v) for v in pr_auc[tecnica]) in DOC, tecnica


def test_antes_y_despues_de_la_prevalencia_de_los_nuevos():
    """Mismos modelos, nuevos al 30 % (antes) y al 10 % (después): la PR-AUC cambia por la prevalencia."""
    antes, despues = sorted(n for n in CITADOS if n.startswith("optuna_"))
    for nombre in (antes, despues):
        nuevos = json.loads((RAIZ / "reports" / "metrics" / nombre).read_text(encoding="utf-8"))["nuevos (n=1000)"]
        assert faltan([nuevos["random_forest"]["pr_auc"], nuevos["hist_gb"]["pr_auc"]]) == []


def test_el_verificador_de_notebooks_detecta_cifras_viejas():
    """scripts/verificar_notebooks.py: una cifra del texto que no sale de las salidas se marca; el redondeo, los
    porcentajes y el signo ("resta 0,052" frente a -0.052) no."""
    spec = importlib.util.spec_from_file_location("verificar_notebooks", RAIZ / "scripts" / "verificar_notebooks.py")
    assert spec and spec.loader
    verificador = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verificador)
    texto = "PR-AUC 0,542; quitarla resta 0,052; la tasa sube a 6,4 %; antes decía 0,492."
    salida = nbformat.v4.new_output("stream", text="pr_auc 0.5418  dif -0.052  tasa 0.064")
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(texto), nbformat.v4.new_code_cell("tabla", outputs=[salida])]
    )
    assert verificador.cifras_sin_respaldo(nb, verificador.numeros_de_salidas(nb)) == [(0, "0,492")]
