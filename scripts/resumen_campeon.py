"""Resumen en Markdown del campeón guardado (models/campeon.json): lo que va en la pestaña Summary del CI.

Titular honesto: PR-AUC con su variación entre particiones y su intervalo, si la elección es concluyente, la
regla de umbrales en validación anidada, cómo le va en asociados nuevos y, sobre todo, en los escenarios con
la señal debilitada. El ROC-AUC no es el titular: con un generador de señal tan fuerte, todos pasan de 0,97.

Uso:
    uv run python scripts/resumen_campeon.py                 # imprime el Markdown
    uv run python scripts/resumen_campeon.py >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proyecto_ml import campeon as c
from proyecto_ml.config import MODELS_DIR, ORIGEN_SINTETICO, aviso_origen, es_experimental

ESCENARIO_DEBIL = "tasa 10 %, señal 25 %"


def resumen(info: dict) -> str:
    cv, h = info["metricas_cv"], info.get("horizonte_meses") or 0
    origen = info.get("origen_datos", ORIGEN_SINTETICO)  # campeones guardados antes: todos eran sintéticos
    lineas = [
        f"## Campeón: {info['titulo']}",
        f"> {'⚠️ ' if es_experimental(origen) else ''}{aviso_origen(origen)}",
        *(
            [
                "> ⚠️ **Temporalidad sin confirmar:** la evaluación no es válida hasta que el fondo confirme que los "
                "conteos se miden a t0 (TEMPORALIDAD_CONFIRMADA=si)."
            ]
            if info.get("temporalidad_confirmada") is False
            else []
        ),
        "",
        f"Foto a {h} meses · {info['filas']} asociados · {info['tasa_retiro']:.1%} se retira "
        f"({round(info['filas'] * info['tasa_retiro'])} retiros)"
        + (
            f" · datos `{info['datos_huella']}` · contrato `{info['contrato_huella']}`"
            if "datos_huella" in info
            else ""
        ),
        "",
        f"**PR-AUC {cv['pr_auc']:.3f} ± {cv['pr_auc_std']:.3f}** entre particiones"
        + (f" · IC 95 % {cv['pr_auc_ic_inf']:.3f}–{cv['pr_auc_ic_sup']:.3f}" if "pr_auc_ic_inf" in cv else ""),
    ]
    empatados = info.get("empatados") or []
    if empatados:
        nombres = ", ".join(c.TITULOS.get(n, n) for n in empatados)
        lineas.append(f"\n⚠️ **Elección no concluyente** frente a: {nombres} (la diferencia cabe en el ruido).")
    lineas += ["", "| Candidato | PR-AUC | ± particiones | campeón − candidato (IC 95 %) |", "|---|---|---|---|"]
    comparacion = info.get("comparacion") or {}
    for nombre, m in sorted(info["candidatos"].items(), key=lambda x: -x[1]["pr_auc"]):
        d = comparacion.get(nombre)
        dif = f"{d['diferencia']:+.3f} ({d['ic_inf']:+.3f} a {d['ic_sup']:+.3f})" if d else "—"
        lineas.append(f"| {c.TITULOS.get(nombre, nombre)} | {m['pr_auc']:.3f} | {m.get('pr_auc_std', 0):.3f} | {dif} |")
    hon = info.get("umbrales_honestos")
    if hon:
        lineas += [
            "",
            "**Umbrales en validación anidada** (elegidos solo con el entrenamiento de cada partición): "
            f"alerta → precisión {hon['precision']:.0%}, recall {hon['recall']:.0%}; "
            f"se retira → precisión {hon['precision_alto']:.0%} (pedida {info['precision_min']:.0%}), "
            f"recall {hon['recall_alto']:.0%}.",
        ]
    nuevos = info.get("nuevos")
    zonas = {z["zona"]: z for z in info.get("zonas_nuevos", [])}
    if nuevos and "se retira" in zonas:
        z = zonas["se retira"]
        lineas.append(
            f"\nAsociados nuevos ({nuevos['asociados']}): la zona 'se retira' acierta "
            f"{z['tasa_retiro']:.0%} ({z['se_retiran']} de {z['asociados']}); "
            f"recall de la alerta {nuevos['recall']:.0%}."
        )
    if t := info.get("validacion_temporal"):
        lineas.append(
            f"\nValidación temporal (entrena con la foto a {t['t0_entrenamiento']}, mide en la de {t['t0_prueba']}, "
            f"umbral de la foto vieja): PR-AUC {t['pr_auc']:.3f} (base {t['base_rate']:.3f}), "
            f"precisión {t['precision']:.0%}, recall {t['recall']:.0%}."
        )
    debil = [
        e for e in info.get("escenarios", []) if e["modelo"] == info["modelo"] and e["escenario"] == ESCENARIO_DEBIL
    ]
    if debil:
        e = debil[0]
        lineas.append(
            f"\n**La cifra honesta: si la señal real es la cuarta parte** ({ESCENARIO_DEBIL}): "
            f"PR-AUC {e['pr_auc']:.2f}, precisión {e['precision']:.2f}, recall {e['recall']:.2f}."
        )
    return "\n".join(lineas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--carpeta", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    print(resumen(json.loads((args.carpeta / c.ARCHIVO_INFO).read_text(encoding="utf-8"))), end="")


if __name__ == "__main__":
    main()
