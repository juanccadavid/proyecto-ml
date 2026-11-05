"""Validación fuera de tiempo con extractos de fechas distintas (make fuera-de-tiempo).

Necesita 3 o más extractos del fondo sacados en fechas distintas, anonimizados con la MISMA ANON_SALT (los
seudónimos deben coincidir) y con sus conteos medidos a la fecha de cada extracto. Requisitos completos en
docs/datos_reales.md. Con un solo extracto no existe esta validación: ``make campeon`` solo puede hacer la prueba
de estrés temporal (las mismas personas en las dos fotos).

Cada archivo debe cumplir su contrato: con ORIGEN_DATOS=real_anonimizado, el del dataset anonimizado
(contrato.para_ml); si no, el de origen. Los errores no muestran valores personales.

Uso (fecha = la de extracción de cada archivo):
    uv run python scripts/validacion_fuera_de_tiempo.py \\
        --extracto data/raw/anon_2026-04.parquet:2026-04-30 \\
        --extracto data/raw/anon_2026-07.parquet:2026-07-31 \\
        --extracto data/raw/anon_2026-10.parquet:2026-10-31

Guarda reports/metrics/fuera_de_tiempo_<fecha>.json (agregados: ninguna fila de asociados).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import contrato as k
from proyecto_ml import extractos as ex
from proyecto_ml.config import METRICS_DIR, ORIGEN_REAL, aviso_origen, origen_datos, temporalidad_confirmada
from proyecto_ml.datos_campeon import incumplimientos


def _extracto(valor: str) -> tuple[Path, str]:
    ruta, _, fecha = valor.rpartition(":")
    if not ruta or not fecha:
        raise argparse.ArgumentTypeError(f"usa RUTA:AAAA-MM-DD, no {valor!r}")
    # Solo AAAA-MM-DD: "01/02/2026" se leería en silencio como 2 de enero (¿o era 1 de febrero?) y movería la foto.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha):
        raise argparse.ArgumentTypeError(f"fecha {fecha!r}: usa AAAA-MM-DD (ej. 2026-04-30)")
    try:
        date.fromisoformat(fecha)
    except ValueError:
        raise argparse.ArgumentTypeError(f"fecha {fecha!r} no existe en el calendario") from None
    return Path(ruta), fecha


def cargar(ruta: Path, fecha: str, origen: str) -> ex.Extracto:
    """Lee y valida un extracto contra su contrato, o ``SystemExit`` con un mensaje sin valores personales."""
    df = a.cargar(ruta)
    real = origen == ORIGEN_REAL
    if errores := k.validar(df, k.para_ml() if real else k.cargar()):
        raise SystemExit(str(incumplimientos(ruta, errores, real)))
    return ex.Extracto.de(df, fecha, ruta.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--extracto", type=_extracto, action="append", default=[], help="RUTA:AAAA-MM-DD (3 o más)")
    parser.add_argument("--horizonte", type=int, default=a.HORIZONTE_MESES, help="Meses del objetivo")
    parser.add_argument("--folds", type=int, default=c.FOLDS)
    parser.add_argument("--semilla", type=int, default=42, help="De los modelos (bosques, boosting)")
    args = parser.parse_args(argv)

    origen = origen_datos()
    print(aviso_origen(origen))
    if len(args.extracto) < 3:
        raise SystemExit(
            f"Hacen falta al menos 3 extractos de fechas distintas (diste {len(args.extracto)}). Con uno solo, la "
            "única evaluación temporal posible es la prueba de estrés de make campeon."
        )
    extractos = [cargar(ruta, fecha, origen) for ruta, fecha in args.extracto]
    try:
        primera = ex.foto_entre_extractos(*sorted(extractos, key=lambda e: e.fecha)[:2], args.horizonte)
        resultado = c.validacion_fuera_de_tiempo(
            c.lista_candidatos(a.separar(primera)[0], random_state=args.semilla),
            extractos,
            args.horizonte,
            folds=args.folds,
        )
    except ValueError as error:  # pares que no cumplen los requisitos (revisar_par) o pocas clases
        raise SystemExit(f"No se pudo validar: {error}") from None

    print(f"Extractos: {', '.join(resultado['extractos'])} · horizonte {args.horizonte} meses")
    print(
        f"Entrena y elige con t0 = {', '.join(resultado['t0_entrenamiento'])} "
        f"({resultado['asociados_entrenamiento']} asociados, {resultado['retiros_entrenamiento']} retiros); "
        f"mide en t0 = {resultado['t0_prueba']} ({resultado['asociados_prueba']}, {resultado['retiros_prueba']})"
    )
    print(f"Elegido: {c.TITULOS.get(resultado['elegido'], resultado['elegido'])} · selección: {resultado['seleccion']}")
    print(
        f"PR-AUC {resultado['pr_auc']:.3f} (base {resultado['base_rate']:.3f}) · ROC-AUC {resultado['roc_auc']:.3f} · "
        f"al umbral {resultado['umbral']:.3f}: precisión {resultado['precision']:.2f}, recall {resultado['recall']:.2f}"
    )
    if origen == ORIGEN_REAL and not temporalidad_confirmada():
        print("OJO: TEMPORALIDAD_CONFIRMADA no es 'si': confirma que cada conteo se midió a la fecha de su extracto.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    salida = METRICS_DIR / f"fuera_de_tiempo_{datetime.now():%Y%m%d_%H%M%S}.json"
    info = {"origen_datos": origen, "horizonte_meses": args.horizonte, "semilla": args.semilla, **resultado}
    salida.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    print(f"Resultado -> {salida}")


if __name__ == "__main__":
    main()
