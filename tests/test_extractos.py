"""Infraestructura para la validación fuera de tiempo con varios extractos (extractos.py y
campeon.validacion_fuera_de_tiempo).

Los "extractos" de estas pruebas se arman recortando un solo archivo generado a distintas fechas: sirven para
probar la interfaz (quién entra, de dónde sale la etiqueta, qué se rechaza), NO para medir nada. Sus conteos no
cambian entre fechas, justo lo contrario de un extracto real.
"""

import importlib.util
import json

import pandas as pd
import pytest

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import config
from proyecto_ml import extractos as ex
from proyecto_ml.generador import GeneradorAsociados

FECHAS = ("2026-01-31", "2026-04-30", "2026-07-31")


@pytest.fixture(scope="module")
def base():
    return GeneradorAsociados(n=3000, seed=5, tasa_retiro=0.3).generar()


def recortar(df: pd.DataFrame, fecha: str) -> pd.DataFrame:
    """Cómo se vería el extracto sacado en ``fecha``: sin retiros posteriores, sin quien no había ingresado."""
    fecha = pd.Timestamp(fecha)
    vista = df[pd.to_datetime(df["FECHAINGRESOLABORA"]) <= fecha].copy()
    despues = pd.to_datetime(vista["FECHARETIRO"]) > fecha
    vista.loc[despues, ["FECHARETIRO"]] = pd.NaT
    vista.loc[despues, "#RETIROS"] = 0
    vista.loc[despues, ["ESTADO", "ESTADO_LABORAL", "MOTIVORETIRO"]] = ["ACTIVO", "VIGENTE", ""]
    for columna in ("FECHAULTACTUALIZACION", "ULTIMAACTUALIZACION"):
        vista.loc[despues, columna] = vista.loc[despues, columna].where(
            pd.to_datetime(vista.loc[despues, columna]) <= fecha, fecha
        )
    return vista


@pytest.fixture(scope="module")
def extractos(base):
    return [ex.Extracto.de(recortar(base, f), f) for f in FECHAS]


def test_la_etiqueta_sale_del_extracto_siguiente(base, extractos):
    viejo, nuevo = extractos[0], extractos[1]
    foto = ex.foto_entre_extractos(viejo, nuevo, horizonte_meses=3)
    t0, fin = viejo.fecha, viejo.fecha + pd.DateOffset(months=3)
    retiro = pd.to_datetime(base.set_index("CEDULA").loc[foto["CEDULA"], "FECHARETIRO"]).to_numpy()
    esperado = ((retiro > t0) & (retiro <= fin)).astype(int)
    assert (foto[a.TARGET].to_numpy() == esperado).all() and foto[a.TARGET].sum() > 0
    assert (foto[a.FECHA_OBSERVACION] == t0).all()  # EDAD y ANTIGUEDAD a la fecha del extracto viejo
    assert pd.to_datetime(foto["FECHARETIRO"]).isna().all()  # en el viejo, todos activos: no ven el retiro
    assert len(foto) == len(ex.activos(viejo)) and foto.attrs["sin_seguimiento"] == 0


def test_revisar_par_rechaza_lo_que_no_se_puede_cruzar(extractos):
    viejo, nuevo, _ = extractos
    temprano = ex.Extracto.de(nuevo.datos, "2026-03-15")
    assert any("no cubre el horizonte" in p for p in ex.revisar_par(viejo, temprano, 3))
    otra_sal = ex.Extracto.de(nuevo.datos.assign(CEDULA=nuevo.datos["CEDULA"] + 10**9), nuevo.fecha)
    assert any("ANON_SALT" in p for p in ex.revisar_par(viejo, otra_sal, 3))
    sin_llave = ex.Extracto.de(nuevo.datos.drop(columns="CEDULA"), nuevo.fecha)
    assert any("no comparten ninguna llave" in p for p in ex.revisar_par(viejo, sin_llave, 3))
    futuro = viejo.datos.copy()
    futuro.loc[futuro.index[0], "FECHARETIRO"] = pd.Timestamp("2026-12-31")
    assert any("posteriores a la fecha" in p for p in ex.revisar_par(ex.Extracto.de(futuro, viejo.fecha), nuevo, 3))
    assert ex.revisar_par(viejo, nuevo, 3) == []
    with pytest.raises(ValueError, match="no cubre el horizonte"):
        ex.foto_entre_extractos(viejo, temprano, 3)


def test_fuera_de_tiempo_elige_solo_con_los_pares_viejos(monkeypatch, extractos):
    modelos = a.modelos_supervisados()
    candidatos = {"logreg": modelos["logreg"], "ridge": modelos["ridge"]}
    with pytest.raises(ValueError, match="al menos 3 extractos"):
        c.validacion_fuera_de_tiempo(candidatos, extractos[:2])
    filas = []
    original = c.puntajes_fuera_de_particion

    def espia(modelo, X, y, *args, **kw):
        filas.append(len(X))
        return original(modelo, X, y, *args, **kw)

    monkeypatch.setattr(c, "puntajes_fuera_de_particion", espia)
    resultado = c.validacion_fuera_de_tiempo(candidatos, list(reversed(extractos)), horizonte_meses=3, folds=3)
    assert resultado["t0_entrenamiento"] == [FECHAS[0]] and resultado["t0_prueba"] == FECHAS[1]  # ordena por fecha
    assert filas == [resultado["asociados_entrenamiento"]] * 2  # la elección no ve el par de prueba
    assert resultado["elegido"] in candidatos and 0 <= resultado["pr_auc"] <= 1


def test_script_pide_tres_extractos_y_guarda_solo_agregados(tmp_path, monkeypatch, base):
    spec = importlib.util.spec_from_file_location(
        "fuera_de_tiempo", config.ROOT_DIR / "scripts" / "validacion_fuera_de_tiempo.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("ORIGEN_DATOS", "sintetico")
    monkeypatch.setattr(script, "METRICS_DIR", tmp_path / "metrics")
    rutas = []
    for fecha in FECHAS:
        ruta = tmp_path / f"extracto_{fecha}.parquet"
        recortar(base, fecha).to_parquet(ruta)
        rutas.append(f"{ruta}:{fecha}")
    with pytest.raises(SystemExit, match="al menos 3"):
        script.main(["--extracto", rutas[0], "--extracto", rutas[1]])
    script.main([arg for r in rutas for arg in ("--extracto", r)] + ["--folds", "3"])
    (salida,) = (tmp_path / "metrics").glob("fuera_de_tiempo_*.json")
    info = json.loads(salida.read_text(encoding="utf-8"))
    assert info["origen_datos"] == "sintetico" and info["t0_prueba"] == FECHAS[1]
    texto = salida.read_text(encoding="utf-8")
    assert not any(str(v) in texto for v in base["CEDULA"].head(50))  # ninguna fila de asociados
