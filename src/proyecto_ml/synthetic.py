"""Datos SINTÉTICOS para probar el pipeline de punta a punta y para demos públicas.

No representan a ninguna persona ni organización real. Sirven mientras se define el
dataset definitivo y para publicar la demo sin exponer datos reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n_rows: int = 5000, random_state: int = 42, base_rate: float = 0.06) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    antiguedad = rng.integers(1, 240, n_rows)
    salario = rng.lognormal(mean=15.0, sigma=0.5, size=n_rows)
    aporte_pct = rng.uniform(0.02, 0.10, n_rows)
    saldo_ahorro = antiguedad * salario * aporte_pct * rng.uniform(0.6, 1.0, n_rows)
    tiene_credito = rng.random(n_rows) < 0.55
    saldo_credito = np.where(tiene_credito, saldo_ahorro * rng.uniform(0.2, 2.5, n_rows), 0.0)
    creditos_negados = rng.poisson(0.3, n_rows)
    servicios = rng.integers(0, 6, n_rows)
    liquidez_neta = saldo_ahorro - saldo_credito

    logit = (
        -1.2
        + 1.1 * (liquidez_neta / salario > 3)
        + 0.9 * creditos_negados
        - 0.35 * servicios
        - 0.004 * antiguedad
        + rng.normal(0, 0.8, n_rows)
    )
    prob = 1 / (1 + np.exp(-logit))
    threshold = np.quantile(prob, 1 - base_rate)
    target = (prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "id_entidad": [f"S{i:06d}" for i in range(n_rows)],
            "periodo": pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n_rows), unit="D"),
            "antiguedad_meses": antiguedad,
            "salario": salario.round(0),
            "aporte_pct_salario": aporte_pct.round(4),
            "saldo_ahorro": saldo_ahorro.round(0),
            "saldo_credito": saldo_credito.round(0),
            "creditos_negados_6m": creditos_negados,
            "servicios_usados": servicios,
            "tipo_contrato": rng.choice(["indefinido", "fijo", "obra_labor"], n_rows, p=[0.7, 0.2, 0.1]),
            "sede": rng.choice(["sede_a", "sede_b", "sede_c", "sede_d"], n_rows),
            "target": target,
        }
    )
    # Faltantes controlados para ejercitar la imputación
    for col in ["salario", "tipo_contrato"]:
        df.loc[rng.random(n_rows) < 0.03, col] = np.nan
    return df
