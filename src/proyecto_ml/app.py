"""Frontend de demostración con Streamlit.

Uso:  uv run streamlit run src/proyecto_ml/app.py
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from proyecto_ml.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "best_model.json"

st.set_page_config(page_title="Proyecto I - Demo", layout="centered")
st.title("Demo del modelo predictivo")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("No hay modelo entrenado. Ejecuta primero: `make train` (o `make demo-data && make train`).")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text(encoding="utf-8"))


model, meta = load_model()
st.caption(f"Modelo: **{meta['model']}** · Objetivo: `{meta['target']}`")

tab_single, tab_batch = st.tabs(["Caso individual", "Archivo (lote)"])

with tab_single:
    with st.form("caso"):
        values: dict[str, object] = {}
        for col in meta["numeric"]:
            values[col] = st.number_input(col, value=meta["numeric_defaults"][col])
        for col in meta["categorical"]:
            values[col] = st.selectbox(col, meta["categories"][col])
        submitted = st.form_submit_button("Calcular probabilidad")
    if submitted:
        prob = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
        st.metric("Probabilidad estimada", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

with tab_batch:
    st.info("Usa solo datos anonimizados o sintéticos en esta demo.")
    uploaded = st.file_uploader("CSV con las mismas columnas del entrenamiento", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in meta["numeric"] + meta["categorical"] if c not in batch.columns]
        if missing:
            st.error(f"Faltan columnas: {missing}")
        else:
            batch["probabilidad"] = model.predict_proba(batch)[:, 1]
            st.dataframe(batch.sort_values("probabilidad", ascending=False), use_container_width=True)
