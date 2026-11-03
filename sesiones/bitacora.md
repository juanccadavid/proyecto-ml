# Bitácora del proyecto

| Fecha | Quién | Qué se hizo / decidió | Pendientes |
|---|---|---|---|
| 2026-10-24 | Juan | Estructura base del repositorio (pipeline, MLflow, Streamlit, pruebas, seguridad). Dataset por definir. | Definir dataset y obtener autorización de uso antes del 26-oct |
| 2026-10-27 | Sebastián | Configuré mi entorno local (Git, uv, VS Code, GitHub CLI); 17 pruebas pasan. | Instalar Docker y make |
| 2026-11-02 | Juan | Foto del asociado: t0 = corte − 3 meses, población = activos en t0, objetivo = se retira entre t0 y el corte (`abandono.foto`, `--horizonte`). Con 50 filas queda 1 retiro: se entrena con 2.000 generados. EDA en `02_juan_eda_foto_asociado.ipynb`. | Preguntar a Edwin: ¿los conteos de solicitudes tienen fecha o son a la extracción? Tasa de retiro anual real, % con solicitudes de cancelación/disminución, mediana de antigüedad y salario |

## Hitos del curso
- **Lun 26-oct**: equipo (3-4) + dataset y su fuente.
- **Mié 11-nov**: EDA + feature engineering (10 min), repo en GitHub con Pull Requests.
- **Mié 25-nov**: repo organizado, resultados del modelo y demo (10-15 min).

## Decisiones pendientes al definir el dataset
- Hugging Face con propósito real: TabPFN como tercer modelo (≤ ~50k filas, licencia 2.5 no comercial, GPU).
- Modelos de texto locales (zero-shot / NER) solo si hay texto libre (ej. motivo de retiro).
- No publicar la demo con datos reales en Hugging Face Spaces; usar datos sintéticos para cualquier demo pública.
