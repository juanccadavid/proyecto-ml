# Bitácora del proyecto

| Fecha | Quién | Qué se hizo / decidió | Pendientes |
|---|---|---|---|
| 2026-10-24 | Juan | Estructura base del repositorio (pipeline, MLflow, Streamlit, pruebas, seguridad). Dataset por definir. | Definir dataset y obtener autorización de uso antes del 26-oct |
| 2026-10-27 | Sebastián | Configuré mi entorno local (Git, uv, VS Code, GitHub CLI); 17 pruebas pasan. | Instalar Docker y make |
| 2026-11-02 | Juan | Foto del asociado: t0 = corte − 3 meses, población = activos en t0, objetivo = se retira entre t0 y el corte (`abandono.foto`, `--horizonte`). Con 50 filas queda 1 retiro: se entrena con 2.000 generados. EDA en `02_juan_eda_foto_asociado.ipynb`. | Preguntar a Edwin: ¿los conteos de solicitudes tienen fecha o son a la extracción? Tasa de retiro anual real, % con solicitudes de cancelación/disminución, mediana de antigüedad y salario |
| 2026-11-21 | Edwin | Documentó las respuestas del fondo sobre temporalidad, disponibilidad de extractos, campos de retiro, publicación del esquema y cuasi-identificadores, sin publicar cifras no autorizadas. | Validar el extracto anonimizado; resolver la ausencia de etiquetas y de extractos históricos; obtener la tasa anual si el fondo la calcula; definir el esquema autorizado y confirmar que el repositorio sea privado antes de subir el PR. |
| 2026-11-23 | Edwin | Definió con el equipo el esquema anonimizado de 44 columnas y confirmó que `#RETIROS` es un conteo acumulado durante la permanencia, no una etiqueta de los últimos tres meses. Se separó la etiqueta sintética y se bloqueó el entrenamiento real sin seguimiento futuro. | Corregir los incumplimientos de calidad del contrato y definir cómo obtener una etiqueta futura válida. |

## Hitos del curso
- **Lun 26-oct**: equipo (3-4) + dataset y su fuente.
- **Mié 11-nov**: EDA + feature engineering (10 min), repo en GitHub con Pull Requests.
- **Mié 25-nov**: repo organizado, resultados del modelo y demo (10-15 min).

## Decisiones pendientes al definir el dataset
- Hugging Face con propósito real: TabPFN como tercer modelo (≤ ~50k filas, licencia 2.5 no comercial, GPU).
- Modelos de texto locales (zero-shot / NER) solo si hay texto libre (ej. motivo de retiro).
- No publicar la demo con datos reales en Hugging Face Spaces; usar datos sintéticos para cualquier demo pública.
