# reports/

Resultados que SÍ se versionan (livianos y sin datos personales):

- `metrics/`: JSON de métricas de las corridas que se lanzan a mano: `make train` (`metrics_<fecha>.json`),
  `abandono`, `afinamiento`, `optuna`, `pycaret` y `tecnicas`. Sirven para que el equipo compare resultados.
- `figures/`: gráficos exportados del EDA y de resultados (PNG/SVG). Los HTML de Plotly no se suben (pesan).
- `perfil_datos.json`: perfil de calidad generado por `scripts/ingesta.py`.

Lo que NO se versiona (está en `.gitignore`):

- Los JSON que `make pipeline` regenera en cada corrida, local y en el CI: `campeon_*`, `regresion_*`,
  `segmentos_*` y `validacion_cruzada_*`. Son reproducibles (semillas fijas) y ya quedan en MLflow y en el
  artefacto `resultados` de cada corrida del CI (14 días). Versionarlos sumaría 4 archivos casi iguales por
  corrida. Los que ya estaban en Git siguen ahí. Para fijar uno a propósito (p. ej. el resultado de un hito):
  `git add -f reports/metrics/campeon_<fecha>.json`.
- Los HTML de `figures/`, los modelos (`models/`) y los experimentos (`mlflow.db`, `mlruns/`).

Revisar que ninguna figura muestre registros individuales identificables.
