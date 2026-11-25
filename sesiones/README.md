# sesiones/

Notebooks de exploración y bitácora del equipo.

- Nombre de notebooks: `NN_autor_tema.ipynb` (ej. `04_autor_tema.ipynb`).
- Las salidas se borran automáticamente al hacer commit (nbstripout): pueden contener filas de datos.
- Lógica reutilizable **no** se queda en notebooks: se mueve a `src/proyecto_ml/` con su prueba.
- `bitacora.md`: registro corto de cada sesión (fecha, quién, qué se decidió, pendientes).

## Notebooks

| Notebook | Qué responde |
|---|---|
| `01_plantilla_eda.ipynb` | Plantilla de EDA; necesita un dataset propio en `data/processed/dataset.parquet` |
| `02_juan_eda_foto_asociado.ipynb` | La foto del asociado: población, etiqueta sintética y diferencias por clase |
| `03_sebastian_feature_engineering.ipynb` | Si las variables nuevas mejoran el modelo, de qué variables depende y cuánto infla la fuga las métricas |

El 02 y el 03 usan los 5.000 asociados sintéticos de desarrollo y se ejecutan completos en el CI (`make notebooks`).
