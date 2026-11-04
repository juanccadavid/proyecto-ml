# Licencia y distribución (propuesta, pendiente de aprobación)

> Parte de la documentación de [proyecto-ml](../README.md).

**Estado: el repositorio NO tiene licencia.** Sin licencia, todos los derechos quedan reservados a sus autores: se
puede ver en GitHub, pero no reutilizar. La propuesta está en [`LICENSE.propuesta`](../LICENSE.propuesta) (MIT) y
**requiere aprobación** antes de adoptarse:

1. **Todos los autores del equipo** (el código tiene varios autores: la licencia la otorgan todos).
2. **El reglamento de la universidad** sobre propiedad intelectual de trabajos del curso: algunas instituciones
   reservan derechos sobre lo producido en sus programas.
3. **El fondo de empleados**, si el repositorio llega a incluir algo derivado de su información (hoy no incluye
   nada: ver "Datos").

Para adoptarla: completar el titular, renombrar el archivo a `LICENSE` y quitar su encabezado de propuesta.

## Qué cubriría la licencia

| Material | ¿Lo cubre MIT? | Nota |
|---|---|---|
| Código propio (`src/`, `scripts/`, `tests/`, `web/`, notebooks) | **Sí** | Es lo que se licencia |
| Documentación (`README.md`, `docs/`) | Sí (como parte del repositorio) | |
| Contrato de datos (`contratos/asociados.yaml`) | Sí | Describe el esquema; no contiene datos de personas |
| Datos sintéticos (`GeneradorAsociados`) y los 50 ficticios de `data/ejemplo/` | Sí | Generados por el propio código; ninguna persona real |
| **Extracto real del fondo** | **No** | Nunca está en el repositorio (`.gitignore`, pre-commit, contrato). Es del fondo y tiene datos personales: no se puede licenciar |
| Material de la universidad (enunciados, plantillas del curso) | **No** | Sus derechos son de la universidad; no se relicencia |
| Dependencias de terceros | **No** | Cada una conserva su licencia (abajo); el repositorio no las incluye, se instalan con uv |

## Dependencias (verificado en los metadatos de las versiones instaladas, `uv.lock`)

| Paquete | Licencia | ¿Compatible con MIT para el código propio? |
|---|---|---|
| pandas, numpy, scikit-learn, joblib, python-dotenv | BSD-3-Clause (numpy incluye además 0BSD, MIT, Zlib, CC0) | Sí |
| pyarrow, streamlit, mlflow | Apache-2.0 | Sí |
| plotly, openpyxl, imbalanced-learn, pandera, polars, optuna, shap, lightgbm | MIT | Sí |
| matplotlib | Licencia de matplotlib (basada en la PSF) | Sí |
| **pycaret 4.0.0a8** (extra opcional `pycaret`) | **FSL-1.1-MIT** (Functional Source License) | El código propio puede ser MIT porque PyCaret **no se redistribuye**. Pero la FSL **no es de código abierto**: prohíbe usos que compitan con el producto y pasa a MIT dos años después de cada versión. Quien instale el extra queda sujeto a ella. Si se quiere un proyecto 100 % abierto, sacar el extra o fijar una versión anterior con licencia MIT |
| tabpfn, transformers, sentence-transformers (extra opcional `hf`) | **Sin verificar** (no están instalados aquí) | Revisar la licencia de TabPFN antes de usar el extra: tiene condiciones propias de su autor |

Las dependencias se instalan, no se copian al repositorio: MIT para el código propio no cambia sus licencias.

## Por qué MIT

- Es corta, conocida y permite que evaluadores y empleadores reutilicen el código con solo mantener el aviso.
- Es compatible con todas las licencias permisivas de las dependencias principales.
- Alternativa: **Apache-2.0**, si el equipo quiere una cláusula explícita de patentes y de marcas. Para un
  proyecto académico sin patentes, MIT es suficiente.
