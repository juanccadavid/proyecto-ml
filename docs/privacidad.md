# Seguridad de datos, privacidad y fecha de corte

> Parte de la documentación de [proyecto-ml](../README.md). Todas las cifras son de **datos sintéticos**.

- `data/`, `models/`, `mlflow.db`, `.env` y `*.bundle` están en `.gitignore`.
- Pre-commit bloquea archivos de datos, borra salidas de notebooks (nbstripout) y escanea secretos (Gitleaks).
- Cualquier demo pública se hace con datos sintéticos.

## Privacidad: dos contratos y la anonimización

| Etapa | Contrato | Quién / dónde |
|---|---|---|
| Extracto original (con datos personales) | **de origen**: `contratos/asociados.yaml` (`contrato.cargar`); también los datos sintéticos y Supabase | el custodio, dentro de su organización |
| Anonimización | el de origen dice qué es personal: `pii: eliminar` (16 columnas: nombres, dirección, teléfonos, correo, cuenta...) o `pii: seudonimizar` (`CEDULA`, `IDASOCIADOS`) | el custodio: `make anonimizar DATA=extracto.xlsx SALIDA=data/raw/extracto_anon.parquet` (Excel lee como número los códigos con dígitos y les quita los ceros a la izquierda: para validar el extracto *original* con `make contrato`, mejor CSV o Parquet; las columnas afectadas, teléfonos y cuenta, se eliminan al anonimizar) |
| Dataset anonimizado para ML | **para ML**: `contrato.para_ml()` = el de origen sin las columnas `eliminar`, con `CEDULA_SEUDO` e `IDASOCIADOS_SEUDO` y sin la regla que usa nombres | el equipo: `make contrato-ml DATA=...` y `make campeon` con `ORIGEN_DATOS=real_anonimizado` |

- **El contrato para ML no relaja nada del modelo:** las variables, el objetivo, sus tipos y rangos, la
  distribución y las reglas que no tocan datos personales siguen exigidos. Es estricto: si en el archivo quedó
  una columna personal (p. ej. `NOMBRES`), lo rechaza. Las columnas personales tampoco pueden entrar al modelo:
  las variables son una lista fija (`abandono.NUMERICAS` + `CATEGORICAS`).
- **`scripts/anonimizar.py` no deja salir PII por un descuido:**
  - toma la configuración del contrato (`pii`), y una columna que el contrato no conoce lo detiene hasta que se
    decida `--conservar`, `--eliminar` o `--seudonimizar`;
  - exige `ANON_SALT` (`make salt`);
  - **antes de escribir**, verifica que no quede PII (abajo) **y** valida la salida contra el contrato para ML;
  - si algo falla, **no escribe nada ni reemplaza la salida que ya existía** (escribe a un temporal y lo renombra
    solo al final);
  - los errores dicen columnas, reglas y cantidades, nunca valores;
  - como el contrato para ML es estricto, una columna que se conserva fuera del contrato (`--conservar`) o en
    rangos (`--bin`) no pasa: primero se agrega al contrato, con revisión del equipo;
  - antes aplica `contrato.limpiar_extracto` (`--sin-limpiar` lo apaga).
- **PII residual (`anonymize.verificar`, configurable con `anonymize.Deteccion`).** Busca en la salida los valores
  de las columnas personales **de ese extracto** (no "cualquier número", que daría falsos positivos en masa):
  - **exacto:** una celda igual a un valor personal (6 caracteres o más);
  - **número incrustado:** cédulas, teléfonos y cuentas del extracto dentro de un texto, también con puntos,
    guiones o espacios (`cc 1.037.000.000`, `300-123-4567`);
  - **texto incrustado:** nombres completos, direcciones y correos del extracto dentro de un texto más largo;
  - **forma de correo:** cualquier correo en una columna de texto que se conserva;
  - **nombre de columna:** `--conservar CELULAR_ALTERNO` se rechaza porque su nombre sugiere un dato personal.

  **Límites: no garantiza una anonimización completa.** Un nombre o apellido suelto solo se busca exacto (dentro
  de un texto marcaría "Santa María"). Tampoco ve:
  - los valores cortos, como una extensión;
  - un dato transformado (abreviado, con errores);
  - un dato personal que no venga de las columnas personales del extracto.

  La revisión humana del extracto sigue siendo necesaria.
- **Seudónimo:** HMAC-SHA256 con la clave `ANON_SALT`, los 64 caracteres hexadecimales completos y en texto
  (nunca un número). El mismo identificador da siempre el mismo seudónimo, así que se pueden unir extractos.
  **Seudonimizar no es anonimizar de forma irreversible:** con la clave se puede recalcular el seudónimo de una
  cédula conocida. La clave la guarda solo el custodio: nunca va al repositorio, a MLflow ni a los logs.
- **Predicciones (`make predecir`):** la salida nunca copia cédulas ni datos personales. Para cruzarla quedan
  `fila` y los seudónimos `*_SEUDO`. Un `*_SEUDO` que no sea un HMAC (p. ej. una cédula renombrada) se rechaza.

  | Origen | Qué se acepta | Qué sale en el CSV |
  |---|---|---|
  | `sintetico` | cualquier archivo (compatibilidad) | `fila`, los `*_SEUDO` válidos, puntaje, zona, alerta, grupo |
  | `real_anonimizado` | solo columnas del contrato para ML (lista blanca), sin PII y con al menos `CEDULA_SEUDO` o `IDASOCIADOS_SEUDO` | `fila`, `CEDULA_SEUDO`, `IDASOCIADOS_SEUDO`, puntaje, zona, alerta, grupo |

  Volver del seudónimo a la persona, si hace falta para contactar a alguien, es una tabla de correspondencia que
  guarda el custodio, **fuera** de este pipeline, del repositorio y de MLflow.
- **Brier:** solo se calcula si el puntaje viene de `predict_proba` (señal explícita `metodo=`, no el rango de los
  valores). Con `decision_function` (Ridge) o sin método queda en NaN. Un método desconocido, o NaN, infinitos o
  valores fuera de [0, 1] con `predict_proba`, dan error.
- **Cuasi-identificadores que quedan:** `FECHANACIMIENTO` (la necesita EDAD), barrio, comuna, cargo y agencia.
  Combinados podrían señalar a alguien en un fondo chico. Si el fondo lo pide, se agrupan con `--bin`, o la edad
  se calcula antes de compartir; eso cambia el contrato para ML y se decide con el equipo.

## Fecha de corte y temporalidad (confirmar con el fondo)

- **Una sola fecha:** `FECHA_CORTE` (`.env`, por defecto `2026-10-31`, la de los sintéticos). La usan la foto
  (t0 = corte − 3 meses), el objetivo, EDAD y ANTIGUEDAD, y la regla de edad del contrato. Ya no hay ninguna
  fecha escrita a mano en el contrato. La efectiva queda en `campeon.json` (`fecha_corte`). En modo real, si el
  archivo trae fechas posteriores al corte, `make campeon` se detiene.
- **Qué depende del tiempo:**
  - EDAD y ANTIGUEDAD se calculan a t0 desde fechas: sin fuga;
  - el objetivo sale de `FECHARETIRO` entre t0 y el corte;
  - los conteos de comportamiento (`abandono.CONTEOS_SIN_FECHA`: solicitudes, ahorros, créditos, servicios,
    intenciones de retiro, incluidos los `ULT3MESES`) llegan como números **sin fecha de medición**.

  El código no puede saber si se midieron a t0 o a la fecha de extracción. Si fue a la extracción, para los
  retirados incluyen lo que pasó después de t0, y eso es fuga.
- **Mientras el fondo no lo confirme**, con datos reales `make campeon` entrena pero avisa. Además marca la
  evaluación como **no válida** (`temporalidad_confirmada: false` en `campeon.json`, en el resumen del CI y en el
  simulador). Cuando el fondo confirme que los conteos son a t0: `TEMPORALIDAD_CONFIRMADA=si` en el `.env`. Si no
  lo son, la solución es pedir los conteos a t0; el pipeline no inventa historia que no está en el extracto.
