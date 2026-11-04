# Contrato de datos y Supabase

> Parte de la documentación de [proyecto-ml](../README.md). Todas las cifras son de **datos sintéticos**.

[`contratos/asociados.yaml`](../contratos/asociados.yaml) se valida con **pandera** en tres niveles:

| Nivel | En el contrato | Qué atrapa | Ejemplos | ¿También en la BD? |
|---|---|---|---|---|
| 1. Por fila | `grupos` → `pa.Column` + `Check.ge/le/isin/str_matches` | registros corruptos | `ESTRATO` en 1–6, `SEXO` en F/M | Sí (CHECK) |
| 2. Distribución | `distribucion` → check sobre el lote | problemas sistemáticos | `salario_en_pesos`, `volumen_minimo` | No: la BD ve fila a fila |
| 3. Entre columnas | `reglas` → check con varias columnas | incoherencias | `retiro_coherente`, `nombreintegrado_coherente` | Sí (CHECK) |

Cada check de nivel 2 tiene `severidad`: **`error`** (por defecto, bloquea) o **`advertencia`** (se informa con
`contrato.advertencias` y no bloquea). `tasa_retiro_plausible` (5 %–50 % de `#RETIROS=1` en el lote) es
advertencia: el rango salió del diseño de los sintéticos (10 % de retiro), no de una regla del fondo, y un
extracto real con otra prevalencia no es por eso un dato malo. **Pendiente: confirmar el rango con el fondo.** Si
da una regla validada, se ajustan `min`/`max` en el YAML y se pasa a `severidad: error`.

De ahí salen la validación en Python, la tabla de Supabase y el formulario web:

```bash
make contrato                              # valida data/ejemplo (o DATA=...)
uv run python scripts/ingesta.py --input ... --target "#RETIROS" --contrato contratos/asociados.yaml
make contrato-sql contrato-web             # regenerar tras cambiar el contrato
```

**Configurar Supabase (una vez):**

1. Crear un proyecto gratuito en [supabase.com](https://supabase.com).
2. *SQL Editor* → pegar [`supabase/asociados.sql`](../supabase/asociados.sql) → *Run*. Crea `asociados` con todas
   las restricciones del contrato, cerrada con RLS (la clave pública sola no lee, inserta, edita ni borra), y
   la parte del formulario: `equipo` y la política que deja insertar al equipo (ver abajo). Si la tabla ya
   existía, correr en orden los archivos de [`supabase/migraciones/`](../supabase/migraciones/) que falten (la
   **003** deja que el equipo inserte directo en `asociados`).
3. *Project Settings → API Keys*: copiar la URL, la clave **pública** (anon / publishable) y crear una
   **secreta** (*Secret keys*). La secreta va solo en tu `.env` (`SUPABASE_SECRET_KEY`): nunca en el HTML,
   Git ni un chat, porque salta el RLS.
4. `.env`: `SUPABASE_URL`, `SUPABASE_KEY` (pública) y `SUPABASE_SECRET_KEY`. `web/config.js` (copiar
   `web/config.example.js`): URL y clave **pública**.

```bash
make supabase-subir                        # sube los 50 de ejemplo con la clave secreta (omite repetidos)
make supabase-bajar                        # baja la tabla a data/raw/ (clave secreta; valida el contrato)
make formulario                            # abre web/formulario.html: sesión del equipo + Enviar
```

**Formulario solo para el equipo** ([migración 003](../supabase/migraciones/003_formulario_directo.sql)):

```
formulario ──(correo y contraseña: Supabase Auth)──► asociados ──► make supabase-bajar
              solo correos de public.equipo          de una vez, con las restricciones del contrato
```

- **Una sola entrada con correo y contraseña.** Se entra una vez por navegador y se envían todos los registros
  que se quiera: la sesión se guarda y se renueva sola hasta *Salir*. Supabase no envía correos (sin SMTP
  propio solo permite unos pocos por hora). Sin registro abierto: solo cuentas creadas en el panel.
- **Lista de correos permitidos.** La tabla `public.equipo`; la política `equipo_inserta` de `asociados` exige
  `public.es_del_equipo()` (el correo del token está en `equipo`). La clave pública sola, sin sesión, no
  inserta; y con sesión el equipo solo inserta (no lee, edita ni borra).
- **Directo a la tabla.** Cada envío queda de una vez en `asociados`. La base aplica todas las restricciones del
  contrato por fila y no deja repetir cédula ni IDASOCIADOS: si algo no cumple, el formulario muestra el motivo
  y no se guarda.

Activarlo en Supabase (una vez, en el panel):

1. *SQL Editor*: correr la migración **003** (si falta, antes la 002; o `asociados.sql` completo en una base nueva).
2. *Authentication → Sign In / Providers → Email*: activo, y **desactivar "Allow new users to sign up"**.
3. *Authentication → Users → Add user → **Create new user***: correo y contraseña de cada integrante, con
   **Auto Confirm User** marcado (así no se envía ningún correo). La contraseña se le pasa a cada uno en persona.
4. *Table Editor → equipo → Insert row*: el mismo correo **en minúsculas** (y el nombre).

Para probar en tu PC: `make formulario` (o la dirección de Pages, `https://<usuario>.github.io/<repo>/formulario.html`).

**Entrenar con lo que llega a Supabase:**

```
formulario / supabase-subir ──► asociados ──► make supabase-bajar ──► data/raw/supabase_asociados.parquet ──► make abandono
                                              (clave secreta,          (+ copia fechada; valida el contrato)
                                               solo en tu .env)
```

1. En Supabase, *Project Settings → API Keys → Secret keys*: crear una y ponerla en el `.env` como
   `SUPABASE_SECRET_KEY`. Nunca en Git, en la web ni en un chat: salta el RLS.
2. `make supabase-bajar`: descarga, valida con el contrato (si no cumple, no guarda) y deja la última copia.
3. En el `.env`, el interruptor `DATOS_ASOCIADOS` elige con qué entrenan los modelos (una línea sin comentar):
   `data/ejemplo/asociados_ejemplo.parquet` o `data/raw/supabase_asociados.parquet`. Luego `make abandono`.

Se entrena con la copia descargada, no con la BD en vivo: así un entrenamiento se puede repetir con los mismos datos.

**Formulario en internet (GitHub Pages, gratis):** `https://juanccadavid.github.io/proyecto-ml/formulario.html`
(la raíz del sitio redirige ahí).
El workflow [`.github/workflows/formulario.yml`](../.github/workflows/formulario.yml) lo publica en cada cambio de
`web/` en `main`, armando `config.js` con variables del repositorio (la clave no está en Git). Activación, una vez:

1. *Settings → Secrets and variables → Actions → Variables*: crear `SUPABASE_URL` y `SUPABASE_KEY`
   (la clave **publishable**; el workflow se niega a publicar una `sb_secret_` o `service_role`).
2. *Settings → Pages → Source*: **GitHub Actions**.
3. *Actions → Formulario (GitHub Pages) → Run workflow*.

Quien abra el enlace sin ser del equipo no puede enviar nada: no tiene cuenta (no hay registro abierto) y,
aunque la tuviera, su correo no está en `equipo`. Si las migraciones 002 y 003 aún no están aplicadas en la
base, lo mínimo es despublicar: *Settings → Pages → Unpublish site*, y publicarlo solo el día de la demo.

## Qué va en el contrato y qué va en el test

| | Contrato | Test |
|---|---|---|
| Qué es | La regla: describe el dato válido ("`ESTRATO` está entre 1 y 6") | La prueba de que la regla está encendida ("si llega estrato 7, la validación falla") |
| Dónde vive | `contratos/asociados.yaml` + `src/proyecto_ml/contrato.py` | `tests/data/test_contrato_datos.py` |
| Cuándo corre | En la frontera del pipeline: ingesta `--contrato`, `make supabase-subir`/`supabase-bajar`, CHECK de la BD | En el CI (`.github/workflows/ci.yml`) y con `make test`; nunca en producción |

Hacen falta los dos: un contrato sin tests se degrada (alguien relaja un rango para desbloquear algo y nadie
lo nota). Por eso los límites del test están **escritos a mano**, no leídos del YAML: relajar una regla rompe
un test con nombre propio (p. ej. `test_nivel_1_por_fila[ESTRATO=7]`), y el PR queda en rojo. Si cambias una
regla a conciencia, actualiza su caso en el mismo PR.
