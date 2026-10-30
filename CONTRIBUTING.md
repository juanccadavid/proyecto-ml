# Cómo contribuir

## Flujo de trabajo (obligatorio: Pull Requests)

1. Actualiza `main`: `git switch main && git pull`.
2. Crea una rama: `git switch -c feature/<autor>-<tema>` (ej. `feature/juan-eda-creditos`).
   Prefijos: `feature/`, `fix/`, `eda/`, `docs/`.
3. Commits pequeños con mensajes en formato *Conventional Commits*:
   `feat: agrega PCA al preprocesador`, `fix: corrige fuga en split temporal`, `docs: ...`, `test: ...`.
4. Antes de subir: `make lint test` (o `uv run ruff check . && uv run pytest`). El CI de GitHub corre lo mismo
   en cada PR; si queda en rojo, no se hace merge.
5. `git push -u origin <rama>` y abre un Pull Request hacia `main`.
6. Mínimo **1 revisión** de otro integrante antes de hacer merge. Preferir *Squash and merge*.

`main` está protegida: el hook `no-commit-to-branch` impide commits directos.
Para el **primer commit** del repositorio (solo esa vez):

- Git Bash: `SKIP=no-commit-to-branch git commit -m "chore: estructura inicial"`
- PowerShell: `$env:SKIP="no-commit-to-branch"; git commit -m "chore: estructura inicial"; Remove-Item Env:SKIP`

Después, en GitHub: *Settings → Branches → Add rule* para `main` con "Require a pull request before merging".

## Reglas de datos (no negociables)

- Nunca subir datos, modelos, `mlflow.db` ni el `.env`. El hook `bloquear-datos` lo impide; no usar `--no-verify`.
- Los datos se comparten solo por el canal privado del equipo, ya anonimizados.
- Las salidas de notebooks se eliminan al hacer commit (pueden contener filas reales).
- Si algo sensible llega a GitHub: avisar de inmediato. Borrarlo en un commit nuevo **no** lo elimina del historial.

## Dónde va cada cosa

| Tipo | Ubicación |
|---|---|
| Exploración, pruebas rápidas | `sesiones/NN_autor_tema.ipynb` |
| Lógica reutilizable (features, modelos, métricas) | `src/proyecto_ml/` + prueba en `tests/` |
| Tareas puntuales (ingesta, anonimización) | `scripts/` |
| Métricas y figuras para la presentación | `reports/` |

## Dependencias

- Agregar: `uv add <paquete>` (o `uv add --dev <paquete>`). Esto actualiza `pyproject.toml` y `uv.lock`;
  ambos van en el mismo commit.
- No usar `pip install` dentro del proyecto.

## Pruebas

- Toda función nueva en `src/` lleva al menos una prueba.
- Pruebas lentas de punta a punta: marcar con `@pytest.mark.integration`.
