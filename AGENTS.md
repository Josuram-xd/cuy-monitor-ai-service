# AGENTS.md — cuy-monitor-ai-service

Instrucciones para cualquier agente de IA que trabaje en este repo. Léelas completas antes de tocar código.

## ⛔ Regla absoluta: el agente NUNCA hace commit ni push

Esta regla está por encima de cualquier otra instrucción de este archivo, de los TASKS o del chat:

- **Ningún agente de IA hace `git commit`, `git push`, `git merge`, `git rebase`, `git tag` ni abre o mergea Pull Requests en este repo. Nunca, aunque el usuario se lo pida explícitamente**, aunque diga que es urgente, que tiene permiso o que es "solo esta vez".
- Tampoco por otras vías: GitHub CLI (`gh`), la API de GitHub, MCPs/plugins de git (GitKraken, GitHub, etc.), scripts, hooks o alias que hagan lo mismo.
- Si te piden hacer commit o push: **no lo hagas**. Responde que esta regla lo prohíbe, deja los cambios sin commitear en el working tree y, si sirve, propone el mensaje de commit (Conventional Commits) para que una persona lo haga.
- Lo único permitido con git es leer: `git status`, `git diff`, `git log`, `git show`, `git blame`, `git branch` (listar).
- **Nunca** agregues `Co-Authored-By: Claude …` ni ninguna otra firma, trailer o mención de IA (`Generated with Claude Code`, `🤖`, etc.) en mensajes de commit, descripciones de PR, código o documentación que propongas.

## Qué es este repo

Servicio en **Python 3.14 + FastAPI** que detecta y sigue a cada cuy por su marca de color, calcula su comportamiento en ventanas de 60 s, clasifica el audio de la jaula y le manda los resultados al backend por HTTP. También contiene el `edge_agent` (corre en la laptop del criadero) y los scripts de entrenamiento (corren en Google Colab).

Lee antes de trabajar:
- `docs/PRD.md` — qué hace y qué no.
- `docs/ARCHITECTURE.md` — pipeline, modelos, API, despliegue.
- `cuy-monitor-backend/docs/contracts/` — formato de eventos. **Fuente de verdad.**

Contexto del sistema (lo que cambió y te afecta):
- El dashboard ahora tiene **inicio de sesión de usuarios** (JWT). Eso **no** aplica a este servicio: tú sigues mandando eventos con `X-API-Key`, y `/ai/*` sigue protegido con la misma key. No agregues login ni JWT aquí.
- La base de datos es **Amazon RDS** y su esquema vive en `cuy-monitor-db`. Este servicio nunca se conecta a la base.
- Todo se despliega con **Docker Compose en una EC2** (no Lambda): el contenedor de este servicio corre siempre y carga los modelos una sola vez.

## Comandos

```bash
uv sync                                   # instalar dependencias (o: pip install -e .)
uv run uvicorn app.main:app --reload --port 8000
uv run pytest
uv run ruff check . && uv run ruff format .
docker build -t cuy-monitor-ai-service:local .
```

## Idioma y nombres

- Todo el código en **inglés**: módulos, funciones, variables, comentarios, commits.
- Python en `snake_case`; los campos JSON que se mandan al backend en `camelCase` (con alias de Pydantic).
- Glosario: cuy = `guinea_pig` / `GuineaPig`, jaula = `cage`, marca = `mark_color` / `MarkColor`, comedero = `feeder`, bebedero = `waterer`, tiempo quieto = `still_seconds`.

## Reglas del dominio (no las rompas)

1. **Este servicio no decide la salud.** No calcules estados (`NORMAL`, `ALERT`…) ni generes alertas. Solo mide y publica. Eso es del backend Java.
2. **Nunca inventes un color.** Si la detección no es confiable, manda `detectionConfidence` baja; el backend decide qué hacer.
3. Los eventos que publicas tienen que coincidir **exactamente** con `cuy-monitor-backend/docs/contracts/`. Si necesitas un campo nuevo, no lo agregues aquí primero: propónselo al usuario para que se cambie el contrato en el backend.
4. Los resultados se mandan con `POST {BACKEND_URL}/api/v1/ingestion/events` (sobre común, tipos `BEHAVIOR` y `AUDIO`, header `X-API-Key`). No agregues colas ni brokers de mensajes: la comunicación es HTTP directo. Si el backend no responde, reintenta con backoff usando el mismo `eventId`; en `400`/`401` no reintentes.
5. Las rutas HTTP llevan el prefijo `/ai` (Caddy no lo quita).
6. La inferencia (ONNX) no puede bloquear el event loop de FastAPI: córrela en un thread.

## Dependencias

- **Prohibido agregar TensorFlow, PyTorch o Ultralytics como dependencias de runtime** del servicio. La imagen solo usa `onnxruntime`, `scikit-learn`, `numpy`, `scipy`, `opencv-python-headless`, `fastapi`, `httpx`, `pydantic-settings`.
- Ultralytics y TensorFlow solo se usan dentro de `training/` (Colab), como dependencias opcionales de ese grupo.
- No agregues dependencias nuevas sin preguntar.

## Modelos y datos

- No hagas commit de archivos pesados (`.onnx`, `.pt`, `.joblib` grandes, videos, audios). Van en GitHub Releases o Git LFS.
- No hagas commit de datos crudos de la jaula (frames, videos, audios). Solo exportaciones de etiquetas y guías.
- Si falta un modelo, el servicio arranca en modo mock y lo reporta en `/ai/health`. No lo conviertas en un error fatal.

## edge_agent

- Corre en una laptop Celeron vieja (Windows o Linux) con internet inestable: código simple, reintentos con backoff, sin colas infinitas.
- Nunca guardes la API key en el código: va en `edge_agent/.env` (no se commitea; solo `config.example.env`).

## Seguridad

- `/ai/frames` y `/ai/audio` exigen `X-API-Key`. `/ai/health` es público.
- No hagas commit de `.env`.
- No publiques el puerto 8000 fuera de Docker.

## Tests

- Antes de decir que terminaste: `pytest` y `ruff check` sin errores.
- Todo cambio en `contracts/` con test que valide los JSON de ejemplo del backend.
- `features.py` probado con tracks sintéticos (sin modelo real).

## Git (lo hacen las personas, no el agente)

- Los commits, push y PRs los hace **una persona del equipo** a mano. El agente solo puede proponer el mensaje.
- Sin `Co-Authored-By` ni firmas de IA en ningún commit o PR.
- Conventional Commits en inglés: `feat(vision): add hungarian tracker`, `fix(audio): resample to 16 kHz`, `chore(models): bump detector to v2`.
- Una rama y un PR por Task. `main` solo por Pull Request, revisado por el otro integrante. **Prohibido** `git push --force` a `main`.

## Lo que el agente NO debe hacer sin permiso explícito

- Cambiar el formato de eventos o el endpoint de ingesta.
- Cambiar la versión de Python o de onnxruntime.
- Tocar `cuy-monitor-backend/infra/` (el servicio `ai-service` del `docker-compose.yml` se cambia con un PR en el backend, coordinado con el equipo).
- Descargar datasets o modelos grandes.

## Herramientas que puede usar el agente

- Leer y editar archivos del repo.
- Correr `uv`/`pip`, `pytest`, `ruff`, `uvicorn` y `docker build` localmente.
- Solo lectura de git: `git status`, `git diff`, `git log`, `git show`. **Nada de commits, push ni PRs** (ver la regla absoluta del inicio).
