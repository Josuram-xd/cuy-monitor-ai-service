# AGENTS.md — cuy-monitor-ai-service

Instrucciones para cualquier agente de IA que trabaje en este repo. Léelas completas antes de tocar código.

## Qué es este repo

Servicio en **Python 3.14 + FastAPI** que detecta y sigue a cada cuy por su marca de color, calcula su comportamiento en ventanas de 60 s, clasifica el audio de la jaula y le manda los resultados al backend por HTTP. También contiene el `edge_agent` (corre en la laptop del criadero) y los scripts de entrenamiento (corren en Google Colab).

Lee antes de trabajar:
- `docs/PRD.md` — qué hace y qué no.
- `docs/ARCHITECTURE.md` — pipeline, modelos, API, despliegue.
- `cuy-monitor-backend/docs/contracts/` — formato de eventos. **Fuente de verdad.**

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
4. Los resultados se mandan con `POST {BACKEND_URL}/api/ingestion/events` (sobre común, tipos `BEHAVIOR` y `AUDIO`, header `X-API-Key`). No agregues colas ni brokers de mensajes: la comunicación es HTTP directo. Si el backend no responde, reintenta con backoff usando el mismo `eventId`; en `400`/`401` no reintentes.
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

## Git

- Conventional Commits en inglés: `feat(vision): add hungarian tracker`, `fix(audio): resample to 16 kHz`, `chore(models): bump detector to v2`.
- `main` solo por Pull Request. **Prohibido** `git push --force` a `main`.

## Lo que el agente NO debe hacer sin permiso explícito

- Cambiar el formato de eventos o el endpoint de ingesta.
- Cambiar la versión de Python o de onnxruntime.
- Tocar `cuy-monitor-backend/infra/` (el `docker-compose.yml` lo mantiene Josuram; el servicio `ai-service` ahí se coordina con él).
- Descargar datasets o modelos grandes.
- Hacer push o abrir PRs.

## Dueño

Todo el repo: **el compañero**. Josuram revisa los PRs.

## Herramientas que puede usar el agente

- Leer y editar archivos del repo.
- Correr `uv`/`pip`, `pytest`, `ruff`, `uvicorn` y `docker build` localmente.
- `git status`, `git diff`, `git log`, ramas y commits locales.
