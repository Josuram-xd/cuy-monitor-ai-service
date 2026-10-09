# TASKS — cuy-monitor-ai-service

> Lista de trabajo del servicio de IA. Cada subtarea = **un commit**: usa el mensaje que está entre comillas invertidas.
> ⛔ Los commits, push y PRs los hace una persona del equipo. **Ningún agente de IA hace commit ni push, aunque se lo pidan**, y nunca se agrega `Co-Authored-By` ni firmas de IA (ver `AGENTS.md`).
> Marca `[x]` cuando hagas push. Una rama por Task: `feature/task-7-detector`, etc.
> Cada PR lo revisa el otro integrante antes de mergear a `main`.
> Este servicio **no** usa el login de usuarios: sigue con `X-API-Key`. Se despliega en la EC2 con Docker Compose (perfil `ai`).

| Símbolo | Significado |
|---|---|
| 🔴 Prioridad 1 | Crítico: el mock que se muestra en el avance |
| 🟠 Prioridad 2 | Importante: los modelos reales de la entrega final |
| 🟢 Prioridad 3 | Cierre: medición, pruebas y documentación |
| 🔗 Depende de | Antes hay que terminar esas tasks (de este u otro repo) |

---

## 🔴 Prioridad 1 — Avance (hasta el 30 de septiembre)

### Task 1 — Proyecto base

- [x] **Task 1.1** — `chore: init Python 3.14 project with uv, ruff and pytest`
- [x] **Task 1.2** — `docs: add PRD, ARCHITECTURE and AGENTS`
- [x] **Task 1.3** — `feat(config): add settings from environment variables`
  `BACKEND_URL`, `API_KEY`, `CAGE_ID`, `WINDOW_SECONDS`, `MOCK_MODE`.
- [x] **Task 1.4** — `feat(api): add FastAPI app with GET /ai/health`
- [x] **Task 1.5** — `feat(security): add X-API-Key dependency`

### Task 2 — Contratos

🔗 **Depende de:** seguir con las Task 3.2–3.4 del repo `cuy-monitor-backend`

- [x] **Task 2.1** — `feat(contracts): add MarkColor and EventType enums`
- [ ] **Task 2.2** — `feat(contracts): add event envelope and payload models with camelCase aliases`
- [ ] **Task 2.3** — `test(contracts): validate backend example JSON files`

### Task 3 — Mock que manda eventos al backend

🔗 **Depende de:** seguir con las Task 2.3–2.6 del repo `cuy-monitor-backend` (endpoint `/api/ingestion/events` desplegado)

- [ ] **Task 3.1** — `feat(messaging): add backend client with retries and bounded buffer`
  `httpx`, reintento con backoff en `5xx`/red, sin reintento en `400`/`401`, mismo `eventId` en cada reintento.
- [ ] **Task 3.2** — `feat(api): add POST /ai/frames returning 202 in mock mode`
- [ ] **Task 3.3** — `feat(api): add POST /ai/audio returning 202 in mock mode`
- [ ] **Task 3.4** — `feat(mock): send fake BEHAVIOR and AUDIO events every window`
- [ ] **Task 3.5** — `test(messaging): cover retry and no-retry cases`

### Task 4 — Docker y despliegue

🔗 **Depende de:** `cuy-monitor-backend` Task 2.5 (compose con el perfil `ai`)

- [ ] **Task 4.1** — `build: add Dockerfile on python:3.14-slim`
- [ ] **Task 4.2** — *(sin commit)* clonar el repo al lado del backend en la EC2 y levantar con `docker compose --profile ai up -d --build`
- [ ] **Task 4.3** — *(sin commit)* verificar `https://cuymonitor.duckdns.org/ai/health` y los eventos en el log del backend
  Desde octubre la base está en RDS (`cuy-monitor-backend` Task 22): no cambia nada para este servicio, pero los eventos ahora se guardan allá.

### Task 5 — Simulador

- [ ] **Task 5.1** — `feat(dev): add simulator that replays a video as a live camera`

---

## 🟠 Prioridad 2 — Entrega final (octubre)

### Task 6 — Datos para entrenar

- [ ] **Task 6.1** — *(sin commit)* grabar la jaula (o pompones de colores) en distintas horas y luces
- [ ] **Task 6.2** — `docs(training): add labeling guide`
- [ ] **Task 6.3** — *(sin commit)* etiquetar 300–600 frames en Label Studio o Roboflow
- [ ] **Task 6.4** — `chore(training): add labeled dataset export (without raw media)`

### Task 7 — Detector

- [ ] **Task 7.1** — `feat(training): add YOLO26n training script with ONNX export`
- [ ] **Task 7.2** — *(sin commit)* entrenar en Colab y subir `detector.onnx` a GitHub Releases
- [ ] **Task 7.3** — `feat(vision): add ONNX detector returning boxes, color and confidence`
- [ ] **Task 7.4** — `feat(models): download models on build or start in mock mode if missing`

### Task 8 — Tracker

- [ ] **Task 8.1** — `feat(vision): add Hungarian tracker keeping identity per color`
- [ ] **Task 8.2** — `test(vision): cover tracker identity with synthetic boxes`

### Task 9 — Comportamiento

- [ ] **Task 9.1** — `feat(vision): add feeder and waterer zones from config`
- [ ] **Task 9.2** — `feat(behavior): add 60 s window per guinea pig`
- [ ] **Task 9.3** — `feat(behavior): compute still seconds, visits and group distance`
- [ ] **Task 9.4** — `test(behavior): cover features with synthetic tracks`

### Task 10 — Clasificador de comportamiento

- [ ] **Task 10.1** — `feat(training): add Random Forest training script`
- [ ] **Task 10.2** — `feat(behavior): load classifier and compute probAnomaly`
- [ ] **Task 10.3** — `feat(api): send real BEHAVIOR events from the frame pipeline`

### Task 11 — Audio

- [ ] **Task 11.1** — `feat(training): add YAMNet to ONNX export script`
- [ ] **Task 11.2** — `feat(audio): add YAMNet ONNX embeddings`
- [ ] **Task 11.3** — `feat(training): add SVM/KNN audio training script`
- [ ] **Task 11.4** — `feat(audio): classify clips and send real AUDIO events`

### Task 12 — edge_agent (laptop)

- [ ] **Task 12.1** — `feat(edge): read frames from the A12 IP Webcam stream`
- [ ] **Task 12.2** — `feat(edge): send 1–2 fps frames with retries`
- [ ] **Task 12.3** — `feat(edge): record and send 1 s audio clips`
- [ ] **Task 12.4** — `docs(edge): add service install guide for Windows and Linux`
- [ ] **Task 12.5** — `build(edge): add optional Dockerfile for the edge agent on Linux` *(opcional)*
  Solo para una laptop con Linux; en Windows se instala como servicio nativo (Task 12.4).

---

## 🟢 Prioridad 3 — Cierre (noviembre)

### Task 13 — Medición y montaje

🔗 **Depende de:** `cuy-monitor-arduino` Task 7 (montaje en la jaula, se hace el mismo día)

- [ ] **Task 13.1** — *(sin commit)* medir fps y RAM en la EC2 y en la laptop (decide ADR-005)
- [ ] **Task 13.2** — `docs: record detector, behavior and audio metrics`

### Task 14 — Entrega

- [ ] **Task 14.1** — `test: raise coverage of pipeline modules`
- [ ] **Task 14.2** — `docs: update README with models, metrics and setup`
