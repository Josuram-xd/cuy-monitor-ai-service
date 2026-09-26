# PRD — Servicio de IA (cuy-monitor-ai-service)

> PRD del componente. El PRD general del producto está en `cuy-monitor-backend/docs/PRD.md`.
> Dueño: compañero · Última revisión: 26 de septiembre de 2026

## 1. Qué es

El servicio que "ve y escucha" la jaula. Recibe frames y clips de audio desde la laptop del criadero, reconoce a cada cuy por su marca de color, calcula cómo se está comportando y clasifica el audio. Le manda los resultados al backend por HTTP para que el backend Java decida el estado de salud.

Este repo también tiene el **edge_agent**, el programa que corre en la laptop y le manda los frames y el audio del celular A12 al servicio.

**Importante:** este servicio **no decide** si un cuy está enfermo ni cambia estados. Solo mide y clasifica. La decisión es del backend (patrones State y Chain of Responsibility).

## 2. Usuarios

No tiene usuario final directo. Sus "clientes" son:
- El **backend** (recibe sus eventos en `POST /api/ingestion/events`).
- El **equipo técnico** (entrena modelos, ajusta zonas y umbrales, revisa que esté vivo).

## 3. Requisitos funcionales

| ID | Requisito | Entrega |
|---|---|---|
| IA-01 | `POST /ai/frames`: recibir un JPEG de la jaula con `X-API-Key` | Avance (mock) |
| IA-02 | `POST /ai/audio`: recibir un WAV de ~1 s con `X-API-Key` | Avance (mock) |
| IA-03 | `GET /ai/health`: indicar si está vivo y si los modelos cargaron | Avance |
| IA-04 | Detectar cuyes en el frame y asignar su `MarkColor` (YOLO26n en ONNX) | Final |
| IA-05 | Mantener la identidad de cada cuy entre frames (tracker con asignación húngara) | Final |
| IA-06 | Calcular por cuy, cada 60 s: `stillSeconds`, `feederVisits`, `watererVisits`, `avgGroupDistance` | Final |
| IA-07 | Clasificar cada ventana con Random Forest → `probAnomaly` | Final |
| IA-08 | Clasificar audio (YAMNet en ONNX + SVM/KNN) → `NORMAL` o `DISTRESS` | Final |
| IA-09 | Mandar eventos `BEHAVIOR` y `AUDIO` a `POST /api/ingestion/events` con el formato del contrato, reintentando si el backend no responde | Avance (valores inventados) / Final |
| IA-10 | edge_agent: leer el stream del A12, mandar 1–2 fps y audio, reintentar si se cae internet | Final |
| IA-11 | Simulador que reproduce un video grabado como si fuera la cámara en vivo | Avance/Final |

## 4. Requisitos no funcionales

| Qué | Meta |
|---|---|
| Rendimiento | Procesar 1–2 frames/s en CPU (sin GPU) |
| Memoria | ≤ 1 GB en el contenedor |
| Robustez | Si un frame llega corrupto o sin cuyes, no se cae: lo descarta y sigue |
| Confianza baja | Si la marca de color se ve mal, reportar `detectionConfidence` baja en vez de inventar un color |
| Reproducibilidad | Los modelos se entrenan con scripts versionados en `training/` |
| Privacidad | No guardar frames ni audio en disco salvo en modo de recolección de datos |

## 5. Para el avance del 30 de septiembre

Un **mock**: FastAPI responde en `/ai/frames` y `/ai/audio` sin modelo y publica eventos con valores inventados pero con el formato exacto del contrato. Así el backend y el dashboard ven datos fluyendo de punta a punta.

## 6. Fuera de alcance

- Diagnóstico de enfermedades.
- Cambiar estados de salud o crear alertas (eso es del backend).
- Reconocer cuyes sin marca de color.
- Streaming de video al dashboard.

## 7. Dependencias

| De | Qué necesita |
|---|---|
| `cuy-monitor-backend/docs/contracts/` | Formato de eventos, endpoint de ingesta y enums |
| `cuy-monitor-backend/infra/docker-compose.yml` | Lo levanta con el perfil `ai`, junto al backend |
| Jaula real | Videos y audios para entrenar (300–600 frames etiquetados, ventanas de comportamiento, audios) |

## 8. Métricas

- mAP del detector por color en frames de prueba.
- Exactitud del Random Forest en ventanas etiquetadas.
- Precisión/recall del clasificador de audio (la clase `DISTRESS` es la difícil).
- FPS reales procesados en la EC2 y en la laptop (para decidir ADR-005).
