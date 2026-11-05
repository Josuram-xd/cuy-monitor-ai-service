# Guía de etiquetado — detector de cuyes por color de marca

Para quien etiqueta los frames de la jaula (Task 6.3) en **Label Studio** o **Roboflow**. El detector aprende
exactamente lo que se marca aquí, así que una caja mal puesta enseña un error.

## 1. Qué se etiqueta

- **Una caja por cuy**, alrededor de **todo el cuerpo** (cabeza a cola), no solo de la marca.
- La **clase es el color de su marca**: `RED`, `BLUE`, `GREEN`, `YELLOW`, `ORANGE`, `PURPLE`, `BLACK`, `WHITE`.
  Sirven también en español (`rojo`, `azul`, `verde`, `amarillo`, `naranja`, `morado`, `negro`, `blanco`):
  `prepare_dataset.py` los traduce.
- En una jaula **no hay dos cuyes con el mismo color**. Si ves dos, uno está mal etiquetado.

## 2. Cómo poner la caja

- Ajustada al cuerpo, sin dejar más de unos pocos píxeles de aire.
- Si dos cuyes se tocan, cada caja va a su propio cuerpo aunque se crucen.
- **Cuy cortado por el borde** de la imagen: se etiqueta si se ve más de la mitad del cuerpo.
- **Cuy tapado** (detrás del comedero, debajo de otro): se etiqueta si se ve más de la mitad **y** la marca.

## 3. Cuándo NO etiquetar

- **No se ve la marca** (de espaldas a la cámara, tapada, desenfocada): **no** se etiqueta ese cuy.
  El servicio nunca adivina un color; el modelo tampoco debe aprender a hacerlo.
- No se etiquetan comederos, bebederos, sombras, manos ni ningún objeto: todo lo que no tiene caja es "fondo".
- Un frame **sin cuyes** se deja sin cajas (se exporta igual): enseña al modelo qué no es un cuy.

## 4. Qué frames elegir (Task 6.1)

- 300 a 600 frames de **distintas horas y luces**: mañana, mediodía, tarde, noche con la luz de la jaula.
- Los cuyes **quietos, comiendo, juntos y separados**.
- No más de 1 frame cada 5 segundos del mismo video: frames casi iguales no enseñan nada nuevo.
- Que **cada color** aparezca al menos 50 veces.

## 5. Exportar

- **Label Studio**: Export → **YOLO**. Trae `images/`, `labels/` y `classes.txt`.
- **Roboflow**: Export → **YOLOv8** (sirve igual para YOLO26). Trae `data.yaml` con los nombres.
- No hace falta que las clases estén en un orden: `prepare_dataset.py` las ordena como las lee el servicio.

```bash
python -m training.prepare_dataset --export ~/export-label-studio --out ~/cuy-dataset
```

El resumen dice cuántos frames quedaron en entrenamiento y validación, cuántas cajas hay por color y qué
líneas se descartaron y por qué. **Revisa las descartadas antes de entrenar.**

## 6. Qué se sube al repo (Task 6.4)

Solo las etiquetas, nunca los frames (son datos crudos de la jaula):

```bash
python -m training.prepare_dataset --export ~/export-label-studio --out training/dataset --labels-only
```

`training/dataset/manifest.csv` guarda el `sha256` de cada frame, para saber con qué imágenes se entrenó.
Los frames se guardan aparte (Drive del equipo) con el mismo nombre.
