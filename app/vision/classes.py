from app.contracts import MarkColor

# The detector has one class per mark colour, in this order: class id 0 is RED, 1 is BLUE...
# Training (training/prepare_dataset.py writes data.yaml from it) and inference read the same tuple,
# so a model can never be trained with one order and read with another.
DETECTOR_CLASSES: tuple[MarkColor, ...] = tuple(MarkColor)

# names a labeler may have typed in Label Studio or Roboflow, in Spanish or English
_ALIASES = {
    "ROJO": MarkColor.RED,
    "AZUL": MarkColor.BLUE,
    "VERDE": MarkColor.GREEN,
    "AMARILLO": MarkColor.YELLOW,
    "NARANJA": MarkColor.ORANGE,
    "MORADO": MarkColor.PURPLE,
    "LILA": MarkColor.PURPLE,
    "NEGRO": MarkColor.BLACK,
    "BLANCO": MarkColor.WHITE,
}


def class_id(color: MarkColor) -> int:
    return DETECTOR_CLASSES.index(color)


def color_of(class_index: int) -> MarkColor:
    if not 0 <= class_index < len(DETECTOR_CLASSES):
        raise ValueError(f"The detector has no class {class_index}")
    return DETECTOR_CLASSES[class_index]


def color_named(name: str) -> MarkColor:
    """'red', 'Rojo', 'mark_red' -> RED. Unknown name: ValueError."""
    key = name.strip().upper().removeprefix("MARK_").removeprefix("MARCA_")
    if key in MarkColor.__members__:
        return MarkColor(key)
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"{name!r} is not a mark colour")
