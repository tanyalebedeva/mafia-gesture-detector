"""Загрузка обученной модели и инференс."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .classes import describe, fixed_names

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "best.pt"


@dataclass
class Detection:
    label: str
    description: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(weights: str | Path = DEFAULT_WEIGHTS):
    """Загрузить YOLO и исправить опечатки в именах классов (back -> black)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.model.names = fixed_names(dict(model.model.names))
    return model


def detections_from_result(result) -> list[Detection]:
    """Преобразовать ``ultralytics.engine.results.Results`` в список Detection."""
    names = result.names
    out: list[Detection] = []
    if result.boxes is None:
        return out
    for cls_id, conf, xyxy in zip(
        result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True
    ):
        label = names[int(cls_id)]
        out.append(
            Detection(
                label=label,
                description=describe(label),
                confidence=round(float(conf), 4),
                box_xyxy=tuple(round(float(v), 1) for v in xyxy),
            )
        )
    return out
