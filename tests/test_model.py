"""Смоук-тесты обученной модели. Пропускаются, если ultralytics не установлен."""

import numpy as np
import pytest

from mafia_cv.classes import RAW_NAMES, fixed_names
from mafia_cv.model import DEFAULT_WEIGHTS, detections_from_result, load_model

pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def model():
    assert DEFAULT_WEIGHTS.exists(), "models/best.pt не найден"
    return load_model(DEFAULT_WEIGHTS)


def test_checkpoint_has_expected_classes(model):
    assert dict(model.names) == fixed_names(RAW_NAMES)


def test_predict_runs_on_blank_image(model):
    img = np.full((640, 640, 3), 127, dtype=np.uint8)
    result = model.predict(img, conf=0.5, device="cpu", verbose=False)[0]
    dets = detections_from_result(result)
    assert isinstance(dets, list)
    # на пустом сером кадре уверенных жестов быть не должно
    assert len(dets) == 0
