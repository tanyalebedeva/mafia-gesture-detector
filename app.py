"""Веб-демо на Gradio: загрузите фото или включите камеру.

    pip install -r requirements-app.txt
    python app.py   # http://127.0.0.1:7860

Этот же файл без изменений запускается на Hugging Face Spaces (SDK: gradio).
"""

from __future__ import annotations

import gradio as gr

from mafia_cv.model import DEFAULT_WEIGHTS, detections_from_result, load_model

model = load_model(DEFAULT_WEIGHTS)


def detect(image, conf: float):
    if image is None:
        return None, []
    result = model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    rows = [[d.description, d.label, d.confidence] for d in detections_from_result(result)]
    return annotated, rows


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", sources=["upload", "webcam"], label="Кадр"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Порог уверенности"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Результат"),
        gr.Dataframe(headers=["Жест", "Класс", "Уверенность"], label="Найденные жесты"),
    ],
    title="Детектор жестов «Спортивной мафии»",
    description="YOLOv8n, 9 классов: чёрный, дон, мирный, шериф и номера 1–5.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
