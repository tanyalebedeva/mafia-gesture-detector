"""Инференс на изображении, папке, видео или веб-камере.

    python -m mafia_cv.predict path/to/photo.jpg
    python -m mafia_cv.predict path/to/folder --save
    python -m mafia_cv.predict game.mp4 --save
    python -m mafia_cv.predict 0 --show          # веб-камера
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import DEFAULT_WEIGHTS, detections_from_result, load_model

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="путь к файлу/папке/видео или номер камеры (0)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.5, help="порог уверенности")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true", help="сохранить картинки/видео с рамками")
    parser.add_argument("--show", action="store_true", help="показать окно с результатом")
    parser.add_argument("--json", action="store_true", help="печатать результаты в JSON")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = load_model(args.weights)

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        stream=True,  # не держим все кадры видео в памяти
        project=str(ROOT / "runs" / "predict"),
        name="latest",
        exist_ok=True,
        verbose=False,
    )

    for r in results:
        dets = detections_from_result(r)
        if args.json:
            print(json.dumps({"path": r.path, "detections": [d.to_dict() for d in dets]}, ensure_ascii=False))
        else:
            found = ", ".join(f"{d.description} ({d.confidence:.2f})" for d in dets) or "жестов не найдено"
            print(f"{Path(r.path).name}: {found}")

    if args.save:
        print(f"Результаты сохранены в {ROOT / 'runs' / 'predict' / 'latest'}")


if __name__ == "__main__":
    main()
