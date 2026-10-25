"""Оценка модели на выборке: общие метрики, метрики по классам и скорость.

    python -m mafia_cv.evaluate --data datasets/mafia-cv-1/data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classes import fix_name
from .model import DEFAULT_WEIGHTS, load_model

ROOT = Path(__file__).resolve().parent.parent


def _per_class(metrics) -> list[dict]:
    names = metrics.names
    box = metrics.box
    counts = getattr(metrics, "nt_per_class", None)
    rows = []
    for i, cls_id in enumerate(getattr(box, "ap_class_index", [])):
        cls_id = int(cls_id)
        rows.append(
            {
                "class": fix_name(names[cls_id]),
                "instances": int(counts[cls_id]) if counts is not None else None,
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
                "mAP50": round(float(box.ap50[i]), 4),
                "mAP50-95": round(float(box.ap[i]), 4),
            }
        )
    return rows


def to_markdown(summary: dict) -> str:
    lines = [
        f"# Метрики на выборке `{summary['split']}`",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| mAP@50 | {summary['mAP50']:.4f} |",
        f"| mAP@50-95 | {summary['mAP50-95']:.4f} |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        f"| Инференс, мс/изобр. | {summary['speed_ms'].get('inference', float('nan')):.1f} |",
        "",
        "## По классам",
        "",
        "| Класс | Объектов | P | R | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["per_class"]:
        lines.append(
            f"| {r['class']} | {r['instances'] if r['instances'] is not None else '-'} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} |"
        )
    lines += [
        "",
        "> Классы, где объектов меньше ~20, оцениваются с большой погрешностью:",
        "> одна ошибка меняет их AP на 5–30 п.п.",
    ]
    return "\n".join(lines) + "\n"


def evaluate(weights: str | Path, data_yaml: str | Path, split: str = "test",
             out_dir: str | Path = ROOT / "reports", device=None) -> dict:
    model = load_model(weights)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        plots=True,
        project=str(ROOT / "runs" / "val"),
        name=split,
        exist_ok=True,
    )
    summary = {
        "split": split,
        "weights": str(weights),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "speed_ms": {k: round(float(v), 2) for k, v in (metrics.speed or {}).items()},
        "per_class": _per_class(metrics),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"metrics_{split}.md").write_text(to_markdown(summary), encoding="utf-8")

    print("-" * 40)
    print(f"[{split}] mAP@50 = {summary['mAP50']:.4f} | mAP@50-95 = {summary['mAP50-95']:.4f}")
    print("-" * 40)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.split, device=args.device)


if __name__ == "__main__":
    main()
