"""Полный пайплайн: (опционально) подбор гиперпараметров -> обучение -> оценка на test.

Примеры:
    python -m mafia_cv.train                       # скачать датасет и обучить
    python -m mafia_cv.train --tune                # сначала подобрать lr0/momentum
    python -m mafia_cv.train --data datasets/mafia-cv-1/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_hyperparameters(tune_dir: Path) -> Path | None:
    """Найти самый свежий best_hyperparameters.yaml, который сохраняет тюнер."""
    candidates = sorted(tune_dir.rglob("best_hyperparameters.yaml"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_tuning(model_name: str, data_yaml: Path, cfg: dict, device, seed: int) -> dict:
    """Эволюционный подбор гиперпараметров Ultralytics. Возвращает лучшие значения."""
    from ultralytics import YOLO

    tune_cfg = cfg["tune"]
    space = {k: tuple(v) for k, v in tune_cfg["space"].items()}
    tune_dir = ROOT / "runs" / "tune"

    print(f"\n--- Подбор гиперпараметров: {list(space)} ---")
    YOLO(model_name).tune(
        data=str(data_yaml),
        space=space,
        epochs=tune_cfg["epochs"],
        iterations=tune_cfg["iterations"],
        optimizer=cfg["train"]["optimizer"],  # тот же оптимизатор, что и в финальном обучении
        imgsz=cfg["train"]["imgsz"],
        batch=cfg["train"]["batch"],
        seed=seed,
        device=device,
        project=str(tune_dir),
        name="search",
        exist_ok=True,
        plots=False,
        save=False,
        val=True,
    )

    best_path = find_best_hyperparameters(tune_dir)
    if best_path is None:
        raise FileNotFoundError(f"Тюнер не сохранил best_hyperparameters.yaml в {tune_dir}")
    with open(best_path, encoding="utf-8") as f:
        best = yaml.safe_load(f) or {}
    best = {k: float(best[k]) for k in space if k in best}

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / "best_hyperparameters.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best, f, allow_unicode=True)
    print(f"Лучшие гиперпараметры: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--data", type=Path, help="готовый data.yaml (иначе скачаем из Roboflow)")
    parser.add_argument("--tune", action="store_true", help="сначала подобрать гиперпараметры")
    parser.add_argument("--epochs", type=int, help="переопределить число эпох")
    parser.add_argument("--device", default=None, help="0, cpu, mps; по умолчанию - автоматически")
    parser.add_argument("--name", default="mafia-yolov8n")
    parser.add_argument(
        "--export-to", type=Path, default=ROOT / "models" / "best.pt",
        help="куда скопировать лучшие веса после обучения",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    from .evaluate import evaluate

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))

    if args.data:
        data_yaml = args.data
    else:
        from .data import download_dataset

        data_yaml = download_dataset(out_dir=ROOT / "datasets")

    train_params = dict(cfg["train"])
    if args.epochs:
        train_params["epochs"] = args.epochs

    if args.tune:
        train_params.update(run_tuning(cfg["model"], data_yaml, cfg, args.device, seed))

    print(f"\n--- Обучение с параметрами: {train_params} ---")
    model = YOLO(cfg["model"])
    model.train(
        data=str(data_yaml),
        seed=seed,
        device=args.device,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        plots=True,
        **train_params,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"Лучшие веса: {best_weights}")

    # Оценка на отложенной выборке - только один раз, в самом конце.
    evaluate(best_weights, data_yaml, split="test", out_dir=ROOT / "reports")

    if args.export_to:
        args.export_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, args.export_to)
        print(f"Веса скопированы в {args.export_to}")


if __name__ == "__main__":
    main()
