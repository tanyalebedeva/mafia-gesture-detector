"""Загрузка датасета из Roboflow.

API-ключ берётся только из переменной окружения ``ROBOFLOW_API_KEY``
(или из файла ``.env``) - никогда не храните его в коде и ноутбуках.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKSPACE = "tanyas-workspace-pfewk"
DEFAULT_PROJECT = "mafia-cv"
DEFAULT_VERSION = 1


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv необязателен
        return
    load_dotenv()


def download_dataset(
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
    version: int = DEFAULT_VERSION,
    out_dir: str | Path = "datasets",
) -> Path:
    """Скачать датасет в формате YOLOv8 и вернуть путь к ``data.yaml``."""
    _load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан ROBOFLOW_API_KEY. Скопируйте .env.example в .env и впишите ключ "
            "или выполните: export ROBOFLOW_API_KEY=..."
        )

    from roboflow import Roboflow

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov8", location=str(out_dir / f"{project}-{version}"))
    )
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Не найден {data_yaml}")
    return data_yaml
