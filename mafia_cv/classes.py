"""Классы жестов «Спортивной мафии».

В датасете Roboflow класс 0 по ошибке назван ``back`` вместо ``black``.
Веса (`models/best.pt`) обучены с этим именем, поэтому при инференсе
переименовываем его, а не переобучаем модель.
"""

from __future__ import annotations

# Имена в том порядке, в котором они лежат в чекпоинте.
RAW_NAMES: dict[int, str] = {
    0: "back",
    1: "don",
    2: "mir",
    3: "num1",
    4: "num2",
    5: "num3",
    6: "num4",
    7: "num5",
    8: "sheriff",
}

# Опечатки разметки -> правильное имя.
NAME_FIXES: dict[str, str] = {"back": "black"}

# Человекочитаемое описание для UI и отчётов.
DESCRIPTIONS_RU: dict[str, str] = {
    "black": "Мафия (чёрный)",
    "don": "Дон",
    "mir": "Мирный житель",
    "sheriff": "Шериф",
    "num1": "Номер 1",
    "num2": "Номер 2",
    "num3": "Номер 3",
    "num4": "Номер 4",
    "num5": "Номер 5",
}


def fix_name(name: str) -> str:
    """Исправить имя класса, если в разметке была опечатка."""
    return NAME_FIXES.get(name, name)


def fixed_names(names: dict[int, str]) -> dict[int, str]:
    """Вернуть копию словаря классов с исправленными именами."""
    return {i: fix_name(n) for i, n in names.items()}


def describe(name: str) -> str:
    """Русское описание класса (принимает и «сырое», и исправленное имя)."""
    fixed = fix_name(name)
    return DESCRIPTIONS_RU.get(fixed, fixed)
