from mafia_cv.classes import DESCRIPTIONS_RU, RAW_NAMES, describe, fixed_names


def test_typo_back_is_fixed():
    names = fixed_names(RAW_NAMES)
    assert names[0] == "black"
    assert "back" not in names.values()


def test_every_class_has_description():
    for name in fixed_names(RAW_NAMES).values():
        assert name in DESCRIPTIONS_RU


def test_describe_accepts_raw_name():
    assert describe("back") == "Мафия (чёрный)"
    assert describe("sheriff") == "Шериф"
