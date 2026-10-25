"""Проверка утечки между train/valid/test и разбиение по видео.

Датасет собран из кадров видео. Соседние кадры одного ролика почти одинаковы,
поэтому если ролик попал и в train, и в test, test перестаёт быть «новыми
данными» и mAP завышается. Правильно - делить выборку по видео (group split).

Roboflow называет файлы так: ``IMG_0596_MOV-0003_jpg.rf.<hash>.jpg``
  * исходный кадр  -> ``IMG_0596_MOV-0003``
  * исходное видео -> ``IMG_0596_MOV`` (отрезаем номер кадра)

    python -m mafia_cv.leakage datasets/mafia-cv-1
    python -m mafia_cv.leakage datasets/mafia-cv-1 --resplit datasets/mafia-cv-grouped
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
_EXT_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)
_FRAME_SUFFIX = re.compile(r"[-_ .]\d+$")
# Префиксы камер: у фото IMG_0596 «номер» - это не кадр видео, а сам снимок.
_CAMERA_PREFIXES = {"img", "dsc", "dscn", "pxl", "photo", "image", "frame", "vid", "video"}


def original_stem(filename: str) -> str:
    """Имя исходного кадра до обработки Roboflow."""
    stem = Path(filename).name
    stem = stem.split(".rf.")[0] if ".rf." in stem else Path(stem).stem
    return _EXT_SUFFIX.sub("", stem)


def video_key(filename: str, pattern: re.Pattern | None = None) -> str:
    """Идентификатор исходного видео (группы) для кадра."""
    stem = original_stem(filename)
    if pattern is not None:
        m = pattern.search(stem)
        if m:
            return m.group("video") if "video" in m.groupdict() else m.group(0)
    key = _FRAME_SUFFIX.sub("", stem)
    if not key or key.lower() in _CAMERA_PREFIXES:
        return stem
    return key


def list_images(dataset_dir: Path) -> dict[str, list[Path]]:
    out = {}
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        out[split] = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT) if img_dir.exists() else []
    return out


def leakage_report(files: dict[str, list[str]], pattern: re.Pattern | None = None) -> dict:
    """Посчитать пересечения видео и исходных кадров между выборками."""
    videos = {s: {video_key(f, pattern) for f in fs} for s, fs in files.items()}
    originals = {s: {original_stem(f) for f in fs} for s, fs in files.items()}
    report = {"splits": {}}
    for s, fs in files.items():
        report["splits"][s] = {
            "images": len(fs),
            "unique_frames": len(originals[s]),
            "videos": len(videos[s]),
        }
    for s in ("valid", "test"):
        fs = files.get(s, [])
        if not fs:
            continue
        leaked = [f for f in fs if video_key(f, pattern) in videos.get("train", set())]
        dup = originals[s] & originals.get("train", set())
        report[s] = {
            "images_from_train_videos": len(leaked),
            "share_from_train_videos": round(len(leaked) / len(fs), 3),
            "same_frames_as_train": len(dup),
        }
    return report


def group_split(groups: dict[str, int], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, str]:
    """Распределить группы по выборкам так, чтобы доли изображений были близки к ratios."""
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: groups[k], reverse=True)  # крупные группы раскладываем первыми
    total = sum(groups.values())
    target = {s: r * total for s, r in zip(SPLITS, ratios, strict=True)}
    filled = dict.fromkeys(SPLITS, 0)
    assign = {}
    for k in keys:
        # кладём туда, где больше всего «недобора» относительно цели
        split = max(SPLITS, key=lambda s: (target[s] - filled[s]) / max(target[s], 1))
        assign[k] = split
        filled[split] += groups[k]
    return assign


def resplit(dataset_dir: Path, out_dir: Path, pattern: re.Pattern | None, seed: int,
            one_per_original: bool) -> dict[str, int]:
    imgs = [p for ps in list_images(dataset_dir).values() for p in ps]
    if one_per_original:
        seen, kept = set(), []
        for p in imgs:
            o = original_stem(p.name)
            if o not in seen:
                seen.add(o)
                kept.append(p)
        imgs = kept

    sizes = Counter(video_key(p.name, pattern) for p in imgs)
    assign = group_split(dict(sizes), seed=seed)
    counts: dict[str, int] = defaultdict(int)
    for p in imgs:
        split = assign[video_key(p.name, pattern)]
        for sub in ("images", "labels"):
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_dir / split / "images" / p.name)
        label = p.parent.parent / "labels" / (p.stem + ".txt")
        if label.exists():
            shutil.copy2(label, out_dir / split / "labels" / label.name)
        counts[split] += 1

    import yaml

    src_yaml = dataset_dir / "data.yaml"
    meta = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) if src_yaml.exists() else {}
    meta.update({"path": str(out_dir.resolve()), "train": "train/images",
                 "val": "valid/images", "test": "test/images"})
    meta.pop("roboflow", None)
    (out_dir / "data.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="папка датасета YOLO (train/valid/test)")
    parser.add_argument("--video-regex", help="regex с группой (?P<video>...) для имени видео")
    parser.add_argument("--resplit", type=Path, help="создать новое разбиение по видео в эту папку")
    parser.add_argument("--one-per-original", action="store_true",
                        help="оставить одну копию на исходный кадр (убрать аугментированные дубли)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pattern = re.compile(args.video_regex) if args.video_regex else None
    files = {s: [p.name for p in ps] for s, ps in list_images(args.dataset).items()}
    rep = leakage_report(files, pattern)

    print("Выборка   изобр.  кадров  видео")
    for s, v in rep["splits"].items():
        print(f"{s:<8} {v['images']:>6} {v['unique_frames']:>7} {v['videos']:>6}")
    for s in ("valid", "test"):
        if s in rep:
            r = rep[s]
            print(f"\n[{s}] из видео, которые есть в train: {r['images_from_train_videos']} "
                  f"({r['share_from_train_videos']:.0%}); те же самые кадры, что в train: {r['same_frames_as_train']}")
    if rep.get("test", {}).get("share_from_train_videos", 0) > 0:
        print("\n⚠  Test пересекается с train по видео - метрики на test завышены. "
              "Используйте --resplit.")

    if args.resplit:
        counts = resplit(args.dataset, args.resplit, pattern, args.seed, args.one_per_original)
        print(f"\nНовое разбиение по видео: {counts} -> {args.resplit / 'data.yaml'}")


if __name__ == "__main__":
    main()
