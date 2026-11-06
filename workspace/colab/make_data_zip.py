"""Пише data.zip для Colab: без стиснення, у train лише кадри, які вибере frame_step.

    python workspace/colab/make_data_zip.py D:/hw2/data.zip
    python workspace/colab/make_data_zip.py --dry-run

Layout: у корені test_images.csv, train/, val/, test/. val і test повністю. У train усі
лейбли, ignore й анотації, а картинки лише вибрані. Якщо frame_step у протоколі зміниться,
zip треба зібрати заново.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from src.data import load_frames, select_frames  # noqa: E402

ENTRY_OVERHEAD = 30 + 46  # заголовки zip на файл, без імені


def _frame_step():
    """frame_step протоколу; конфіг із кроком, не кратним йому, захотів би відсутні кадри."""
    configs = WORKSPACE / "configs"
    step = yaml.safe_load((configs / "_protocol.yaml").read_text(encoding="utf-8"))["frame_step"]
    for path in sorted(configs.glob("*.yaml")):
        other = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("frame_step", step)
        if other % step:
            print(f"УВАГА: {path.name} має frame_step={other}, цих кадрів у zip не буде")
    return step


def _entries(data_root, step):
    """Пари (файл, ім'я в zip) у порядку обходу."""
    selected = {os.path.normcase(f["path"]) for f in select_frames(load_frames(data_root, "train"), step)}
    train_images = data_root / "train" / "images"
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_relative_to(train_images) and os.path.normcase(str(path)) not in selected:
            continue
        yield path, path.relative_to(data_root).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", type=Path, help="шлях до zip поза репозиторієм")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data")
    parser.add_argument("--dry-run", action="store_true", help="лише порахувати очікуваний розмір")
    args = parser.parse_args()
    if not args.dry_run:
        if args.output is None:
            parser.error("вкажи шлях до zip або --dry-run")
        if args.output.resolve().is_relative_to(WORKSPACE.parent):
            parser.error("zip має лежати поза репозиторієм")

    step = _frame_step()
    entries = _entries(args.data, step)
    count = total = 0
    if args.dry_run:
        for path, name in entries:
            total += path.stat().st_size + ENTRY_OVERHEAD + 2 * len(name.encode())
            count += 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_name(args.output.name + ".tmp")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
            for path, name in entries:
                archive.write(path, name)
                count += 1
        os.replace(partial, args.output)
        total = args.output.stat().st_size
    label = "Очікуваний розмір" if args.dry_run else f"{args.output}"
    print(f"{label}: {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB), файлів {count}, frame_step={step}")


if __name__ == "__main__":
    main()
