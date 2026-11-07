"""Runner `wandb_backfill`: зворотне логування готових ранів у W&B (рани пройшли без ключа).

Для кожного рану з RESULTS (є metrics.json) один W&B-ран у проєкті common.WANDB_PROJECT:
config.yaml як config, криві з train_log.csv (крок = епоха), metrics.json у summary, per_size і per_class
таблицями, N кадрів val із рамками (передбачення й GT як wandb.Image boxes), plots/*.png.

Ідентифікація. common.wandb_init називає ран `run_id` (у SMOKE `_smoke_<run_id>`) і дає йому випадковий id.
Тут id детермінований (`bf-<run_id>`, у SMOKE `bfsmoke-<run_id>`), resume="allow": повторний запуск
оновлює той самий ран. Ран, що вже логувався наживо (у проєкті є ран з таким name і іншим id), пропускається.

Без ключа чи без пакета wandb: повідомлення в log.txt, metrics.json із кількістю пропущених, без падіння.
Конфіг: runs (список або null = усі), run_prefixes, exclude_runners, n_frames, score_thr, display_side.
У SMOKE: SMOKE_RESULTS, не більше 2 ранів по 2 кадри.
"""
import csv
import json
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import yaml

from . import common, data

SMOKE_RUNS = 2
SMOKE_FRAMES = 2
MAX_BOXES = 200


def _discover(root, cfg):
    """Назви ран-тек із metrics.json: або cfg['runs'], або за префіксом і runner-ом з config.yaml."""
    if cfg.get("runs"):
        return [r for r in cfg["runs"] if (root / r / "metrics.json").exists()]
    found = []
    for folder in sorted(root.iterdir()) if root.exists() else []:
        if not folder.is_dir() or folder.name.startswith("_") or not (folder / "metrics.json").exists():
            continue
        if folder.name[0] not in cfg["run_prefixes"]:
            continue
        runner = None
        config_path = folder / "config.yaml"
        if config_path.exists():
            runner = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("runner")
        if runner in cfg["exclude_runners"]:
            continue
        found.append(folder.name)
    return found


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _number(text):
    try:
        return float(text)
    except ValueError:
        return text


def _live_run_exists(wandb, project, name, own_id):
    """True, якщо в проєкті є ран із цим name, але не наш backfill-id (його залогували наживо)."""
    try:
        api = wandb.Api()
        for run in api.runs(f"{api.default_entity}/{project}", filters={"display_name": name}):
            if run.id != own_id:
                return True
    except Exception as e:  # проєкту ще немає чи API недоступний: вважаємо, що наживо не логувалося
        print(f"wandb_backfill: перевірка дублікатів {name}: {e!r}")
    return False


def _box_items(boxes, labels, scale, scores=None):
    items = []
    for i in range(min(len(boxes), MAX_BOXES)):
        x1, y1, x2, y2 = [float(v) * scale for v in boxes[i]]
        label = int(labels[i])
        caption = data.CLASS_NAMES[label] if scores is None else f"{data.CLASS_NAMES[label]} {float(scores[i]):.2f}"
        item = {
            "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
            "class_id": label,
            "box_caption": caption,
            "domain": "pixel",
        }
        if scores is not None:
            item["scores"] = {"score": float(scores[i])}
        items.append(item)
    return items


def _val_images(wandb, run_id, smoke, data_root, n_frames, score_thr, side):
    """wandb.Image із передбаченнями й GT для n_frames кадрів val, рівномірно по відсортованому списку."""
    from PIL import Image

    preds = data.load_preds(common.find_preds(run_id, smoke))
    frames = [f for f in data.load_frames(data_root, "val") if f["image_id"] in preds]
    if not frames:
        return []
    picks = sorted(set(np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int).tolist()))
    class_labels = {i: name for i, name in enumerate(data.CLASS_NAMES)}
    images = []
    for index in picks:
        frame = frames[index]
        pred = preds[frame["image_id"]]
        keep = pred["scores"] >= score_thr
        image = Image.fromarray(data.read_image_rgb(frame["path"]))
        scale = min(1.0, side / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
        boxes = {
            "predictions": {
                "box_data": _box_items(pred["boxes"][keep], pred["labels"][keep], scale, pred["scores"][keep]),
                "class_labels": class_labels,
            },
            "ground_truth": {
                "box_data": _box_items(frame["boxes"], frame["labels"], scale),
                "class_labels": class_labels,
            },
        }
        images.append(wandb.Image(np.asarray(image), boxes=boxes, caption=f"{frame['image_id']} (score >= {score_thr})"))
    return images


def _backfill_one(wandb, run_id, root, cfg, smoke, data_root):
    """Логує один ран. Повертає URL W&B-рану."""
    folder = root / run_id
    name = f"_smoke_{run_id}" if smoke else run_id
    config_path = folder / "config.yaml"
    run_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    run = wandb.init(
        project=common.WANDB_PROJECT,
        id=("bfsmoke-" if smoke else "bf-") + run_id,
        name=name,
        resume="allow",
        config=run_cfg,
        tags=["backfill"] + (["smoke"] if smoke else []),
        dir=tempfile.gettempdir(),
        settings=wandb.Settings(console="off"),
    )
    try:
        # криві: крок = епоха
        log_path = folder / "train_log.csv"
        if log_path.exists():
            rows = _read_csv(log_path)
            for row in rows[1:]:
                values = {k: _number(v) for k, v in zip(rows[0], row) if v != ""}
                if "epoch" in values:
                    run.log(values, step=int(values["epoch"]))
        # summary: скаляри як є, решта JSON-рядком
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        for key, value in metrics.items():
            run.summary[key] = value if isinstance(value, (bool, int, float, str)) else json.dumps(value, ensure_ascii=False)
        for table_name in ("per_size", "per_class"):
            path = folder / f"{table_name}.csv"
            if path.exists():
                rows = _read_csv(path)
                if rows:
                    run.log({table_name: wandb.Table(columns=rows[0], data=[[_number(v) for v in r] for r in rows[1:]])})
        # передбачення на val
        try:
            images = _val_images(
                wandb, run_id, smoke, data_root, SMOKE_FRAMES if smoke else cfg["n_frames"], cfg["score_thr"], cfg["display_side"]
            )
            if images:
                run.log({"val_predictions": images})
            else:
                print(f"wandb_backfill: {run_id}: немає кадрів val для передбачень")
        except FileNotFoundError as e:
            print(f"wandb_backfill: {run_id}: preds_val.json.gz немає, передбачення пропущено ({e})")
        # картинки рану
        for plot in sorted((folder / "plots").glob("*.png")):
            run.log({f"plots/{plot.stem}": wandb.Image(str(plot))})
        return run.url, run.entity
    finally:
        run.finish()


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    root = common.SMOKE_RESULTS if smoke else common.RESULTS
    run_ids = _discover(Path(root), cfg)
    if smoke:
        run_ids = run_ids[:SMOKE_RUNS]
    print(f"wandb_backfill: кандидатів {len(run_ids)}: {run_ids}")
    metrics = {"candidates": run_ids, "logged": [], "skipped_live": [], "skipped_no_key": 0, "errors": {}, "wandb_url": None}

    def finish():
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))

    key = common._wandb_key()
    if not key:
        print("wandb_backfill: ключа WANDB_API_KEY немає (Colab Secrets / змінна середовища), нічого не залоговано")
        metrics["skipped_no_key"] = len(run_ids)
        return finish()
    try:
        import wandb
    except ImportError:
        print("wandb_backfill: пакет wandb не встановлений, нічого не залоговано")
        metrics["skipped_no_key"] = len(run_ids)
        return finish()
    os.environ["WANDB_API_KEY"] = key
    os.environ.setdefault("WANDB_SILENT", "true")

    for run_id in run_ids:
        name = f"_smoke_{run_id}" if smoke else run_id
        own_id = ("bfsmoke-" if smoke else "bf-") + run_id
        try:
            if _live_run_exists(wandb, common.WANDB_PROJECT, name, own_id):
                print(f"wandb_backfill: {run_id}: уже залогований наживо, пропускаю")
                metrics["skipped_live"].append(run_id)
                continue
            url, entity = _backfill_one(wandb, run_id, Path(root), cfg, smoke, data_root)
            print(f"wandb_backfill: {run_id}: {url}")
            metrics["logged"].append(run_id)
            metrics["wandb_url"] = f"https://wandb.ai/{entity}/{common.WANDB_PROJECT}"
        except Exception as e:
            traceback.print_exc()
            metrics["errors"][run_id] = repr(e)
            if smoke:
                finish()
                raise
    finish()
