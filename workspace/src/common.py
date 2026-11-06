"""Спільне для всіх ранів: seed, smoke, ваги, W&B, Predictor, пакування результатів.

Важкі залежності (torch, wandb) імпортуються лише всередині функцій.
"""
import csv
import glob
import os
import random
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results"
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = Path("/kaggle/working")
WANDB_PROJECT = "ucu-ai-hw2-detection"
TRAIN_LOG_COLUMNS = ["epoch", "train_loss", "val_mAP", "val_AP50", "minutes"]

SMOKE = False  # виставляє pipeline.configure
COMPUTED = []  # рани, пораховані в цій сесії; дописує pipeline.compute
FAILED = []  # рани, що впали в цій сесії; дописує pipeline.compute


def seed_everything(seed):
    """Фіксує random, numpy і torch (якщо він є)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """worker_init_fn для DataLoader: numpy і random у воркері від seed torch."""
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def smoke_subset(frames, max_clips=2, max_frames=64):
    """Перші max_clips кліпів і рівномірно до max_frames кадрів із них."""
    clips = sorted({f["clip"] for f in frames})[:max_clips]
    kept = [f for f in frames if f["clip"] in clips]
    if len(kept) <= max_frames:
        return kept
    picks = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
    return [kept[i] for i in picks]


def _find_in_runs(run_id, patterns, what, smoke):
    """Шукає файл рану: results/_smoke (у SMOKE), results, потім Kaggle Input."""
    smoke = SMOKE if smoke is None else smoke
    roots = ([RESULTS / "_smoke" / run_id] if smoke else []) + [RESULTS / run_id]
    searched = []
    for root in roots:
        for pattern in patterns:
            searched.append(str(root / pattern))
            found = sorted(glob.glob(glob.escape(str(root)) + "/" + pattern))
            if found:
                return Path(found[0])
    for pattern in patterns:
        wildcard = f"{KAGGLE_INPUT}/**/{run_id}/{pattern}"
        searched.append(wildcard)
        found = sorted(glob.glob(wildcard, recursive=True))
        if found:
            return Path(found[0])
    raise FileNotFoundError(f"{run_id}: не знайшов {what}. Шукав: {searched}")


def find_weights(run_id, smoke=None):
    """Шлях до weights/best.* рану. Dataset на Kaggle: <будь-де>/<run_id>/weights/best.*."""
    return _find_in_runs(run_id, ["weights/best.*", "best.*"], "ваги", smoke)


def find_preds(run_id, smoke=None):
    """Шлях до preds_val.json.gz рану (читати через data.load_preds)."""
    return _find_in_runs(run_id, ["preds_val.json.gz"], "передбачення", smoke)


def append_train_log(out_dir, epoch, train_loss, val_mAP=None, val_AP50=None, minutes=None):
    """Дописує рядок у train_log.csv одразу після епохи: падіння не втрачає криву."""
    path = Path(out_dir) / "train_log.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(TRAIN_LOG_COLUMNS)
        writer.writerow(["" if v is None else v for v in (epoch, train_loss, val_mAP, val_AP50, minutes)])


class _NoopRun:
    """Замінник W&B-рану: будь-який виклик нічого не робить."""

    def __init__(self):
        self.summary = {}
        self.config = {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _SafeRun:
    """W&B-ран, чий log() не може зупинити тренування."""

    def __init__(self, run):
        self._run = run

    def log(self, *args, **kwargs):
        try:
            self._run.log(*args, **kwargs)
        except Exception as e:
            print(f"wandb.log: {e}")

    def __getattr__(self, name):
        return getattr(self._run, name)


_wandb_active = None


def _wandb_key():
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        return None


def wandb_init(cfg, out_dir, tags=None):
    """Запускає W&B-ран. Без ключа, без пакета чи без мережі повертає no-op і не падає.

    Повертає об'єкт із .log(dict), .finish(), .summary. SMOKE-рани йдуть із префіксом
    _smoke_ і тегом smoke. Завершення: pipeline викликає wandb_finish() сам.
    """
    global _wandb_active
    wandb_finish()
    key = _wandb_key()
    if not key:
        print("W&B: ключа немає, логування вимкнене")
        return _NoopRun()
    out_dir = Path(out_dir)
    smoke = out_dir.parent.name == "_smoke"
    name = cfg.get("run_id") or out_dir.name
    try:
        import wandb

        os.environ["WANDB_API_KEY"] = key
        os.environ.setdefault("WANDB_SILENT", "true")
        run = wandb.init(
            project=cfg.get("wandb_project", WANDB_PROJECT),
            name=f"_smoke_{name}" if smoke else name,
            config=cfg,
            tags=list(tags or []) + (["smoke"] if smoke else []),
            dir=tempfile.gettempdir(),
            settings=wandb.Settings(console="off"),
        )
    except Exception as e:
        print(f"W&B вимкнений: {e}")
        return _NoopRun()
    _wandb_active = _SafeRun(run)
    return _wandb_active


def wandb_finish():
    """Завершує активний W&B-ран, якщо він є."""
    global _wandb_active
    run, _wandb_active = _wandb_active, None
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        print(f"wandb.finish: {e}")


class Predictor:
    """Контракт інференсу. Модуль моделі дає load_predictor(cfg, weights) -> Predictor."""

    def predict(self, images_bgr, conf=0.001):
        """Для кожного кадру (ndarray HxWx3 BGR, розміри можуть різнитись) повертає dict:
        boxes (Nx4 xyxy, пікселі оригіналу), scores (N), labels (N, класи 0-7)."""
        raise NotImplementedError


def _as_array(values, dtype):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def clean_prediction(pred, width, height, max_det=500):
    """Приводить вихід Predictor до норми: numpy, межі кадру, без порожніх рамок, топ max_det."""
    boxes = _as_array(pred["boxes"], np.float32).reshape(-1, 4).copy()
    scores = _as_array(pred["scores"], np.float32).reshape(-1)
    labels = _as_array(pred["labels"], np.int64).reshape(-1)
    if labels.size and (labels.min() < 0 or labels.max() > 7):
        raise ValueError(f"Класи мають бути 0-7, отримано {labels.min()}..{labels.max()}")
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    order = np.argsort(-scores, kind="stable")[:max_det]
    return {"boxes": boxes[order], "scores": scores[order], "labels": labels[order]}


def predict_frames(predictor, frames, conf, batch_size=8, max_det=500, workers=4):
    """Проганяє predictor по кадрах. Повертає preds: {image_id: {boxes, scores, labels}}."""
    from .data import read_image_bgr

    preds = {}
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            images = list(pool.map(read_image_bgr, [f["path"] for f in batch]))
            for frame, pred in zip(batch, predictor.predict(images, conf)):
                preds[frame["image_id"]] = clean_prediction(pred, frame["width"], frame["height"], max_det)
    return preds


def measure_fps(predictor, frames, conf, n=100):
    """Кадрів за секунду: по одному кадру в predict, без читання з диска."""
    from .data import read_image_bgr

    picked = frames[:: max(1, len(frames) // n)][:n]
    images = [read_image_bgr(f["path"]) for f in picked]
    predictor.predict(images[:1], conf)  # прогрів
    start = time.perf_counter()
    for image in images:
        predictor.predict([image], conf)
    return len(images) / (time.perf_counter() - start)


def pack_results(zip_name="light.zip"):
    """Пакує results/ без ваг у light.zip; розпаковувати у workspace/results/. Друкує упалі рани."""
    if FAILED:
        print(f"УВАГА, впали рани: {FAILED}")
    if not COMPUTED:
        print("pack_results: у цій сесії нічого не рахували")
        return None
    target_dir = KAGGLE_WORKING if KAGGLE_WORKING.exists() else WORKSPACE
    target = target_dir / zip_name
    partial = target_dir / f"{zip_name}.tmp"  # обрив під час запису не псує попередній архів
    skip_dirs = {"weights", "dataset", "wandb", "__pycache__"}
    skip_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".cache"}
    count = 0
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RESULTS.rglob("*")):
            rel = path.relative_to(RESULTS)
            if not path.is_file() or skip_dirs & set(rel.parts) or path.suffix in skip_suffixes:
                continue
            archive.write(path, rel.as_posix())
            count += 1
    os.replace(partial, target)
    print(f"{target}: {count} файлів, {target.stat().st_size / 1e6:.1f} MB")
    return target
