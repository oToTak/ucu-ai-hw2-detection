"""Спільне для всіх ранів: seed, smoke, ваги, W&B, Predictor, відновлення, пакування результатів.

Важкі залежності (torch, wandb) імпортуються лише всередині функцій.
"""
import csv
import filecmp
import glob
import json
import os
import random
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
# Env виставляє colab.setup() до першого імпорту цього модуля. Без env усе як раніше.
RESULTS = Path(os.environ.get("HW2_RESULTS") or WORKSPACE / "results")
SMOKE_RESULTS = Path(os.environ.get("HW2_SMOKE_RESULTS") or RESULTS / "_smoke")
WIP = Path(os.environ["HW2_WIP"]) if os.environ.get("HW2_WIP") else None  # None: відновлення вимкнене
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = Path("/kaggle/working")
WANDB_PROJECT = "ucu-ai-hw2-detection"
TRAIN_LOG_COLUMNS = ["epoch", "train_loss", "val_mAP", "val_AP50", "minutes"]

SMOKE = False  # виставляє pipeline.configure
COMPUTED = []  # рани, пораховані чи пропущені як готові в цій сесії; дописує pipeline.compute
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
    """Шукає файл рану: SMOKE_RESULTS (у SMOKE), RESULTS, потім Kaggle Input."""
    smoke = SMOKE if smoke is None else smoke
    roots = ([SMOKE_RESULTS / run_id] if smoke else []) + [RESULTS / run_id]
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
        pass
    try:
        from google.colab import userdata

        return userdata.get("WANDB_API_KEY")
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


STATE_FILE = "state.json"  # останній стан; на Drive щоразу
CKPT_STATE_FILE = "ckpt_state.json"  # стан, що відповідає чекпоінту на Drive; пишеться після нього
SMALL_PROGRESS_FILES = ("train_log.csv", "meta.json", "config.yaml", "log.txt")

_progress = {"sync_minutes": 15, "last_sync": None, "thread": None, "files": set()}


def wip_dir(run_id):
    """Постійний каталог незавершеного рану: HUB/wip/<ран>; у SMOKE на Colab локальний _smoke/_wip/<ран>."""
    if WIP is None:
        return None
    return SMOKE_RESULTS / "_wip" / run_id if SMOKE else WIP / run_id


def _copy_synced(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        try:
            os.fsync(dst.fileno())
        except OSError:
            pass


def _copy_atomic(source, target):
    """Копія в *.tmp, потім os.replace: обрив посеред копіювання не лишає битий файл."""
    partial = target.with_name(target.name + ".tmp")
    _copy_synced(source, partial)
    os.replace(partial, target)


def _write_json_atomic(path, payload):
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(payload, default=float), encoding="utf-8")
    os.replace(partial, path)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def copy_tree(source, target, skip=None):
    """Копіює дерево файлів без copystat (Drive може його не прийняти). skip(rel: Path) -> bool."""
    source, target = Path(source), Path(target)
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_file() and not (skip and skip(rel)):
            _copy_synced(path, target / rel)


def progress_begin(cfg):
    """Скидає стан механізму перед раном. Кличе pipeline."""
    _progress.update(sync_minutes=cfg.get("ckpt_sync_minutes", 15), last_sync=None, files=set())


def wait_progress():
    """Чекає на фонове копіювання чекпоінта. Pipeline кличе його перед виходом із рану."""
    thread = _progress["thread"]
    if thread is not None:
        thread.join()
    _progress["thread"] = None


def _sync_checkpoint(snapshot, wip, rels, payload):
    """Спершу всі файли в *.tmp, потім швидка заміна, і лише тоді стан чекпоінта."""
    try:
        for rel in rels:
            _copy_synced(snapshot / rel, wip / f"{rel}.tmp")
        for rel in rels:
            os.replace(wip / f"{rel}.tmp", wip / rel)
        _write_json_atomic(wip / CKPT_STATE_FILE, payload)
    except OSError as e:
        print(f"save_progress: чекпоінт не скопійовано: {e}")


def _start_checkpoint_sync(out_dir, wip, rels, state):
    wait_progress()  # наступне копіювання чекає завершення попереднього
    # Знімок на локальному диску: ранер може перезаписати файл, поки він їде на Drive.
    snapshot = Path(tempfile.gettempdir()) / "hw2_snap" / out_dir.name
    sizes = {}
    for rel in rels:
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / rel, snapshot / rel)
        sizes[rel] = (snapshot / rel).stat().st_size
    payload = {**state, "_files": rels, "_sizes": sizes}
    _progress["last_sync"] = time.monotonic()
    _progress["thread"] = threading.Thread(
        target=_sync_checkpoint, args=(snapshot, wip, rels, payload), daemon=True
    )
    _progress["thread"].start()


def save_progress(out_dir, state, files=(), force=False):
    """Зберігає прогрес рану. Поза Colab no-op. Помилка Drive лише друкується, ран не падає.

    state: JSON-сумісний dict (домовленість: epoch, minutes, stage; ключі на "_" зайняті).
    files: файли чекпоінта всередині out_dir. Щоразу на Drive йдуть state.json, train_log.csv,
    meta.json, config.yaml, log.txt; чекпоінт у фоні й не частіше ніж раз на ckpt_sync_minutes
    (force=True: негайно, для зміни етапу). Файли перезаписуються на місці, історії немає.
    """
    out_dir = Path(out_dir)
    wip = wip_dir(out_dir.name)
    if wip is None:
        return
    rels = []
    for path in files:
        path = Path(path)
        rels.append((path if path.is_absolute() else out_dir / path).relative_to(out_dir).as_posix())
    _progress["files"].update(rels)
    state = dict(state)
    try:
        wip.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(wip / STATE_FILE, {**state, "_files": sorted(_progress["files"])})
        for name in SMALL_PROGRESS_FILES:
            if (out_dir / name).exists():
                _copy_atomic(out_dir / name, wip / name)
        last = _progress["last_sync"]
        due = last is None or time.monotonic() - last >= _progress["sync_minutes"] * 60
        if rels and (force or due):
            _start_checkpoint_sync(out_dir, wip, rels, state)
    except OSError as e:
        print(f"save_progress: {e}")


def _trim_train_log(path, epoch):
    """Прибирає епохи новіші за чекпоінт: їх перерахують, дублів у кривій не буде."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    kept = rows[:1] + [row for row in rows[1:] if row and float(row[0]) <= epoch]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)


def _discard_progress(out_dir, reason):
    print(f"load_progress: починаю спочатку, {reason}")
    for name in (STATE_FILE, CKPT_STATE_FILE, "train_log.csv"):
        (out_dir / name).unlink(missing_ok=True)


def load_progress(out_dir):
    """Стан, з якого продовжувати, або None (починати спочатку). Поза Colab завжди None.

    Якщо ран зберігав чекпоінти, це стан саме останнього чекпоінта, що доїхав на Drive,
    а не найновіший state.json. train_log.csv обрізається до його епохи.
    """
    if WIP is None:
        return None
    out_dir = Path(out_dir)
    latest = _read_json(out_dir / STATE_FILE)
    if latest is None:
        return None
    state = latest
    if latest.get("_files"):
        state = _read_json(out_dir / CKPT_STATE_FILE)
        if state is None:
            _discard_progress(out_dir, "чекпоінт не встиг доїхати на Drive")
            return None
        for rel, size in state["_sizes"].items():
            path = out_dir / rel
            if not path.exists() or path.stat().st_size != size:
                _discard_progress(out_dir, f"чекпоінт {rel} неповний")
                return None
    state = {key: value for key, value in state.items() if not key.startswith("_")}
    if "epoch" in state:
        _trim_train_log(out_dir / "train_log.csv", state["epoch"])
    return state


def restore_wip(run_id, out_dir):
    """Копіює wip/<ран> в out_dir, нічого не стираючи. True, якщо є що відновлювати."""
    wip = wip_dir(run_id)
    if wip is None:
        return False
    latest = None if SMOKE else _read_json(wip / STATE_FILE)
    if latest is None:
        clear_wip(run_id)
        return False
    copy_tree(wip, out_dir, skip=lambda rel: rel.name.endswith(".tmp"))
    _progress["files"].update(latest.get("_files", []))
    return True


def clear_wip(run_id):
    wip = wip_dir(run_id)
    if wip is not None:
        shutil.rmtree(wip, ignore_errors=True)


def progress_private_files():
    """Службові файли рану, що не йдуть у RESULTS: стан і чекпоінти, крім best.*."""
    return {STATE_FILE, CKPT_STATE_FILE} | {
        rel for rel in _progress["files"] if not Path(rel).name.startswith("best.")
    }


def progress_selfcheck(out_dir):
    """SMOKE на Colab: один раз зберігає чекпоінт у _smoke/_wip і відтворює його. Інакше no-op."""
    out_dir = Path(out_dir)
    if WIP is None or not SMOKE:
        return
    source = (sorted((out_dir / "weights").glob("best.*")) or [out_dir / "config.yaml"])[0]
    state = {"epoch": 0, "minutes": 0.0, "stage": "selfcheck"}
    save_progress(out_dir, state, [source], force=True)
    wait_progress()
    with tempfile.TemporaryDirectory() as tmp:
        restored = Path(tmp) / out_dir.name
        copy_tree(wip_dir(out_dir.name), restored)
        same = load_progress(restored) == state and filecmp.cmp(
            source, restored / source.relative_to(out_dir), shallow=False
        )
    clear_wip(out_dir.name)
    if not same:
        raise RuntimeError("Механізм відновлення не відтворив чекпоінт")
    print(f"Відновлення перевірено на {source.name}")


def pack_results(zip_name="light.zip"):
    """Пакує results/ без ваг у light.zip; розпаковувати у workspace/results/. Друкує упалі рани.

    З HW2_PACK_DIR (Colab) архів іде туди й без важких передбачень; у SMOKE окремий
    light_smoke.zip лише зі _smoke.
    """
    if FAILED:
        print(f"УВАГА, впали рани: {FAILED}")
    if not COMPUTED:
        print("pack_results: у цій сесії нічого не рахували")
        return None
    pack_dir = os.environ.get("HW2_PACK_DIR")
    target_dir = Path(pack_dir) if pack_dir else KAGGLE_WORKING if KAGGLE_WORKING.exists() else WORKSPACE
    root, prefix = RESULTS, ""
    if pack_dir and SMOKE:
        root, prefix, zip_name = SMOKE_RESULTS, "_smoke/", "light_smoke.zip"
    target = target_dir / zip_name
    partial = target_dir / f"{zip_name}.tmp"  # обрив під час запису не псує попередній архів
    skip_dirs = {"weights", "dataset", "wandb", "__pycache__", "_wip"}
    skip_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".cache"}
    count = 0
    with tempfile.TemporaryDirectory() as tmp:  # локально: zip робить seek, Drive цього не любить
        local = Path(tmp) / zip_name
        with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                rel = path.relative_to(root)
                if not path.is_file() or skip_dirs & set(rel.parts) or path.suffix in skip_suffixes:
                    continue
                if rel.parts[0].startswith("_partial_"):
                    continue
                if pack_dir and path.name.startswith(("preds", "submission")):
                    continue
                archive.write(path, prefix + rel.as_posix())
                count += 1
        shutil.copyfile(local, partial)
    os.replace(partial, target)
    print(f"{target}: {count} файлів, {target.stat().st_size / 1e6:.1f} MB")
    return target
