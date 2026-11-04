"""compute() рахує ран і пише результати, show() лише показує збережене."""
import contextlib
import glob
import importlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIGS = WORKSPACE / "configs"
RESULTS = WORKSPACE / "results"

_compute = []
_smoke = False


def configure(compute_list, smoke=False):
    """Запам'ятовує налаштування з першої клітинки ноутбука."""
    global _compute, _smoke
    missing = [run_id for run_id in compute_list if not (CONFIGS / f"{run_id}.yaml").exists()]
    if missing:
        raise ValueError(f"Немає конфігів для: {missing}")
    _compute, _smoke = list(compute_list), smoke


def find_data_root():
    """Дані: на Kaggle в /kaggle/input/..., локально у workspace/data."""
    local = WORKSPACE / "data"
    if (local / "test_images.csv").exists():
        return local
    found = glob.glob("/kaggle/input/**/test_images.csv", recursive=True)
    if found:
        return Path(found[0]).parent
    raise FileNotFoundError("Не знайшов дані: ні workspace/data, ні /kaggle/input")


def load_config(run_id):
    """Спільний протокол плюс налаштування конкретного рану."""
    with open(CONFIGS / "_protocol.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(CONFIGS / f"{run_id}.yaml", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


def _run_dir(run_id):
    return RESULTS / "_smoke" / run_id if _smoke else RESULTS / run_id


def _git_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=WORKSPACE, capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def _load_runner(cfg, run_id):
    name = f"src.{cfg['runner']}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            raise NotImplementedError(f"{run_id}: модуль {name} ще не написаний") from e
        raise


def compute(*run_ids):
    """Рахує лише ті рани, які є у списку COMPUTE."""
    for run_id in run_ids:
        if run_id not in _compute:
            continue
        cfg = load_config(run_id)
        runner = _load_runner(cfg, run_id)
        out_dir = _run_dir(run_id)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "plots").mkdir(parents=True)
        shutil.copy(CONFIGS / f"{run_id}.yaml", out_dir / "config.yaml")
        meta = {
            "run": run_id,
            "git": _git_hash(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "smoke": _smoke,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with open(out_dir / "log.txt", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            runner.run(cfg, out_dir, find_data_root(), _smoke)
        print(f"{run_id}: готово → {out_dir}")


def show(*run_ids):
    """Показує метрики й графіки зі збережених результатів."""
    try:
        from IPython.display import Image, display
    except ImportError:
        Image, display = None, None
    for run_id in run_ids:
        out_dir = _run_dir(run_id)
        metrics = out_dir / "metrics.json"
        plots = sorted((out_dir / "plots").glob("*.png"))
        if not metrics.exists() and not plots:
            print(f"{run_id}: ще немає результатів")
            continue
        print(f"=== {run_id}")
        if metrics.exists():
            print(json.dumps(json.loads(metrics.read_text(encoding="utf-8")), ensure_ascii=False))
        for plot in plots:
            if display:
                display(Image(filename=str(plot)))
            else:
                print(plot)
