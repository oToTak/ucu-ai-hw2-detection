"""compute() рахує ран і пише результати, show() лише показує збережене."""
import contextlib
import fnmatch
import functools
import glob
import importlib
import json
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import common

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIGS = WORKSPACE / "configs"
RESULTS = WORKSPACE / "results"
LOG_TAIL_LINES = 15

_compute = []
_smoke = False


def _expand(patterns):
    """Шаблони fnmatch ("1?_*") у список ранів; конфіги _* не рани."""
    runs = sorted(p.stem for p in CONFIGS.glob("*.yaml") if not p.stem.startswith("_"))
    expanded = []
    for pattern in patterns:
        expanded += [run for run in runs if fnmatch.fnmatchcase(run, pattern) and run not in expanded]
    return expanded


def configure(compute_list, smoke=False):
    """Запам'ятовує налаштування з першої клітинки ноутбука."""
    global _compute, _smoke
    missing = [pattern for pattern in compute_list if not _expand([pattern])]
    if missing:
        raise ValueError(f"Немає конфігів для: {missing}")
    _compute, _smoke = _expand(compute_list), smoke
    common.SMOKE = smoke


@functools.lru_cache(maxsize=1)
def find_data_root():
    """Дані: на Kaggle в /kaggle/input/..., локально у workspace/data."""
    local = WORKSPACE / "data"
    if (local / "test_images.csv").exists():
        return local
    found = glob.glob("/kaggle/input/**/test_images.csv", recursive=True)
    if found:
        return Path(found[0]).parent
    raise FileNotFoundError("Не знайшов дані: ні workspace/data, ні /kaggle/input")


def _read_yaml(name):
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base, override):
    """Глибоке злиття словників; решта значень, зокрема списки, замінюється."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_chain(run_id, seen=()):
    """Конфіг рану разом з його ланцюжком extends."""
    if run_id in seen:
        raise ValueError(f"Цикл у extends: {[*seen, run_id]}")
    cfg = _read_yaml(run_id)
    parent = cfg.pop("extends", None)
    return _merge(_load_chain(parent, (*seen, run_id)), cfg) if parent else cfg


def load_config(run_id):
    """Спільний протокол, зверху ланцюжок extends і налаштування рану, плюс run_id."""
    cfg = _merge(_read_yaml("_protocol"), _load_chain(run_id))
    cfg["run_id"] = run_id
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


def _print_log_tail(log_path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- хвіст {log_path}:")
    print("\n".join(lines[-LOG_TAIL_LINES:]))


def _pack_safely():
    """Архів після кожного рану; його збій не має зупиняти решту сесії."""
    try:
        common.pack_results()
    except Exception as e:
        print(f"pack_results не вдався: {e}")


def _on_run_failed(run_id):
    """Падіння рану: архів готового і перехід до наступного рану. У SMOKE падаємо одразу."""
    common.FAILED.append(run_id)
    print(f"{run_id}: ВПАВ")
    _pack_safely()
    if _smoke:
        raise


def compute(*patterns):
    """Рахує лише ті рани, які є у списку COMPUTE. Приймає імена й шаблони.

    Без SMOKE упалий ран не зупиняє сесію: хвіст логу друкується, ран іде в common.FAILED,
    light.zip оновлюється після кожного рану. Ctrl+C і подібне зупиняють сесію завжди.
    """
    for run_id in _expand(patterns):
        if run_id not in _compute:
            continue
        try:
            cfg = load_config(run_id)
            runner = _load_runner(cfg, run_id)
        except Exception:  # збережені результати цього рану лишаються недоторканими
            traceback.print_exc()
            _on_run_failed(run_id)
            continue
        out_dir = _run_dir(run_id)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "plots").mkdir(parents=True)
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        meta = {
            "run": run_id,
            "git": _git_hash(),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "smoke": _smoke,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        common.COMPUTED.append(run_id)
        log_path = out_dir / "log.txt"
        try:
            with open(log_path, "w", buffering=1, encoding="utf-8") as log, contextlib.redirect_stdout(log):
                try:
                    common.seed_everything(cfg["seed"])
                    runner.run(cfg, out_dir, find_data_root(), _smoke)
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
                finally:
                    common.wandb_finish()
        except Exception:
            _print_log_tail(log_path)
            _on_run_failed(run_id)
            continue
        except BaseException:
            _print_log_tail(log_path)
            raise
        print(f"{run_id}: готово → {out_dir}")
        _pack_safely()


def show(*patterns):
    """Показує метрики й графіки зі збережених результатів. Приймає імена й шаблони."""
    try:
        from IPython.display import Image, display
    except ImportError:
        Image, display = None, None
    for run_id in _expand(patterns):
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
