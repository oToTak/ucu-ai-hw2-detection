"""compute() рахує ран і пише результати, show() лише показує збережене."""
import contextlib
import fnmatch
import functools
import glob
import importlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import common

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIGS = WORKSPACE / "configs"
RESULTS = common.RESULTS
LOG_TAIL_LINES = 15
ALWAYS_RECOMPUTE = {"compare", "summary", "wandb_backfill"}  # ці ранери зводять чужі результати
FAILED_LOGS = ("log.txt", "meta.json", "config.yaml", "train_log.csv")

_compute = []
_smoke = False
_force = set()


def _expand(patterns):
    """Шаблони fnmatch ("1?_*") у список ранів; конфіги _* не рани."""
    runs = sorted(p.stem for p in CONFIGS.glob("*.yaml") if not p.stem.startswith("_"))
    expanded = []
    for pattern in patterns:
        expanded += [run for run in runs if fnmatch.fnmatchcase(run, pattern) and run not in expanded]
    return expanded


def configure(compute_list, smoke=False, force=False):
    """Запам'ятовує налаштування з першої клітинки ноутбука.

    force: True або список шаблонів; ці рани рахуються заново, без пропуску й відновлення.
    """
    global _compute, _smoke, _force
    missing = [pattern for pattern in compute_list if not _expand([pattern])]
    if missing:
        raise ValueError(f"Немає конфігів для: {missing}")
    _compute, _smoke = _expand(compute_list), smoke
    _force = set(_compute) if force is True else set(_expand(force or []))
    common.SMOKE = smoke


@functools.lru_cache(maxsize=1)
def find_data_root():
    """Дані: HW2_DATA (Colab), на Kaggle в /kaggle/input/..., локально у workspace/data."""
    override = os.environ.get("HW2_DATA")
    if override:
        if not (Path(override) / "test_images.csv").exists():
            raise FileNotFoundError(f"HW2_DATA={override}: там немає test_images.csv")
        return Path(override)
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
    if "num_workers" in cfg:  # на Colab vCPU мало
        cfg["num_workers"] = min(cfg["num_workers"], os.cpu_count() or 1)
    return cfg


def _run_dir(run_id):
    return common.SMOKE_RESULTS / run_id if _smoke else RESULTS / run_id


def _work_dir(run_id):
    """Де ран рахується. З HW2_WORK (Colab) це локальний диск, у RESULTS він їде лише готовим."""
    work = os.environ.get("HW2_WORK")
    return Path(work) / run_id if work and not _smoke else _run_dir(run_id)


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


def _already_done(cfg, run_id):
    """Colab: готовий ран пропускається, щоб Run All після дисконекту йшов далі.

    Лише з HW2_WIP: на Kaggle й локально явний COMPUTE рахує заново, як і раніше.
    """
    if common.WIP is None or _smoke or run_id in _force or cfg["runner"] in ALWAYS_RECOMPUTE:
        return False
    return (RESULTS / run_id / "metrics.json").exists()


def _prepare_out_dir(run_id, cfg, out_dir, resumed):
    """Відновлений ран зберігає свої файли, новий починає з чистої теки."""
    config_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    meta_path = out_dir / "meta.json"
    meta = {
        "run": run_id,
        "git": _git_hash(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": _smoke,
    }
    if resumed:
        config_path = out_dir / "config.yaml"
        if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
            print(f"{run_id}: УВАГА, конфіг змінився після збереження; перерахунок заново: force")
        saved = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta["started"] = saved.get("started", meta["started"])
        meta["resumes"] = saved.get("resumes", 0) + 1
    elif out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _prepare_run(run_id):
    """Готує ран. Повертає (cfg, runner, out_dir, resumed) або None, якщо він уже готовий."""
    cfg = load_config(run_id)
    runner = _load_runner(cfg, run_id)
    if _already_done(cfg, run_id):
        print(f"{run_id}: уже пораховано, пропускаю ({RESULTS / run_id})")
        common.clear_wip(run_id)
        common.COMPUTED.append(run_id)
        return None
    if run_id in _force:
        common.clear_wip(run_id)
    out_dir = _work_dir(run_id)
    common.progress_begin(cfg)
    resumed = common.restore_wip(run_id, out_dir)
    if resumed:
        print(f"{run_id}: відновлюю з {common.wip_dir(run_id)}")
    _prepare_out_dir(run_id, cfg, out_dir, resumed)
    common.COMPUTED.append(run_id)
    return cfg, runner, out_dir, resumed


def _publish(out_dir, run_id):
    """Готовий ран їде в RESULTS цілим: копія в _partial_, потім перейменування."""
    private = common.progress_private_files()

    def skip(rel):
        return rel.as_posix() in private or rel.parts[0] == "dataset"

    final, partial = RESULTS / run_id, RESULTS / f"_partial_{run_id}"
    shutil.rmtree(partial, ignore_errors=True)
    common.copy_tree(out_dir, partial, skip)
    if final.exists():
        shutil.rmtree(final)
    os.replace(partial, final)
    common.clear_wip(run_id)
    shutil.rmtree(RESULTS / "_failed" / run_id, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)


def _save_failed_logs(out_dir, run_id):
    """Логи упалого рану в RESULTS/_failed: метрик там немає, тож ран не вважається готовим."""
    target = RESULTS / "_failed" / run_id
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in FAILED_LOGS:
            if (out_dir / name).exists():
                shutil.copyfile(out_dir / name, target / name)
    except OSError as e:
        print(f"{run_id}: логи не збережено: {e}")


def compute(*patterns):
    """Рахує лише ті рани, які є у списку COMPUTE. Приймає імена й шаблони.

    Без SMOKE упалий ран не зупиняє сесію: хвіст логу друкується, ран іде в common.FAILED,
    light.zip оновлюється після кожного рану. Ctrl+C і подібне зупиняють сесію завжди.
    На Colab ран рахується локально, а wip/<ран> дає відновлення посеред рану.
    """
    for run_id in _expand(patterns):
        if run_id not in _compute:
            continue
        try:
            prepared = _prepare_run(run_id)
        except Exception:  # збережені результати цього рану лишаються недоторканими
            traceback.print_exc()
            _on_run_failed(run_id)
            continue
        if prepared is None:
            continue
        cfg, runner, out_dir, resumed = prepared
        staged = out_dir != _run_dir(run_id)
        log_path = out_dir / "log.txt"
        log_mode = "a" if resumed else "w"
        try:
            with open(log_path, log_mode, buffering=1, encoding="utf-8") as log, contextlib.redirect_stdout(log):
                try:
                    common.seed_everything(cfg["seed"])
                    runner.run(cfg, out_dir, find_data_root(), _smoke)
                    if _smoke:
                        common.progress_selfcheck(out_dir)
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
                finally:
                    common.wandb_finish()
                    common.wait_progress()
        except Exception:
            _print_log_tail(log_path)
            if staged:
                _save_failed_logs(out_dir, run_id)
            _on_run_failed(run_id)
            continue
        except BaseException:
            _print_log_tail(log_path)
            raise
        if staged:
            try:
                _publish(out_dir, run_id)
            except Exception:
                traceback.print_exc()
                _on_run_failed(run_id)
                continue
        print(f"{run_id}: готово → {_run_dir(run_id)}")
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
