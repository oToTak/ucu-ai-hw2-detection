"""Runner `final` (T7): готова модель (чи кілька) плюс трюки інференсу з src.infer_tricks.

Не тренує. Конфіг:
  weights_from: <ран> або [<ран>, ...]   ваги через common.find_weights, у SMOKE із results/_smoke/
  model_weights: [1.0, ...]              (необов'язково) ваги моделей для WBF
  tricks: {tta: {...}, sahi: {...}, merge: {...}}   див. DEFAULT_* в infer_tricks.py
  val_full: true                         повний val (false: кожен val_step-й кадр)
Модель завантажується через load_predictor(cfg, weights) модуля `runner` того рану, з його власного
конфігу. Від конкретної моделі код не залежить, лише від контракту Predictor.

Довгий інференс (val і test) зберігає прогрес: _val_partial.json.gz і _test_partial.json.gz
разом зі state.json через common.save_progress, на старті common.load_progress.
"""
import importlib
import json
import time
from pathlib import Path

from . import common, data, evaluator, infer_tricks
from .submission import write_submission

VAL_PARTIAL = "_val_partial.json.gz"
TEST_PARTIAL = "_test_partial.json.gz"
CHUNK_BATCHES = 10  # кадрів між перевірками збереження = eval_batch * CHUNK_BATCHES


def _as_list(value):
    return [value] if isinstance(value, str) else list(value)


def load_predictors(cfg, smoke):
    """Predictor для кожного рану з weights_from. Повертає (predictors, base_cfgs)."""
    from . import pipeline  # імпорт тут: pipeline сам імпортує runner-и ліниво

    predictors, base_cfgs = [], []
    for run_id in _as_list(cfg["weights_from"]):
        base = pipeline.load_config(run_id)
        module = importlib.import_module(f".{base['runner']}", __package__)
        if not hasattr(module, "load_predictor"):
            raise AttributeError(f"{run_id}: модуль {module.__name__} не має load_predictor(cfg, weights)")
        weights = common.find_weights(run_id, smoke)
        print(f"{run_id}: ваги {weights}")
        predictors.append(module.load_predictor(base, weights))
        base_cfgs.append(base)
    return predictors, base_cfgs


def infer_resumable(predictor, frames, cfg, out_dir, partial_name, stage, elapsed, keep_files=()):
    """Передбачення на frames із відновленням. Проміжний результат у out_dir/partial_name не рідше
    ніж раз на ckpt_sync_minutes (Colab: разом із save_progress), наприкінці завжди повний.
    elapsed(): сумарні хвилини рану. Повертає preds {image_id: pred}."""
    out_dir = Path(out_dir)
    partial = out_dir / partial_name
    preds = data.load_preds(partial) if partial.exists() else {}
    todo = [f for f in frames if f["image_id"] not in preds]
    if len(todo) < len(frames):
        print(f"{stage}: відновлено {len(frames) - len(todo)} із {len(frames)} кадрів")
    files = [*keep_files, partial]
    chunk = cfg["eval_batch"] * CHUNK_BATCHES
    last_dump = time.time()

    def dump(force):
        data.save_preds(preds, partial)
        common.save_progress(
            out_dir, {"stage": stage, "minutes": elapsed(), "done": len(preds)}, files=files, force=force
        )

    for start in range(0, len(todo), chunk):
        preds.update(
            common.predict_frames(
                predictor, todo[start : start + chunk], cfg["conf"], cfg["eval_batch"], cfg["max_det"], cfg["num_workers"]
            )
        )
        print(f"{stage}: {len(preds)}/{len(frames)} кадрів, {elapsed():.1f} хв")
        if start + chunk < len(todo) and time.time() - last_dump >= cfg["ckpt_sync_minutes"] * 60:
            dump(False)
            last_dump = time.time()
    dump(True)  # повний набір завжди в партіал: наступний етап і відновлення беруть його звідси
    return preds


def _plot(result, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    values = [result.get(k) or 0.0 for k in keys]
    axes[0].bar(keys, values)
    axes[0].set_title("AP за розміром")
    axes[0].tick_params(axis="x", rotation=45)
    classes = [row["class"] for row in result["per_class"]]
    axes[1].bar(classes, [row["AP"] or 0.0 for row in result["per_class"]])
    axes[1].set_title("AP за класом")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    state = common.load_progress(out_dir) or {}
    minutes0, started = float(state.get("minutes", 0.0)), time.time()

    def elapsed():
        return minutes0 + (time.time() - started) / 60

    wb = common.wandb_init(cfg, out_dir, tags=["final"])
    predictors, base_cfgs = load_predictors(cfg, smoke)
    predictor = infer_tricks.TrickPredictor.from_config(
        predictors, cfg.get("tricks"), cfg["max_det"], cfg.get("model_weights")
    )
    description = predictor.describe()
    print(f"final: {description}")

    val_frames = data.split_frames(cfg, data_root, "val", smoke, full=cfg.get("val_full", True))
    val_preds = infer_resumable(predictor, val_frames, cfg, out_dir, VAL_PARTIAL, "val", elapsed)
    test_partial = out_dir / TEST_PARTIAL
    test_preds = None
    if cfg.get("predict_test"):
        test_frames = data.split_frames(cfg, data_root, "test", smoke)
        test_preds = infer_resumable(
            predictor, test_frames, cfg, out_dir, TEST_PARTIAL, "test", elapsed, keep_files=[out_dir / VAL_PARTIAL]
        )
        write_submission(test_preds, data.load_frames(data_root, "test"), out_dir / "submission.csv", cfg["max_det"])

    result = evaluator.Evaluator(val_frames, cfg["max_det"]).evaluate(val_preds)
    n_fps = cfg["smoke_limits"]["fps_frames"] if smoke else cfg["fps_frames"]
    fps = common.measure_fps(predictor, val_frames, cfg["conf"], n_fps)
    extra = {
        "weights_from": _as_list(cfg["weights_from"]),
        "tricks": description,
        "infer_minutes": elapsed(),
        "val_full": bool(cfg.get("val_full", True)),
    }
    metrics = evaluator.write_outputs(out_dir, result, fps, 0.0, 0, base_cfgs[0]["imgsz"], extra)
    data.save_preds(val_preds, out_dir / "preds_val.json.gz")
    _plot(result, f"{cfg['run_id']}: {description}", out_dir / "plots" / "final_ap.png")
    wb.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    # партіали більше не потрібні: повні результати лежать в preds_val.json.gz і submission.csv
    (out_dir / VAL_PARTIAL).unlink(missing_ok=True)
    test_partial.unlink(missing_ok=True)
    print(json.dumps(metrics, ensure_ascii=False))
