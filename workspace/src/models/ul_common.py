"""Спільна обгортка Ultralytics (YOLO, RT-DETR). Основа для T2 (baseline), T5 (YOLO), T6 (RT-DETR).

Модель задає конфіг: `model: yolo11n.pt` або `rtdetr-l.pt` (клас обирається за назвою).
Усі гіперпараметри лише в yaml: ключі верхнього рівня нижче плюс `train_args` (довільні
аргументи model.train, перекривають усе) і `predict_args` (аргументи model.predict).

Ключі конфігу (на додачу до _protocol.yaml):
    model, epochs, batch, device (None = авто), train_args, predict_args.

Оцінка (кожну епоху на val_step і наприкінці на повному val) лише через src.evaluator.
Нативний mAP Ultralytics лише логується (`mAP_native` у metrics.json). best.pt
вибирається за нашим val_mAP: копія last.pt у out_dir/weights/best.pt, коли mAP зріс.

Рант-модуль (baseline.py, yolo.py, rtdetr.py) має лише:
    def run(cfg, out_dir, data_root, smoke): ul_common.run(cfg, out_dir, data_root, smoke)
    def load_predictor(cfg, weights): return ul_common.load_predictor(cfg, weights)
"""

import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .. import common, data, evaluator


def _model_class(name):
    """RT-DETR має свій клас; усе інше (yolo11*.pt, yolov8*.pt, свій .yaml) іде в YOLO."""
    import ultralytics

    return ultralytics.RTDETR if "rtdetr" in Path(str(name)).name.lower() else ultralytics.YOLO


class UltralyticsPredictor(common.Predictor):
    """Predictor над будь-якою моделлю Ultralytics. Рамки повертає в пікселях оригіналу, класи 0-7."""

    def __init__(self, model_name, weights, imgsz, max_det=500, device=None, predict_args=None):
        self.model = _model_class(model_name)(str(weights))
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.predict_args = dict(predict_args or {})

    def predict(self, images_bgr, conf=0.001):
        results = self.model.predict(
            list(images_bgr),
            conf=conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            **self.predict_args,
        )
        out = []
        for r in results:
            boxes = r.boxes
            out.append(
                {
                    "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4),
                    "scores": boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1),
                    "labels": boxes.cls.detach().cpu().numpy().astype(np.int64).reshape(-1),
                }
            )
        return out


def load_predictor(cfg, weights):
    return UltralyticsPredictor(
        cfg["model"], weights, cfg["imgsz"], cfg["max_det"], cfg.get("device"), cfg.get("predict_args")
    )


def _attach_log_file(out_dir):
    """Логер Ultralytics пише в ноутбук (pipeline імпортує runner до перенаправлення stdout): дублюємо у файл."""
    try:
        from ultralytics.utils import LOGGER

        handler = logging.FileHandler(Path(out_dir) / "ultralytics.log", encoding="utf-8")
        LOGGER.addHandler(handler)
        # Ultralytics імпортується вперше всередині redirect_stdout рану: його StreamHandler
        # тримає stdout першого рану й після закриття log.txt друкує traceback. Перебиваємо на живий stdout.
        for h in LOGGER.handlers:
            if type(h) is logging.StreamHandler:
                h.stream = sys.stdout
        return handler
    except Exception as e:  # лог не варте падіння рану
        print(f"ultralytics.log: {e}")
        return None


def _plot_curves(out_dir):
    """plots/train_curves.png: train_loss і val mAP/AP50 по епохах з train_log.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(Path(out_dir) / "train_log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(epochs, [float(r["train_loss"]) for r in rows], marker="o")
    a.set(title="train loss", xlabel="epoch")
    for key in ("val_mAP", "val_AP50"):
        b.plot(epochs, [float(r[key]) if r[key] != "" else np.nan for r in rows], marker="o", label=key)
    b.set(title="val (evaluator, val_step)", xlabel="epoch")
    b.legend()
    fig.tight_layout()
    (Path(out_dir) / "plots").mkdir(exist_ok=True)
    fig.savefig(Path(out_dir) / "plots" / "train_curves.png", dpi=120)
    plt.close(fig)


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_path = weights_dir / "best.pt"
    log_handler = _attach_log_file(out_dir)

    from ultralytics import settings
    from ultralytics.utils.torch_utils import strip_optimizer

    try:
        settings.update({k: False for k in ("wandb", "raytune", "mlflow", "comet", "clearml", "dvc")})  # свій W&B через common.wandb_init
    except Exception as e:
        print(f"ultralytics settings: {e}")
    wb = common.wandb_init(cfg, out_dir, tags=["ultralytics", str(cfg["model"])])

    data_yaml = data.build_ultralytics_dataset(cfg, out_dir, data_root, smoke)
    val_frames = data.split_frames(cfg, data_root, "val", smoke)
    val_evaluator = evaluator.Evaluator(val_frames, cfg["max_det"])
    epochs = cfg["smoke_limits"]["epochs"] if smoke else cfg["epochs"]

    # Відновлення після дисконекту (Colab): load_progress дає стан останнього чекпоінта або None.
    # Той самий out_dir, data.yaml і project/name на новій VM, тож Ultralytics resume знаходить усе сам.
    last_rel = Path("dataset/ultralytics/train/weights/last.pt")
    saved = common.load_progress(out_dir)
    stage = saved.get("stage") if saved else None
    if stage == "trained" and not best_path.exists():
        stage = None
    if stage == "train" and not (out_dir / last_rel).exists():
        stage = None
    if stage is None:
        saved = None
        (out_dir / "train_log.csv").unlink(missing_ok=True)  # старт з нуля: старого логу не лишаємо
    minutes_before = float(saved.get("minutes", 0.0)) if saved else 0.0
    state = {
        "epochs_done": int(saved.get("epoch", 0)) if saved else 0,
        "best_mAP": float(saved.get("best_mAP", -1.0)) if saved else -1.0,
        "native": None,
        "finished": stage == "trained",
        "prev": minutes_before,
    }
    start = time.time() - minutes_before * 60  # хвилини накопичуються між сесіями
    print(f"resume: stage={stage} state={saved}")

    def on_fit_epoch_end(trainer):
        # Після останньої епохи (або ранньої зупинки) Ultralytics кличе колбек ще раз із final_eval: пропускаємо.
        if state["finished"]:
            return
        if trainer.epoch + 1 <= state["epochs_done"]:  # повтор епохи: рядок у train_log.csv вже є
            return
        import torch

        tloss = getattr(trainer, "tloss", None)  # у 8.4.203 це dict {назва: значення}
        loss = sum(float(v) for v in tloss.values()) if tloss else float("nan")
        state["native"] = (trainer.metrics or {}).get("metrics/mAP50-95(B)")
        predictor = load_predictor(cfg, trainer.last)
        preds = common.predict_frames(
            predictor, val_frames, cfg["conf"], cfg["eval_batch"], cfg["max_det"], cfg["num_workers"]
        )
        result = val_evaluator.evaluate(preds)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        minutes = (time.time() - start) / 60
        epoch = trainer.epoch + 1
        common.append_train_log(out_dir, epoch, loss, result["mAP"], result["AP50"], minutes)
        wb.log({"epoch": epoch, "train_loss": loss, "val_mAP": result["mAP"], "val_AP50": result["AP50"],
                "val_mAP_native": state["native"]})
        state["epochs_done"] = epoch
        if result["mAP"] > state["best_mAP"]:
            state["best_mAP"] = result["mAP"]
            shutil.copyfile(trainer.last, best_path)
        # Бюджет: Ultralytics його не знає. Зупиняємось, якщо наступна епоха не вкладається
        # у budget_minutes мінус резерв на повний val, fps і test. Читається після колбека.
        over = not smoke and minutes + (minutes - state["prev"]) > cfg["budget_minutes"] - cfg.get("reserve_minutes", 12)
        state["prev"] = minutes
        if over:
            trainer.stop = True
        state["finished"] = bool(trainer.stop) or epoch >= trainer.epochs
        # Прогрес для Colab (поза Colab no-op), після append_train_log. Зміна етапу: force.
        files = [rel for rel in (last_rel, best_path.relative_to(out_dir)) if (out_dir / rel).exists()]
        common.save_progress(
            out_dir,
            {"epoch": epoch, "minutes": minutes, "stage": "trained" if state["finished"] else "train",
             "best_mAP": state["best_mAP"]},
            files=files,
            force=state["finished"],
        )

    resuming = stage == "train"
    model = _model_class(cfg["model"])(str(out_dir / last_rel) if resuming else cfg["model"])
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    train_args = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["num_workers"],
        seed=cfg["seed"],
        deterministic=False,
        device=cfg.get("device"),
        project=str(out_dir / "dataset" / "ultralytics"),  # dataset/ не йде в git і light.zip
        name="train",
        exist_ok=True,
        plots=False,
        val=True,
    )
    train_args.update(cfg.get("train_args") or {})
    if resuming:
        # Ultralytics бере решту аргументів з чекпоінта (епохи, imgsz, project/name: ті самі шляхи); дозволені змінні:
        model.train(resume=True, data=str(data_yaml), device=train_args["device"], workers=train_args["workers"])
    elif stage != "trained":  # "trained": навчання вже завершене, одразу фінальна оцінка
        model.train(**train_args)
    train_minutes = (time.time() - start) / 60

    strip_optimizer(best_path)  # fp16 без оптимізатора
    metrics = evaluator.finish_run(
        load_predictor(cfg, best_path),
        cfg,
        out_dir,
        data_root,
        smoke,
        train_minutes=train_minutes,
        epochs=state["epochs_done"],
        extra={"model": str(cfg["model"]), "mAP_native_last": state["native"], "best_val_mAP_step": state["best_mAP"]},
    )
    _plot_curves(out_dir)
    wb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    if log_handler is not None:
        logging.getLogger("ultralytics").removeHandler(log_handler)
        log_handler.close()
