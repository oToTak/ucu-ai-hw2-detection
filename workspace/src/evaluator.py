"""Єдиний оцінювач усіх детекторів: COCO mAP у пікселях оригінального кадру.

Клас 0-7 зовні; category_id 1-8 живе лише тут. Ignore-регіони йдуть як iscrowd=1 у кожній
категорії, тож детекції в них не штрафуються. COCO рахує maxDets на категорію, а умова дає
500 на кадр, тому кадр спершу обрізається до max_det найкращих за score.
"""
import csv
import json
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from .common import measure_fps, predict_frames
from .data import CLASS_NAMES, NUM_CLASSES, save_preds, split_frames

SIZE_NAMES = ["small", "medium", "large"]
SMALL_MAX, MEDIUM_MAX = 32**2, 96**2


def _mean(values):
    """Середнє без -1 (COCO так позначає «немає GT»). Немає нічого: nan."""
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def _xyxy_to_xywh(boxes):
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4).copy()
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


class Evaluator:
    """Тримає GT для набору кадрів; evaluate(preds) можна кликати багато разів (кожну епоху)."""

    def __init__(self, frames, max_det=500):
        self.frames = frames
        self.max_det = max_det
        self._ids = {f["image_id"]: i + 1 for i, f in enumerate(frames)}
        self._coco = COCO(self._gt_dataset())
        areas = np.concatenate(
            [(f["boxes"][:, 2:] - f["boxes"][:, :2]).prod(axis=1) for f in frames] or [np.zeros(0)]
        )
        labels = np.concatenate([f["labels"] for f in frames] or [np.zeros(0, np.int64)])
        self._n_gt_class = np.bincount(labels, minlength=NUM_CLASSES)
        self._n_gt_size = [
            int((areas <= SMALL_MAX).sum()),
            int(((areas > SMALL_MAX) & (areas <= MEDIUM_MAX)).sum()),
            int((areas > MEDIUM_MAX).sum()),
        ]

    def _gt_dataset(self):
        images, annotations = [], []

        def add(image_id, category_id, xywh, iscrowd):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": iscrowd,
                }
            )

        for frame in self.frames:
            image_id = self._ids[frame["image_id"]]
            images.append({"id": image_id, "width": frame["width"], "height": frame["height"]})
            for xywh, label in zip(_xyxy_to_xywh(frame["boxes"]).tolist(), frame["labels"].tolist()):
                add(image_id, label + 1, xywh, 0)
            for xywh in _xyxy_to_xywh(frame["ignore"]).tolist():
                for category_id in range(1, NUM_CLASSES + 1):
                    add(image_id, category_id, xywh, 1)
        categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
        return {"images": images, "annotations": annotations, "categories": categories}

    def _detections(self, preds):
        detections = []
        for frame in self.frames:
            pred = preds.get(frame["image_id"])
            if pred is None:  # кадр без передбачень рахується порожнім
                continue
            scores = np.asarray(pred["scores"], np.float64).reshape(-1)
            labels = np.asarray(pred["labels"], np.int64).reshape(-1)
            if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
                raise ValueError(f"Класи мають бути 0-7, отримано {labels.min()}..{labels.max()}")
            boxes = _xyxy_to_xywh(pred["boxes"])
            order = np.argsort(-scores, kind="stable")[: self.max_det]
            image_id = self._ids[frame["image_id"]]
            for xywh, score, label in zip(boxes[order].tolist(), scores[order].tolist(), labels[order].tolist()):
                detections.append(
                    {"image_id": image_id, "category_id": label + 1, "bbox": xywh, "score": score}
                )
        return detections

    def evaluate(self, preds):
        """Повертає dict: mAP, AP50, AP75, AP_small, AP_medium, AP_large, AR, n_val_frames, n_gt,
        n_det, per_size (список рядків), per_class (список рядків). Кадри поза набором ігноруються."""
        detections = self._detections(preds)
        coco_eval = COCOeval_faster(self._coco, self._coco.loadRes(detections), "bbox")
        coco_eval.params.maxDets = [1, 100, self.max_det]
        coco_eval.evaluate()
        coco_eval.accumulate()
        # precision: [IoU, recall, клас, площа, maxDets]; площі: all, small, medium, large
        precision = coco_eval.eval["precision"][:, :, :, :, -1]
        recall = coco_eval.eval["recall"][:, :, :, -1]
        iou_50 = int(np.argmin(np.abs(coco_eval.params.iouThrs - 0.5)))
        iou_75 = int(np.argmin(np.abs(coco_eval.params.iouThrs - 0.75)))
        area_index = {name: i + 1 for i, name in enumerate(SIZE_NAMES)}

        result = {
            "mAP": _mean(precision[:, :, :, 0]),
            "AP50": _mean(precision[iou_50, :, :, 0]),
            "AP75": _mean(precision[iou_75, :, :, 0]),
            "AR": _mean(recall[:, :, 0]),
            "n_val_frames": len(self.frames),
            "n_gt": int(self._n_gt_class.sum()),
            "n_det": len(detections),
            "per_size": [],
            "per_class": [],
        }
        for name, n_gt in zip(SIZE_NAMES, self._n_gt_size):
            area = area_index[name]
            result[f"AP_{name}"] = _mean(precision[:, :, :, area])
            result["per_size"].append(
                {
                    "size": name,
                    "AP": result[f"AP_{name}"],
                    "AP50": _mean(precision[iou_50, :, :, area]),
                    "n_gt": n_gt,
                }
            )
        for class_id, name in enumerate(CLASS_NAMES):
            result["per_class"].append(
                {
                    "class_id": class_id,
                    "class": name,
                    "AP": _mean(precision[:, :, class_id, 0]),
                    "AP50": _mean(precision[iou_50, :, class_id, 0]),
                    "AP_small": _mean(precision[:, :, class_id, area_index["small"]]),
                    "n_gt": int(self._n_gt_class[class_id]),
                }
            )
        return result


def _json_number(value):
    return None if isinstance(value, float) and np.isnan(value) else value


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, result, fps, train_minutes, epochs, imgsz, extra=None):
    """Пише metrics.json (ключі контракту), per_size.csv, per_class.csv. Повертає metrics."""
    out_dir = Path(out_dir)
    metrics = {key: _json_number(value) for key, value in result.items() if key not in ("per_size", "per_class")}
    metrics.update(fps=fps, train_minutes=train_minutes, epochs=epochs, imgsz=imgsz)
    metrics.update(extra or {})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_csv(out_dir / "per_size.csv", result["per_size"])
    _write_csv(out_dir / "per_class.csv", result["per_class"])
    return metrics


def finish_run(predictor, cfg, out_dir, data_root, smoke, train_minutes, epochs, extra=None):
    """Спільний кінець будь-якого детектора: повний val (у SMOKE обрізаний), fps, усі виходи
    контракту, крім train_log.csv і weights: metrics.json, per_size.csv, per_class.csv,
    preds_val.json.gz, submission.csv (якщо predict_test). Повертає metrics."""
    from .submission import predict_test

    frames = split_frames(cfg, data_root, "val", smoke, full=True)
    conf, max_det = cfg["conf"], cfg["max_det"]
    preds = predict_frames(predictor, frames, conf, cfg["eval_batch"], max_det, cfg["num_workers"])
    n_fps = cfg["smoke_limits"]["fps_frames"] if smoke else cfg["fps_frames"]
    fps = measure_fps(predictor, frames, conf, n_fps)
    result = Evaluator(frames, max_det).evaluate(preds)
    metrics = write_outputs(out_dir, result, fps, train_minutes, epochs, cfg["imgsz"], extra)
    save_preds(preds, Path(out_dir) / "preds_val.json.gz")
    if cfg.get("predict_test"):
        predict_test(predictor, cfg, out_dir, data_root, smoke)
    print(json.dumps(metrics))
    return metrics
