"""submission.csv: image_id,PredictionString; група "class score x y w h" у пікселях оригіналу, класи 0-7."""
from pathlib import Path

import numpy as np

from .common import predict_frames
from .data import load_frames, split_frames

MAX_DET = 500


def prediction_string(pred):
    """Рядок для одного кадру; порожній, якщо детекцій немає."""
    boxes = np.asarray(pred["boxes"], np.float64).reshape(-1, 4)
    scores = np.asarray(pred["scores"]).tolist()
    labels = np.asarray(pred["labels"]).tolist()
    parts = []
    for (x1, y1, x2, y2), score, label in zip(boxes.tolist(), scores, labels):
        parts.append(f"{int(label)} {score:.5f} {x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f}")
    return " ".join(parts)


def write_submission(preds, test_frames, path, max_det=MAX_DET):
    """По рядку на кожен кадр test_frames; кадр без preds лишається порожнім. Обрізає до max_det."""
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("image_id,PredictionString\n")
        for frame in test_frames:
            pred = preds.get(frame["image_id"])
            text = ""
            if pred is not None:
                order = np.argsort(-np.asarray(pred["scores"]), kind="stable")[:max_det]
                text = prediction_string(
                    {key: np.asarray(pred[key])[order] for key in ("boxes", "scores", "labels")}
                )
            f.write(f"{frame['image_id']},{text}\n")


def predict_test(predictor, cfg, out_dir, data_root, smoke=False):
    """Передбачення на test і out_dir/submission.csv. У SMOKE рахує кілька кадрів, решта порожні,
    але файл має всі рядки. Повертає шлях."""
    frames = split_frames(cfg, data_root, "test", smoke)
    preds = predict_frames(predictor, frames, cfg["conf"], cfg["eval_batch"], cfg["max_det"], cfg["num_workers"])
    path = Path(out_dir) / "submission.csv"
    write_submission(preds, load_frames(data_root, "test"), path, cfg["max_det"])
    return path
