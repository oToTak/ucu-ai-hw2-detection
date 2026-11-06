"""Читання split-ів, вибірка кадрів, ignore-регіони, датасети для Ultralytics і torchvision.

Кадр (frame) це dict: image_id ("<clip>_<frame:07d>", як у сабміті), clip, frame, path,
width, height, boxes (Nx4 float32 xyxy, пікселі оригіналу), labels (N int64, 0-7),
ignore (Mx4 float32 xyxy), occlusion, truncation, track_id. У test масиви порожні.

Передбачення (preds): {image_id: {"boxes": Nx4 xyxy px, "scores": N, "labels": N 0-7}}.
"""
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .common import smoke_subset

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASS_NAMES)
PREDS_FORMAT = "xyxy_px_v1"

_frames_cache = {}


def _empty_frame(image_id, clip, frame, path, width, height):
    return {
        "image_id": image_id,
        "clip": clip,
        "frame": frame,
        "path": path,
        "width": width,
        "height": height,
        "boxes": np.zeros((0, 4), np.float32),
        "labels": np.zeros(0, np.int64),
        "ignore": np.zeros((0, 4), np.float32),
        "occlusion": np.zeros(0, np.int32),
        "truncation": np.zeros(0, np.int32),
        "track_id": np.zeros(0, np.int32),
    }


def _xywh_to_xyxy(rows):
    boxes = np.array(rows, np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _load_labelled(data_root, split):
    with open(Path(data_root) / split / "annotations_coco.json", encoding="utf-8") as f:
        coco = json.load(f)
    objects, regions = {}, {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            # Кожен ignore-регіон у COCO повторений для всіх 8 категорій: лишаємо одну копію.
            regions.setdefault(ann["image_id"], set()).add(tuple(ann["bbox"]))
        else:
            objects.setdefault(ann["image_id"], []).append(ann)
    frames = []
    for image in coco["images"]:
        stem = Path(image["file_name"]).stem
        frame = _empty_frame(
            f"{image['video']}_{stem}",
            image["video"],
            int(image["frame_id"]),
            str(Path(data_root) / split / image["file_name"]),
            int(image["width"]),
            int(image["height"]),
        )
        anns = objects.get(image["id"], [])
        if anns:
            frame["boxes"] = _xywh_to_xyxy([a["bbox"] for a in anns])
            frame["labels"] = np.array([a["category_id"] - 1 for a in anns], np.int64)
            frame["occlusion"] = np.array([a.get("occlusion", 0) for a in anns], np.int32)
            frame["truncation"] = np.array([a.get("truncation", 0) for a in anns], np.int32)
            frame["track_id"] = np.array([a.get("track_id", -1) for a in anns], np.int32)
        if image["id"] in regions:
            frame["ignore"] = _xywh_to_xyxy(sorted(regions[image["id"]]))
        frames.append(frame)
    return frames


def _load_test(data_root):
    frames = []
    with open(Path(data_root) / "test_images.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frames.append(
                _empty_frame(
                    row["image_id"],
                    row["clip"],
                    int(row["frame"]),
                    str(Path(data_root) / row["file_name"]),
                    int(row["width"]),
                    int(row["height"]),
                )
            )
    return frames


def load_frames(data_root, split):
    """Усі кадри split-а ("train", "val", "test"), відсортовані за (clip, frame). Кеш у процесі."""
    key = (str(data_root), split)
    if key not in _frames_cache:
        frames = _load_test(data_root) if split == "test" else _load_labelled(data_root, split)
        _frames_cache[key] = sorted(frames, key=lambda f: (f["clip"], f["frame"]))
    return _frames_cache[key]


def select_frames(frames, step):
    """Кожен step-й кадр усередині кожного кліпу, починаючи з першого."""
    counters = {}
    picked = []
    for frame in frames:
        index = counters.get(frame["clip"], 0)
        counters[frame["clip"]] = index + 1
        if index % step == 0:
            picked.append(frame)
    return picked


def split_frames(cfg, data_root, split, smoke=False, full=False):
    """Кадри за протоколом: train з кроком frame_step; val з кроком val_step, а з full=True
    повний (фінальна оцінка); test повний. У SMOKE застосовує smoke_limits."""
    step = {"train": cfg["frame_step"], "val": 1 if full else cfg["val_step"], "test": 1}[split]
    frames = select_frames(load_frames(data_root, split), step)
    if smoke:
        limits = cfg["smoke_limits"]
        frames = smoke_subset(frames, limits[f"{split}_clips"], limits[f"{split}_frames"])
    return frames


def read_image_bgr(path):
    """Кадр як ndarray HxWx3 uint8 BGR (контракт Predictor)."""
    return read_image_rgb(path)[:, :, ::-1].copy()


def read_image_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def training_targets(frame, drop_ioa=0.5):
    """(boxes, labels) для тренування: без об'єктів, що лежать в ignore-регіоні на >= drop_ioa площі."""
    boxes, labels, regions = frame["boxes"], frame["labels"], frame["ignore"]
    if not len(boxes) or not len(regions):
        return boxes, labels
    left_top = np.maximum(boxes[:, None, :2], regions[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:], regions[None, :, 2:])
    overlap = (right_bottom - left_top).clip(0).prod(axis=2)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1).clip(1e-6)
    keep = (overlap / areas[:, None]).max(axis=1) < drop_ioa
    return boxes[keep], labels[keep]


def _pixel_ranges(box):
    x1, y1, x2, y2 = box
    return int(y1), math.ceil(y2), int(x1), math.ceil(x2)


def mask_ignore(image, frame, fill=114, drop_ioa=0.5):
    """Заливає ignore-регіони сірим, не зачіпаючи пікселів об'єктів, що лишились у тренуванні."""
    if not len(frame["ignore"]):
        return image
    masked = image.copy()
    for box in frame["ignore"]:
        y1, y2, x1, x2 = _pixel_ranges(box)
        masked[y1:y2, x1:x2] = fill
    for box in training_targets(frame, drop_ioa)[0]:
        y1, y2, x1, x2 = _pixel_ranges(box)
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return masked


def _resized(image, max_side):
    """PIL.Image зі зменшеною до max_side довгою стороною; збільшення немає."""
    pil = Image.fromarray(image)
    if max_side and max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.BILINEAR)
    return pil


def _yolo_label_lines(frame, drop_ioa):
    boxes, labels = training_targets(frame, drop_ioa)
    size = np.array([frame["width"], frame["height"]], np.float32)
    centers = ((boxes[:, :2] + boxes[:, 2:]) / 2 / size).clip(0, 1)
    sizes = ((boxes[:, 2:] - boxes[:, :2]) / size).clip(0, 1)
    return [
        f"{label} {c[0]:.6f} {c[1]:.6f} {s[0]:.6f} {s[1]:.6f}\n"
        for label, c, s in zip(labels.tolist(), centers.tolist(), sizes.tolist())
    ]


def cache_root():
    """Кеш картинок поза out_dir (не потрапляє в Output Kaggle); живе одну сесію."""
    return Path(os.environ.get("HW2_CACHE", Path(tempfile.gettempdir()) / "hw2_cache"))


def build_masked_cache(frames, fill=114, drop_ioa=0.5, max_side=None, workers=4):
    """Пише images/<clip>/<стем>.jpg із залитими ignore і labels/<clip>/<стем>.txt (YOLO).

    Той самий набір параметрів використовує готовий кеш. Повертає шляхи до картинок.
    """
    key = json.dumps([[f["image_id"] for f in frames], fill, drop_ioa, max_side])
    cache = cache_root() / f"masked_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    done = cache / ".done"

    def target(frame):
        return cache / "images" / frame["clip"] / f"{Path(frame['path']).stem}.jpg"

    def write_one(frame):
        image_path = target(frame)
        label_path = cache / "labels" / frame["clip"] / f"{image_path.stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        masked = mask_ignore(read_image_rgb(frame["path"]), frame, fill, drop_ioa)
        _resized(masked, max_side).save(image_path, quality=95)
        label_path.write_text("".join(_yolo_label_lines(frame, drop_ioa)), encoding="utf-8")

    if not done.exists():
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(write_one, frames))
        done.touch()
    return [str(target(frame)) for frame in frames]


def build_ultralytics_dataset(cfg, out_dir, data_root, smoke=False):
    """Готує датасет для Ultralytics і повертає шлях до yaml (передавати як data=).

    train: кадри з кроком frame_step із залитими ignore (кеш поза out_dir; вимикається
    mask_ignore: false, тоді беруться оригінали). val: оригінали з кроком val_step.
    Списки файлів і yaml лежать в out_dir/dataset (у git і light.zip не йдуть).
    """
    train = split_frames(cfg, data_root, "train", smoke)
    val = split_frames(cfg, data_root, "val", smoke)
    if cfg.get("mask_ignore", True):
        train_paths = build_masked_cache(
            train,
            cfg["ignore_fill"],
            cfg["ignore_drop_ioa"],
            cfg.get("cache_max_side", cfg["imgsz"]),
            cfg["num_workers"],
        )
    else:
        train_paths = [f["path"] for f in train]
    dataset_dir = Path(out_dir) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (dataset_dir / "val.txt").write_text("\n".join(f["path"] for f in val) + "\n", encoding="utf-8")
    spec = {
        "path": str(dataset_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")  # JSON це валідний YAML
    return yaml_path


class TorchDetectionDataset:
    """Map-style датасет для torch DataLoader (collate_fn=collate_detection).

    Елемент: (image FloatTensor CHW RGB 0..1, target {"boxes" xyxy у пікселях виходу, "labels", "image_id"}).
    labels 0-7 плюс label_offset: Faster R-CNN хоче 1-8 (label_offset=1), RetinaNet 0-7.
    max_side зменшує довгу сторону ще в датасеті. transforms(image_rgb, boxes, labels)
    повертає те саме (numpy) і викликається після заливки та зменшення.
    """

    def __init__(self, frames, max_side=None, mask=True, fill=114, drop_ioa=0.5, label_offset=0, transforms=None):
        self.frames = frames
        self.max_side = max_side
        self.mask = mask
        self.fill = fill
        self.drop_ioa = drop_ioa
        self.label_offset = label_offset
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        import torch

        frame = self.frames[index]
        image = read_image_rgb(frame["path"])
        boxes, labels = training_targets(frame, self.drop_ioa)
        if self.mask:
            image = mask_ignore(image, frame, self.fill, self.drop_ioa)
        pil = _resized(image, self.max_side)
        image = np.array(pil)
        boxes = boxes * np.array(
            [pil.width / frame["width"], pil.height / frame["height"]] * 2, np.float32
        )
        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255)
        target = {
            "boxes": torch.as_tensor(np.array(boxes), dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(np.array(labels) + self.label_offset, dtype=torch.int64),
            "image_id": torch.tensor(index),
        }
        return tensor, target


def collate_detection(batch):
    """Кадри різного розміру: (список картинок, список target)."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def save_preds(preds, path):
    """Пише preds у json.gz: {"format", "frames": {image_id: {"boxes", "scores", "labels"}}}."""
    frames = {
        image_id: {
            "boxes": np.round(np.asarray(p["boxes"], np.float64), 1).tolist(),
            "scores": np.round(np.asarray(p["scores"], np.float64), 5).tolist(),
            "labels": np.asarray(p["labels"]).astype(int).tolist(),
        }
        for image_id, p in preds.items()
    }
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump({"format": PREDS_FORMAT, "frames": frames}, f, separators=(",", ":"))


def load_preds(path):
    """Читає preds із json.gz у numpy."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        image_id: {
            "boxes": np.array(p["boxes"], np.float32).reshape(-1, 4),
            "scores": np.array(p["scores"], np.float32),
            "labels": np.array(p["labels"], np.int64),
        }
        for image_id, p in payload["frames"].items()
    }
