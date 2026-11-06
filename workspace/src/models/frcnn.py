"""Runner Faster R-CNN (torchvision fasterrcnn_resnet50_fpn, COCO-pretrained) для T3: рани 1x.

Конфіг: секції model (trainable_backbone_layers, anchor_sizes, kwargs у FasterRCNN: rpn_*, box_*), train
(див. tv_common), failure (score_thr, iou_thr, examples). Розмір входу (min_size = max_size = imgsz)
і box_score_thresh/box_detections_per_img беруться з протоколу, якщо не задані в model.kwargs.
"""
import csv
import traceback
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .. import data
from . import tv_common

LABEL_OFFSET = 1  # 0 це фон; класи 0-7 -> мітки 1-8


def build_model(cfg, pretrained):
    """pretrained=True: COCO-ваги Faster R-CNN і нова голова на 8 класів + фон (потребує інтернету).
    pretrained=False: та сама архітектура без завантажень, ваги потім з weights/best.pt."""
    mcfg = cfg["model"]
    kwargs = {
        "min_size": cfg["imgsz"],
        "max_size": cfg["imgsz"],
        "box_score_thresh": cfg["conf"],
        "box_detections_per_img": cfg["max_det"],
    }
    kwargs.update(mcfg.get("kwargs", {}))
    if mcfg.get("anchor_sizes"):  # по одному розміру на рівень FPN, 3 співвідношення: ваги RPN з COCO завантажаться
        sizes = mcfg["anchor_sizes"]
        kwargs["rpn_anchor_generator"] = AnchorGenerator(tuple((s,) for s in sizes), ((0.5, 1.0, 2.0),) * len(sizes))
    num_classes = data.NUM_CLASSES + 1
    if pretrained:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=mcfg.get("trainable_backbone_layers", 3),
            **kwargs,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs)


def load_predictor(cfg, weights):
    """Контракт Predictor: ваги з find_weights(run_id)."""
    return tv_common.load_predictor(cfg, build_model, weights, LABEL_OFFSET)


@torch.no_grad()
def rpn_recall(predictor, cfg, frames, any_hit, out_path, n_frames=100):
    """Внесок першої стадії: частка GT, що мають пропозал RPN з IoU >= 0.5, проти частки GT, знайдених
    фінальними детекціями (score >= failure.score_thr). Тільки неігноровані GT, перші n_frames
    рівномірно з val. Пише rpn_recall.csv: size, n_gt, rpn_recall, final_recall, post_nms_top_n_test."""
    model, iou_thr = predictor.model, cfg["failure"]["iou_thr"]
    model.eval()
    picks = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
    sizes, rpn_hit, final_hit = [], [], []
    for index in picks:
        frame = frames[index]
        if not len(frame["boxes"]):
            continue
        tensor, _ = predictor.prepare(data.read_image_bgr(frame["path"]))
        images, _ = model.transform([tensor])
        proposals, _ = model.rpn(images, model.backbone(images.tensors))
        height, width = images.image_sizes[0]
        gt = frame["boxes"] * np.array([width / frame["width"], height / frame["height"]] * 2, np.float32)
        iou = tv_common._iou_matrix(proposals[0].float().cpu().numpy(), gt)
        rpn_hit += (iou >= iou_thr).any(axis=0).tolist() if len(iou) else [False] * len(gt)
        final_hit += any_hit[frame["image_id"]].tolist()
        areas = (frame["boxes"][:, 2:] - frame["boxes"][:, :2]).prod(axis=1)
        sizes += [tv_common._size_name(a) for a in areas]
    sizes, rpn_hit, final_hit = np.array(sizes), np.array(rpn_hit, bool), np.array(final_hit, bool)
    top_n = model.rpn.post_nms_top_n()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "n_gt", "rpn_recall", "final_recall", "post_nms_top_n_test"])
        for name in ["all", *tv_common.SIZE_NAMES]:
            mask = np.ones(len(sizes), bool) if name == "all" else sizes == name
            n = int(mask.sum())
            writer.writerow([name, n, round(float(rpn_hit[mask].mean()), 4) if n else "",
                             round(float(final_hit[mask].mean()), 4) if n else "", top_n])


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    metrics, predictor = tv_common.run_detector(
        cfg, out_dir, data_root, smoke, build_model, LABEL_OFFSET, tags=["two-stage", "faster-rcnn"]
    )
    any_hit = tv_common.failure_analysis(cfg, out_dir, data_root, smoke)
    try:  # додатковий аналіз не повинен губити основні виходи рану
        frames = data.split_frames(cfg, data_root, "val", smoke, full=True)
        rpn_recall(predictor, cfg, frames, any_hit, out_dir / "rpn_recall.csv")
    except Exception as e:
        traceback.print_exc()
        print(f"rpn_recall не пораховано: {e!r}")
