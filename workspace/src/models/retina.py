"""Runner RetinaNet (torchvision retinanet_resnet50_fpn, COCO-pretrained) для T4: рани 2x.

Конфіг, секція model:
    trainable_backbone_layers
    kwargs: у RetinaNet (fg_iou_thresh, bg_iou_thresh, nms_thresh, topk_candidates, ...)
    focal: {alpha, gamma}   focal loss; у torchvision вони зашиті (0.25, 2.0), тут явні
    anchors (необов'язково): якорі з k-means замість стандартних 32-512 px
        wh: список [w, h] у пікселях при imgsz; per_level: скільки якорів на локацію;
        extra_levels_scale: множники останньої групи для рівнів P6, P7
Секції train і протокол як у frcnn (див. tv_common). Розбір фейлів для 2x не рахується.
"""
import math
import types
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import sigmoid_focal_loss

from .. import data
from . import tv_common

LABEL_OFFSET = 0  # RetinaNet не має фону: мітки 0-7 це індекси колонок логітів
PRIOR = 0.01  # початкова ймовірність класу (bias останнього шару класифікатора)


class WHAnchorGenerator(AnchorGenerator):
    """Якорі з явними (w, h) на кожному рівні FPN. level_wh: список рівнів, кожен список пар (w, h)."""

    def __init__(self, level_wh):
        super().__init__(
            sizes=tuple((1,) * len(wh) for wh in level_wh), aspect_ratios=((1.0,),) * len(level_wh)
        )  # лише щоб num_anchors_per_location() був правильним
        self.cell_anchors = [
            torch.tensor([[-w / 2, -h / 2, w / 2, h / 2] for w, h in wh], dtype=torch.float32) for wh in level_wh
        ]


def anchor_levels(acfg):
    """Групи якорів за зростанням площі по per_level: перші рівні P3.., решта рівнів з множників."""
    wh = sorted(acfg["wh"], key=lambda pair: pair[0] * pair[1])
    n = acfg["per_level"]
    levels = [wh[i : i + n] for i in range(0, len(wh), n)]
    for scale in acfg["extra_levels_scale"]:
        levels.append([[w * scale, h * scale] for w, h in levels[-1]])
    if len(levels) != 5 or any(len(level) != n for level in levels):
        raise ValueError(f"RetinaNet має 5 рівнів FPN по {n} якорів, отримано {[len(l) for l in levels]}")
    return levels


def _focal_compute_loss(self, targets, head_outputs, matched_idxs):
    """compute_loss голови класифікації torchvision, але з alpha і gamma з конфігу (self.focal_alpha/gamma)."""
    losses = []
    for target, logits, matched in zip(targets, head_outputs["cls_logits"], matched_idxs):
        foreground = matched >= 0
        gt_classes = torch.zeros_like(logits)
        gt_classes[foreground, target["labels"][matched[foreground]]] = 1.0
        valid = matched != self.BETWEEN_THRESHOLDS  # якорі між порогами IoU ігноруються
        loss = sigmoid_focal_loss(
            logits[valid].float(), gt_classes[valid].float(), alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum"
        )
        losses.append(loss / max(1, int(foreground.sum())))
    return sum(losses) / len(targets)


def _new_conv(in_channels, out_channels, bias):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    nn.init.normal_(conv.weight, std=0.01)
    nn.init.constant_(conv.bias, bias)
    return conv


def _adapt(model, mcfg):
    """Голова на 8 класів, за потреби інші якорі, focal з конфігу. Однаково для pretrained і ні."""
    cls_head, reg_head = model.head.classification_head, model.head.regression_head
    in_channels = cls_head.cls_logits.in_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    if mcfg.get("anchors"):
        levels = anchor_levels(mcfg["anchors"])
        model.anchor_generator = WHAnchorGenerator(levels)
        if len(levels[0]) != num_anchors:  # інша кількість якорів: регресор теж новий (вага COCO не підходить)
            num_anchors = len(levels[0])
            reg_head.bbox_reg = _new_conv(in_channels, num_anchors * 4, 0.0)
            reg_head.num_anchors = num_anchors
    cls_head.cls_logits = _new_conv(in_channels, num_anchors * data.NUM_CLASSES, -math.log((1 - PRIOR) / PRIOR))
    cls_head.num_classes, cls_head.num_anchors = data.NUM_CLASSES, num_anchors
    cls_head.focal_alpha, cls_head.focal_gamma = float(mcfg["focal"]["alpha"]), float(mcfg["focal"]["gamma"])
    cls_head.compute_loss = types.MethodType(_focal_compute_loss, cls_head)


def build_model(cfg, pretrained):
    """pretrained=True: COCO-ваги RetinaNet (потребує інтернету), нові шари там, де форма змінилась.
    pretrained=False: та сама архітектура без завантажень, ваги потім з weights/best.pt."""
    mcfg = cfg["model"]
    kwargs = {
        "min_size": cfg["imgsz"],
        "max_size": cfg["imgsz"],
        "score_thresh": cfg["conf"],
        "detections_per_img": cfg["max_det"],
    }
    kwargs.update(mcfg.get("kwargs", {}))
    if pretrained:
        model = retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=mcfg.get("trainable_backbone_layers", 3),
            **kwargs,
        )
    else:
        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=data.NUM_CLASSES, **kwargs)
        tv_common.set_bn_eps(model, 0.0)  # як overwrite_eps у pretrained-гілці torchvision
    _adapt(model, mcfg)
    return model


def load_predictor(cfg, weights):
    """Контракт Predictor: ваги з find_weights(run_id)."""
    return tv_common.load_predictor(cfg, build_model, weights, LABEL_OFFSET)


def run(cfg, out_dir, data_root, smoke):
    tv_common.run_detector(
        cfg, Path(out_dir), data_root, smoke, build_model, LABEL_OFFSET, tags=["one-stage", "retinanet"]
    )
