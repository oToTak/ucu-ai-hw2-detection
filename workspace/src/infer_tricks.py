"""Трюки інференсу над Predictor і збереженими передбаченнями: нарізання кадру (SAHI-подібне),
TTA (флип, масштаб), NMS, Soft-NMS і Weighted Boxes Fusion. Лише numpy: torch не потрібен.

Усе працює в пікселях оригіналу, класи 0-7. Predictor не знає про трюки: TrickPredictor сам
готує вхід (флип, масштаб, тайли) і повертає рамки в координатах оригінального кадру.

Джерела ідей і алгоритмів (цитувати у звіті):
- SAHI: Akyon, Altinuc, Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection", 2022, arXiv:2202.06934 (нарізання з перекриттям, прохід по цілому кадру, злиття NMS).
- Soft-NMS: Bodla, Singh, Chellappa, Davis, "Soft-NMS: Improving Object Detection With One Line of
  Code", ICCV 2017, arXiv:1704.04503 (лінійне й гаусове загасання score).
- WBF: Solovyev, Wang, Gabruseva, "Weighted boxes fusion: Ensembling boxes from different object
  detection models", Image and Vision Computing 107 (2021), arXiv:1910.13302. Реалізація за статтею,
  не копія бібліотеки ensemble-boxes.
"""
import contextlib

import numpy as np

from .common import Predictor

DEFAULT_MERGE = {
    "method": "wbf",  # wbf | nms | soft_nms
    "iou_thr": 0.6,  # wbf: поріг збігу з уже злитою рамкою; nms: поріг придушення
    "skip_box_thr": 0.0,  # wbf: рамки з score нижче не беруть участі
    "conf_type": "avg",  # wbf: avg (за статтею) | max
    "nms_metric": "iou",  # nms: iou | ios (перетин на меншу площу)
    "soft_method": "gaussian",  # soft_nms: gaussian | linear
    "sigma": 0.5,  # soft_nms gaussian
    "linear_iou_thr": 0.3,  # soft_nms linear
    "score_thr": 0.001,  # soft_nms: рамки нижче після загасання відкидаються
}
DEFAULT_SAHI = {
    "slice_size": 640,  # сторона тайла в пікселях оригіналу
    "overlap": 0.2,  # частка перекриття сусідніх тайлів
    "slice_imgsz": None,  # розмір входу моделі для тайлів; None лишає як у Predictor
    "edge_px": 3,  # рамки тайла, що торкаються внутрішнього краю тайла, відкидаються (їх покриє сусід)
    "tile_batch": 16,  # тайлів за один виклик predict
    "pre_nms": 5000,  # топ за score перед злиттям тайлів
    "nms_iou": 0.5,  # NMS між тайлами одного кадру
    "nms_metric": "iou",
    "source_max_det": 1000,  # скільки рамок лишається від усіх тайлів кадру
}
SIZE_STRIDE = 32


# ---------------------------------------------------------------- прості операції над передбаченнями

def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def make_pred(boxes, scores, labels):
    return {
        "boxes": _np(boxes).astype(np.float32).reshape(-1, 4),
        "scores": _np(scores).astype(np.float32).reshape(-1),
        "labels": _np(labels).astype(np.int64).reshape(-1),
    }


def concat_preds(preds):
    if not preds:
        return make_pred([], [], [])
    return make_pred(
        np.concatenate([_np(p["boxes"]).reshape(-1, 4) for p in preds]),
        np.concatenate([_np(p["scores"]).reshape(-1) for p in preds]),
        np.concatenate([_np(p["labels"]).reshape(-1) for p in preds]),
    )


def take(pred, index):
    index = np.asarray(index, dtype=np.int64)
    return {key: np.asarray(pred[key])[index] for key in ("boxes", "scores", "labels")}


def top_k(pred, k):
    """Лишає k рамок із найвищим score (за спаданням)."""
    order = np.argsort(-np.asarray(pred["scores"]), kind="stable")[:k]
    return take(pred, order)


def flip_boxes(boxes, width):
    """Дзеркалення по горизонталі в межах кадру шириною width."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def overlap(box, boxes, metric="iou"):
    """Перекриття однієї рамки xyxy з масивом рамок. iou або ios (перетин / менша площа)."""
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(right_bottom - left_top, 0, None).prod(axis=1)
    area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    if metric == "ios":
        union = np.minimum(area, areas)
    elif metric == "iou":
        union = area + areas - inter
    else:
        raise ValueError(f"metric: iou або ios, отримано {metric!r}")
    return inter / np.maximum(union, 1e-9)


# ---------------------------------------------------------------- NMS, Soft-NMS, WBF

def nms(pred, iou_thr=0.5, metric="iou"):
    """Жадібний NMS окремо по класах."""
    boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
    keep = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        while idx.size:
            best = idx[0]
            keep.append(best)
            if idx.size == 1:
                break
            rest = idx[1:]
            idx = rest[overlap(boxes[best], boxes[rest], metric) <= iou_thr]
    return take(pred, keep)


def soft_nms(pred, method="gaussian", sigma=0.5, iou_thr=0.3, score_thr=0.001):
    """Soft-NMS (Bodla et al. 2017) окремо по класах. gaussian: s *= exp(-IoU^2 / sigma);
    linear: s *= (1 - IoU) для IoU > iou_thr. Рамки зі score < score_thr відкидаються."""
    if method not in ("gaussian", "linear"):
        raise ValueError(f"soft_nms method: gaussian або linear, отримано {method!r}")
    out_boxes, out_scores, out_labels = [], [], []
    for label in np.unique(pred["labels"]):
        mask = pred["labels"] == label
        boxes = pred["boxes"][mask].astype(np.float64)
        scores = pred["scores"][mask].astype(np.float64)
        while len(scores):
            best = int(np.argmax(scores))
            out_boxes.append(boxes[best])
            out_scores.append(scores[best])
            out_labels.append(label)
            best_box = boxes[best]
            boxes = np.delete(boxes, best, axis=0)
            scores = np.delete(scores, best)
            if not len(scores):
                break
            iou = overlap(best_box, boxes)
            if method == "gaussian":
                scores = scores * np.exp(-(iou**2) / sigma)
            else:
                scores = np.where(iou > iou_thr, scores * (1.0 - iou), scores)
            alive = scores >= score_thr
            boxes, scores = boxes[alive], scores[alive]
    return make_pred(
        np.array(out_boxes).reshape(-1, 4), np.array(out_scores), np.array(out_labels)
    )


def wbf(sources, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type="avg"):
    """Weighted Boxes Fusion (Solovyev et al. 2021) над списком передбачень (по одному на модель
    чи вигляд). Алгоритм статті для кожного класу:
    1. усі рамки за score за спаданням (score помножено на вагу джерела);
    2. кожна рамка шукає серед уже злитих рамок ту з найбільшим IoU; якщо IoU > iou_thr, вона
       додається в цей кластер, інакше починає новий;
    3. після кожного додавання злита рамка перераховується: координати середнє зважене на score,
       score кластера середнє score його рамок;
    4. наприкінці score множиться на min(T, n) / T (T кількість джерел, n рамок у кластері):
       кластер, який бачила лише частина моделей, штрафується.
    conf_type='max': score кластера це максимум, поділений на найбільшу вагу.
    """
    if conf_type not in ("avg", "max"):
        raise ValueError(f"conf_type: avg або max, отримано {conf_type!r}")
    n_sources = len(sources)
    if n_sources == 0:
        return make_pred([], [], [])
    weights = np.ones(n_sources) if weights is None else np.asarray(weights, np.float64)
    if weights.shape != (n_sources,):
        raise ValueError(f"WBF: ваг {weights.shape}, джерел {n_sources}")
    all_boxes, all_scores, all_labels = [], [], []
    for pred, weight in zip(sources, weights):
        keep = np.asarray(pred["scores"]) >= skip_box_thr
        all_boxes.append(np.asarray(pred["boxes"], np.float64).reshape(-1, 4)[keep])
        all_scores.append(np.asarray(pred["scores"], np.float64).reshape(-1)[keep] * weight)
        all_labels.append(np.asarray(pred["labels"]).reshape(-1)[keep])
    boxes, scores, labels = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_labels)

    out_boxes, out_scores, out_labels = [], [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        cls_boxes, cls_scores = boxes[idx], scores[idx]
        fused = np.empty_like(cls_boxes)  # поточні злиті рамки (перші n_fused рядків)
        weighted_sum = np.zeros_like(cls_boxes)  # сума score * координати
        coord_weight = np.zeros(len(idx))  # сума score (знаменник зваженого середнього координат)
        score_sum = np.zeros(len(idx))
        count = np.zeros(len(idx), np.int64)
        best_score = np.zeros(len(idx))
        n_fused = 0
        for j in range(len(idx)):
            cluster = -1
            if n_fused:
                iou = overlap(cls_boxes[j], fused[:n_fused])
                top = int(np.argmax(iou))
                if iou[top] > iou_thr:
                    cluster = top
            if cluster < 0:
                cluster = n_fused
                n_fused += 1
            weight = max(cls_scores[j], 1e-9)  # нульовий score не має обнуляти координати
            weighted_sum[cluster] += weight * cls_boxes[j]
            coord_weight[cluster] += weight
            score_sum[cluster] += cls_scores[j]
            count[cluster] += 1
            best_score[cluster] = max(best_score[cluster], cls_scores[j])
            fused[cluster] = weighted_sum[cluster] / coord_weight[cluster]
        if conf_type == "avg":
            final = score_sum[:n_fused] / count[:n_fused] * np.minimum(n_sources, count[:n_fused]) / weights.sum()
        else:
            final = best_score[:n_fused] / weights.max()
        out_boxes.append(fused[:n_fused])
        out_scores.append(np.clip(final, 0.0, 1.0))
        out_labels.append(np.full(n_fused, label))
    if not out_boxes:
        return make_pred([], [], [])
    return make_pred(np.concatenate(out_boxes), np.concatenate(out_scores), np.concatenate(out_labels))


def merge_predictions(sources, cfg=None, weights=None, max_det=500):
    """Зводить передбачення одного кадру з кількох джерел (моделей, виглядів TTA, тайлів) в одне.
    cfg: словник як DEFAULT_MERGE. Одне джерело повертається без змін (лише топ max_det)."""
    cfg = {**DEFAULT_MERGE, **(cfg or {})}
    if len(sources) == 1:
        return top_k(sources[0], max_det)
    method = cfg["method"]
    if method == "wbf":
        merged = wbf(sources, weights, cfg["iou_thr"], cfg["skip_box_thr"], cfg["conf_type"])
    elif method == "nms":
        merged = nms(concat_preds(sources), cfg["iou_thr"], cfg["nms_metric"])
    elif method == "soft_nms":
        merged = soft_nms(
            concat_preds(sources), cfg["soft_method"], cfg["sigma"], cfg["linear_iou_thr"], cfg["score_thr"]
        )
    else:
        raise ValueError(f"merge.method: wbf, nms або soft_nms, отримано {method!r}")
    return top_k(merged, max_det)


# ---------------------------------------------------------------- SAHI-подібне нарізання

def slice_boxes(width, height, size, overlap_frac):
    """Тайли (x0, y0, x1, y1) розміром size із перекриттям overlap_frac; останній тайл по кожній осі
    зсувається до краю кадру (як у SAHI). Кадр меншим за size дає один тайл на весь кадр."""
    step = max(1, int(size * (1.0 - overlap_frac)))

    def starts(length):
        if length <= size:
            return [0]
        return [*range(0, length - size, step), length - size]

    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in starts(height)
        for x in starts(width)
    ]


def shift_tile_pred(pred, x0, y0, x1, y1, width, height, edge_px):
    """Тайл -> координати кадру. Рамки, що впираються у внутрішній край тайла (не край кадру),
    відкидаються: об'єкт обрізаний, його покриває сусідній тайл завдяки перекриттю."""
    boxes = np.asarray(_np(pred["boxes"]), np.float32).reshape(-1, 4) + np.array([x0, y0, x0, y0], np.float32)
    keep = np.ones(len(boxes), bool)
    if x0 > 0:
        keep &= boxes[:, 0] > x0 + edge_px
    if y0 > 0:
        keep &= boxes[:, 1] > y0 + edge_px
    if x1 < width:
        keep &= boxes[:, 2] < x1 - edge_px
    if y1 < height:
        keep &= boxes[:, 3] < y1 - edge_px
    return take(make_pred(boxes, pred["scores"], pred["labels"]), np.flatnonzero(keep))


# ---------------------------------------------------------------- розмір входу Predictor

@contextlib.contextmanager
def input_size(predictor, size):
    """Тимчасово міняє довгу сторону входу моделі (масштабний TTA, розмір тайлів).
    size=None нічого не міняє. Знає атрибут Predictor.imgsz і torchvision GeneralizedRCNNTransform
    (model.transform.min_size / max_size); без жодного з них піднімає ValueError."""
    if size is None:
        yield
        return
    changes = []
    if hasattr(predictor, "imgsz"):
        changes.append((predictor, "imgsz", predictor.imgsz, int(size)))
    transform = getattr(getattr(predictor, "model", None), "transform", None)
    if transform is not None and hasattr(transform, "min_size") and hasattr(transform, "max_size"):
        changes.append((transform, "min_size", transform.min_size, (int(size),)))
        changes.append((transform, "max_size", transform.max_size, int(size)))
    if not changes:
        raise ValueError(f"{type(predictor).__name__} не має imgsz: розмір входу змінити не можна")
    for obj, name, _, new in changes:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old, _ in changes:
            setattr(obj, name, old)


def scaled_size(predictor, scale):
    """Розмір входу для масштабу scale (кратний 32); None для масштабу 1."""
    if scale == 1.0:
        return None
    if not hasattr(predictor, "imgsz"):
        raise ValueError(f"{type(predictor).__name__} не має imgsz: масштабний TTA неможливий")
    return max(SIZE_STRIDE, int(round(predictor.imgsz * scale / SIZE_STRIDE)) * SIZE_STRIDE)


# ---------------------------------------------------------------- Predictor з трюками

class TrickPredictor(Predictor):
    """Один Predictor над кількома моделями, виглядами TTA і SAHI-тайлами.

    Джерела злиття: для кожної моделі кожен вигляд (масштаб x флип) по цілому кадру, плюс один
    набір тайлів, якщо sahi задано. Вигляд (1.0, без флипа) це прохід по цілому кадру, як у SAHI.
    Усі джерела зводить merge_predictions."""

    def __init__(self, predictors, scales=(1.0,), flip=False, sahi=None, merge=None, model_weights=None, max_det=500):
        self.predictors = list(predictors)
        if not self.predictors:
            raise ValueError("TrickPredictor: порожній список моделей")
        self.views = [(float(s), f) for s in scales for f in ((False, True) if flip else (False,))]
        self.sahi = {**DEFAULT_SAHI, **sahi} if sahi is not None else None
        self.merge = {**DEFAULT_MERGE, **(merge or {})}
        self.max_det = max_det
        model_weights = [1.0] * len(self.predictors) if model_weights is None else list(model_weights)
        if len(model_weights) != len(self.predictors):
            raise ValueError(f"model_weights: {len(model_weights)} ваг на {len(self.predictors)} моделей")
        per_model = len(self.views) + (1 if self.sahi else 0)
        self.source_weights = [w for w in model_weights for _ in range(per_model)]

    @classmethod
    def from_config(cls, predictors, tricks, max_det=500, model_weights=None):
        """tricks: словник {tta: {enabled, flip, scales}, sahi: {enabled, ...}, merge: {...}}."""
        tricks = tricks or {}
        tta = tricks.get("tta") or {}
        use_tta = bool(tta.get("enabled", False))
        sahi = tricks.get("sahi") or {}
        return cls(
            predictors,
            scales=tta.get("scales", [1.0]) if use_tta else [1.0],
            flip=bool(tta.get("flip", True)) if use_tta else False,
            sahi={k: v for k, v in sahi.items() if k != "enabled"} if sahi.get("enabled", False) else None,
            merge=tricks.get("merge"),
            model_weights=model_weights,
            max_det=max_det,
        )

    def describe(self):
        parts = [f"models={len(self.predictors)}", f"views={len(self.views)}"]
        if self.sahi:
            parts.append(f"sahi={self.sahi['slice_size']}/{self.sahi['overlap']}")
        if len(self.source_weights) > 1:
            parts.append(f"merge={self.merge['method']}")
        return " ".join(parts)

    def predict(self, images_bgr, conf=0.001):
        images = list(images_bgr)
        per_image = [[] for _ in images]
        for predictor in self.predictors:
            for scale, flip in self.views:
                for sources, pred in zip(per_image, self._predict_view(predictor, images, conf, scale, flip)):
                    sources.append(pred)
            if self.sahi:
                for sources, pred in zip(per_image, self._predict_tiles(predictor, images, conf)):
                    sources.append(pred)
        return [merge_predictions(sources, self.merge, self.source_weights, self.max_det) for sources in per_image]

    @staticmethod
    def _predict_view(predictor, images, conf, scale, flip):
        inputs = [np.ascontiguousarray(image[:, ::-1]) for image in images] if flip else images
        with input_size(predictor, scaled_size(predictor, scale)):
            outputs = predictor.predict(inputs, conf)
        results = []
        for image, out in zip(images, outputs):
            boxes = np.asarray(_np(out["boxes"]), np.float32).reshape(-1, 4)
            if flip:
                boxes = flip_boxes(boxes, image.shape[1])
            results.append(make_pred(boxes, out["scores"], out["labels"]))
        return results

    def _predict_tiles(self, predictor, images, conf):
        cfg = self.sahi
        jobs = []  # (індекс кадру, x0, y0, x1, y1)
        for index, image in enumerate(images):
            height, width = image.shape[:2]
            jobs += [(index, *box) for box in slice_boxes(width, height, cfg["slice_size"], cfg["overlap"])]
        found = [[] for _ in images]
        with input_size(predictor, cfg["slice_imgsz"]):
            for start in range(0, len(jobs), cfg["tile_batch"]):
                batch = jobs[start : start + cfg["tile_batch"]]
                crops = [np.ascontiguousarray(images[i][y0:y1, x0:x1]) for i, x0, y0, x1, y1 in batch]
                for (i, x0, y0, x1, y1), out in zip(batch, predictor.predict(crops, conf)):
                    height, width = images[i].shape[:2]
                    found[i].append(shift_tile_pred(out, x0, y0, x1, y1, width, height, cfg["edge_px"]))
        results = []
        for tiles in found:
            pred = top_k(concat_preds(tiles), cfg["pre_nms"])
            pred = nms(pred, cfg["nms_iou"], cfg["nms_metric"])
            results.append(top_k(pred, cfg["source_max_det"]))
        return results
