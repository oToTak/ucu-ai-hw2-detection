"""Спільне для torchvision-детекторів (Faster R-CNN у T3, RetinaNet у T4).

Модель-специфічне лише в `build_model(cfg, pretrained) -> nn.Module` і `label_offset`
(Faster R-CNN: 1, бо 0 це фон; RetinaNet: 0). Решта (датасет, цикл тренування з AMP, val mAP
щоепохи через evaluator, Predictor, розбір фейлів) спільна.

Конфіг (ключі секції train, усі обов'язкові, значення в yaml рану):
    epochs, batch_size, lr, momentum, weight_decay, warmup_iters, final_lr_frac, grad_clip,
    amp, hflip, log_every, reserve_minutes
`reserve_minutes`: скільки хвилин лишити бюджету на повний val, test і fps після тренування.
Тренування зупиняється після епохи, якщо наступна не вміститься в budget_minutes - reserve_minutes.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import common, data, evaluator

SIZE_NAMES = ["small", "medium", "large"]
SMALL_MAX, MEDIUM_MAX = 32**2, 96**2


# ---------------------------------------------------------------- датасет

class Augment:
    """Горизонтальний фліп (з ймовірністю flip_prob) і викидання виродженних рамок (< 1 px)."""

    def __init__(self, flip_prob=0.0):
        self.flip_prob = flip_prob

    def __call__(self, image, boxes, labels):
        if self.flip_prob and random.random() < self.flip_prob:
            width = image.shape[1]
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        return np.ascontiguousarray(image), boxes[keep], labels[keep]


# ---------------------------------------------------------------- інференс

def set_inference_limits(model, conf, max_det):
    """Поріг score і ліміт детекцій на кадр: у Faster R-CNN живуть в roi_heads, у RetinaNet на моделі."""
    head = model.roi_heads if hasattr(model, "roi_heads") else model
    head.score_thresh = conf
    head.detections_per_img = max_det


class TorchvisionPredictor(common.Predictor):
    """Predictor над моделлю torchvision (інференс у fp32, без autocast). Кадр зменшується до довгої сторони imgsz (PIL, як у
    тренуванні), боксі повертаються у пікселі оригіналу, класи 0-7."""

    def __init__(self, model, device, imgsz, label_offset, max_det=500):
        self.model, self.device = model, device
        self.imgsz, self.label_offset = imgsz, label_offset
        self.max_det = max_det

    def prepare(self, image_bgr):
        """(тензор CHW RGB 0..1 на device, (sx, sy) масштаб оригінал -> вхід)."""
        height, width = image_bgr.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        if max(width, height) > self.imgsz:
            scale = self.imgsz / max(width, height)
            pil = pil.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(255).to(self.device)
        return tensor, (pil.width / width, pil.height / height)

    @torch.no_grad()
    def predict(self, images_bgr, conf=0.001):
        self.model.eval()
        set_inference_limits(self.model, conf, self.max_det)
        prepared = [self.prepare(image) for image in images_bgr]
        outputs = self.model([tensor for tensor, _ in prepared])  # fp32: під autocast якорі й декодування рамок квантуються
        results = []
        for output, (_, (sx, sy)) in zip(outputs, prepared):
            boxes = output["boxes"].float().cpu().numpy() / np.array([sx, sy, sx, sy], np.float32)
            results.append(
                {
                    "boxes": boxes,
                    "scores": output["scores"].float().cpu().numpy(),
                    "labels": output["labels"].cpu().numpy() - self.label_offset,
                }
            )
        return results


# ---------------------------------------------------------------- ваги

def save_weights(model, path):
    """Лише ваги: матриці conv/linear у fp16, 1-D (bias, BN) у fp32."""
    state = {k: (v.half() if v.is_floating_point() and v.ndim > 1 else v) for k, v in model.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_weights(model, path):
    """Завантажує ваги. strict=False лише через num_batches_tracked: після тренування з
    FrozenBatchNorm2d їх немає в чекпоінті, а ран без pretrained будує звичайний BatchNorm2d."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad or unexpected:
        raise RuntimeError(f"Ваги не збігаються з моделлю: немає {bad[:5]}, зайві {list(unexpected)[:5]}")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_predictor(cfg, build_model, weights, label_offset):
    """Контракт `load_predictor(cfg, weights)` для модуля моделі (обгортка в frcnn.py/retina.py)."""
    device = _device()
    model = build_model(cfg, False)
    load_weights(model, weights)
    return TorchvisionPredictor(model.to(device), device, cfg["imgsz"], label_offset, cfg["max_det"])


# ---------------------------------------------------------------- тренування

def _lr_factor(it, total, warmup, final_frac):
    if it < warmup:
        return 0.001 + (1 - 0.001) * it / max(1, warmup)
    progress = min(1.0, (it - warmup) / max(1, total - warmup))
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def run_detector(cfg, out_dir, data_root, smoke, build_model, label_offset, tags=()):
    """Тренує модель і записує всі виходи контракту (train_log.csv, weights/best.pt, metrics.json,
    per_size/per_class, preds_val.json.gz, submission.csv). Повертає (metrics, predictor)."""
    out_dir = Path(out_dir)
    t, limits = cfg["train"], cfg["smoke_limits"]
    device = _device()
    use_amp = bool(t["amp"]) and device.type == "cuda"
    epochs = limits["epochs"] if smoke else t["epochs"]
    max_iters = limits["max_iters"] if smoke else None
    run = common.wandb_init(cfg, out_dir, tags=list(tags))

    train_frames = data.split_frames(cfg, data_root, "train", smoke)
    val_frames = data.split_frames(cfg, data_root, "val", smoke)
    dataset = data.TorchDetectionDataset(
        train_frames,
        max_side=cfg["imgsz"],
        mask=cfg["mask_ignore"],
        fill=cfg["ignore_fill"],
        drop_ioa=cfg["ignore_drop_ioa"],
        label_offset=label_offset,
        transforms=Augment(t["hflip"]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=t["batch_size"],
        shuffle=True,
        drop_last=len(dataset) >= t["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=data.collate_detection,
        worker_init_fn=common.seed_worker,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
    )
    total_iters = epochs * len(loader) if max_iters is None else min(max_iters, epochs * len(loader))

    model = build_model(cfg, True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: _lr_factor(it, total_iters, t["warmup_iters"], t["final_lr_frac"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    predictor = TorchvisionPredictor(model, device, cfg["imgsz"], label_offset, cfg["max_det"])
    val_evaluator = evaluator.Evaluator(val_frames, cfg["max_det"])
    print(f"train {len(train_frames)} кадрів, val {len(val_frames)}, {len(loader)} ітерацій/епоха, device {device}, amp {use_amp}")

    weights_path = out_dir / "weights" / "best.pt"
    best, best_epoch, it, start = None, 0, 0, time.time()
    epochs_done, last_epoch_minutes = 0, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start, loss_sum, loss_count, skipped = time.time(), 0.0, 0, 0
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                skipped += 1
                print(f"epoch {epoch} it {it}: loss не скінченний, пропускаю батч")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            it += 1
            loss_sum += loss.item()
            loss_count += 1
            if it % t["log_every"] == 0:
                parts = {k: round(v.item(), 4) for k, v in losses.items()}
                print(f"epoch {epoch} it {it}/{total_iters} loss {loss.item():.4f} {parts} lr {scheduler.get_last_lr()[0]:.5f} "
                      f"{(time.time() - epoch_start) / loss_count:.2f} с/it")
                run.log({"iter": it, "loss": loss.item(), **{f"loss_{k}": v for k, v in parts.items()}})
            if max_iters is not None and it >= max_iters:
                break
        train_loss = loss_sum / max(1, loss_count)
        res = val_evaluator.evaluate(common.predict_frames(
            predictor, val_frames, cfg["conf"], cfg["eval_batch"], cfg["max_det"], cfg["num_workers"]))
        minutes = (time.time() - start) / 60
        common.append_train_log(out_dir, epoch, round(train_loss, 5), round(res["mAP"], 5), round(res["AP50"], 5), round(minutes, 2))
        run.log({"epoch": epoch, "train_loss": train_loss, "val_mAP": res["mAP"], "val_AP50": res["AP50"], "minutes": minutes})
        print(f"epoch {epoch}: loss {train_loss:.4f} val mAP {res['mAP']:.4f} AP50 {res['AP50']:.4f} "
              f"({minutes:.1f} хв, пропущено батчів {skipped})")
        epochs_done = epoch
        if best is None or (res["mAP"] == res["mAP"] and res["mAP"] > best):
            best, best_epoch = res["mAP"], epoch
            save_weights(model, weights_path)
        last_epoch_minutes = (time.time() - epoch_start) / 60
        if max_iters is not None and it >= max_iters:
            break
        if not smoke and minutes + last_epoch_minutes > cfg["budget_minutes"] - t["reserve_minutes"] and epoch < epochs:
            print(f"бюджет: наступна епоха не вміститься в {cfg['budget_minutes']} хв, зупиняюсь після епохи {epoch}")
            break

    train_minutes = (time.time() - start) / 60
    load_weights(model, weights_path)  # далі всюди best
    metrics = evaluator.finish_run(
        predictor, cfg, out_dir, data_root, smoke, round(train_minutes, 2), epochs_done,
        extra={"best_epoch": best_epoch, "best_val_mAP_curve": best, "iters": it, "batch_size": t["batch_size"], "lr": t["lr"]},
    )
    try:
        run.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception as e:
        print(f"wandb summary: {e}")
    return metrics, predictor


# ---------------------------------------------------------------- розбір фейлів

def _iou_matrix(a, b):
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (right_bottom - left_top).clip(0).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def _size_name(area):
    return "small" if area <= SMALL_MAX else "medium" if area <= MEDIUM_MAX else "large"


def _analyse_frame(frame, pred, score_thr, iou_thr):
    """Фейли одного кадру. Повертає (рядки, any_hit): any_hit[i] = чи є детекція будь-якого класу
    зі score >= score_thr та IoU >= iou_thr з i-м GT.
    missed_small: GT-small без жодної такої детекції. duplicate: детекція того ж класу на GT,
    уже зайнятому вищою за score детекцією. confusion: детекція, що збіглась лише з GT іншого класу."""
    gt, gt_labels = frame["boxes"], frame["labels"]
    keep = pred["scores"] >= score_thr
    boxes, scores, labels = pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]
    any_hit = np.zeros(len(gt), bool)
    rows = []

    def row(category, g, d):
        box = gt[g]
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        return {
            "category": category, "image_id": frame["image_id"], "clip": frame["clip"], "frame": frame["frame"],
            "gt_class": data.CLASS_NAMES[gt_labels[g]],
            "pred_class": data.CLASS_NAMES[labels[d]] if d is not None else "",
            "score": round(float(scores[d]), 4) if d is not None else "",
            "iou": round(float(iou[d, g]), 3) if d is not None else "",
            "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
            "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1),
            "area": round(area, 1), "size": _size_name(area),
            "occlusion": int(frame["occlusion"][g]), "truncation": int(frame["truncation"][g]),
            "similar_class": data.CLASS_NAMES[labels[d]] if category == "confusion" else "",
            "n_gt_in_frame": len(gt),
        }

    if len(gt) and len(boxes):
        iou = _iou_matrix(boxes, gt)
        any_hit = (iou >= iou_thr).any(axis=0)
        taken = np.zeros(len(gt), bool)
        for d in np.argsort(-scores, kind="stable"):
            same = (gt_labels == labels[d]) & (iou[d] >= iou_thr)
            free = same & ~taken
            if free.any():
                taken[np.argmax(np.where(free, iou[d], -1))] = True
            elif same.any():
                rows.append(row("duplicate", int(np.argmax(np.where(same, iou[d], -1))), int(d)))
            else:
                other = iou[d] >= iou_thr
                if other.any():
                    rows.append(row("confusion", int(np.argmax(np.where(other, iou[d], -1))), int(d)))
    else:
        iou = np.zeros((0, len(gt)))
    areas = (gt[:, 2:] - gt[:, :2]).prod(axis=1) if len(gt) else np.zeros(0)
    for g in np.flatnonzero(~any_hit & (areas <= SMALL_MAX)):
        rows.append(row("missed_small", int(g), None))
    return rows, any_hit


def _crop_window(frame_w, frame_h, row, win_w=480, win_h=360):
    cx, cy = (row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2
    x0 = int(min(max(0, cx - win_w / 2), max(0, frame_w - win_w)))
    y0 = int(min(max(0, cy - win_h / 2), max(0, frame_h - win_h)))
    return x0, y0, min(win_w, frame_w), min(win_h, frame_h)


def _pick_examples(rows, k):
    """По одному кадру з кліпу (найбільше фейлів у кадрі), кліпи за спаданням; до k прикладів."""
    by_frame = {}
    for r in rows:
        by_frame.setdefault((r["clip"], r["image_id"]), []).append(r)
    best_per_clip = {}
    for (clip, image_id), items in by_frame.items():
        if clip not in best_per_clip or len(items) > len(best_per_clip[clip][1]):
            best_per_clip[clip] = (image_id, items)
    ordered = sorted(best_per_clip.values(), key=lambda x: -len(x[1]))
    return ordered[:k]


def _plot_failures(category, rows, frames_by_id, preds, score_thr, path, k, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    examples = _pick_examples(rows, k)
    cols = 3
    fig, axes = plt.subplots(max(1, math.ceil(len(examples) / cols)), cols, figsize=(15, 4.6 * max(1, math.ceil(len(examples) / cols))), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (image_id, items) in zip(axes.ravel(), examples):
        frame, pred = frames_by_id[image_id], preds[image_id]
        x0, y0, w, h = _crop_window(frame["width"], frame["height"], items[0])
        ax.imshow(data.read_image_rgb(frame["path"])[y0 : y0 + h, x0 : x0 + w])

        def draw(box, color, width, style="-"):
            if box[2] < x0 or box[0] > x0 + w or box[3] < y0 or box[1] > y0 + h:
                return
            ax.add_patch(Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1],
                                   fill=False, edgecolor=color, linewidth=width, linestyle=style))

        for box in frame["boxes"]:
            draw(box, "#2ca02c", 0.8)
        keep = pred["scores"] >= score_thr
        for box in pred["boxes"][keep]:
            draw(box, "#d62728", 0.8, "--")
        for r in items:
            draw((r["x1"], r["y1"], r["x2"], r["y2"]), "#ffd400", 2.2)
        first = items[0]
        label = f"{first['gt_class']}" + (f" → {first['pred_class']}" if first["pred_class"] else "")
        ax.set_title(f"{image_id}\n{label}, у кадрі таких: {len(items)}", fontsize=9)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
    fig.suptitle(f"{title}  (зелений = GT, червоний пунктир = детекції score ≥ {score_thr}, жовтий = фейл)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def failure_analysis(cfg, out_dir, data_root, smoke):
    """Рахує 3 категорії фейлів на val з out_dir/preds_val.json.gz і малює їх.

    Виходи: failures.csv (рядок на фейл: розмір, occlusion, truncation, схожий клас),
    failures_summary.json (лічильники, топ пар плутанини), plots/failure_missed_small.png,
    plots/failure_duplicates.png, plots/failure_confusion.png. Повертає any_hit:
    {image_id: bool-масив по GT кадру} (чи знайдений об'єкт будь-якою детекцією score >= score_thr).
    Параметри в cfg["failure"]: score_thr, iou_thr, examples.
    """
    out_dir = Path(out_dir)
    f = cfg["failure"]
    frames = data.split_frames(cfg, data_root, "val", smoke, full=True)
    preds = data.load_preds(out_dir / "preds_val.json.gz")
    frames_by_id = {fr["image_id"]: fr for fr in frames}
    rows, any_hit = [], {}
    for frame in frames:
        pred = preds.get(frame["image_id"])
        if pred is None:
            pred = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32), "labels": np.zeros(0, np.int64)}
        frame_rows, hit = _analyse_frame(frame, pred, f["score_thr"], f["iou_thr"])
        rows += frame_rows
        any_hit[frame["image_id"]] = hit
    if rows:
        with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (out_dir / "failures.csv").write_text("category\n", encoding="utf-8")

    titles = {
        "missed_small": "Пропущені дрібні об'єкти (≤ 32² px)",
        "duplicate": "Дублікати в натовпі",
        "confusion": "Плутанина класів",
    }
    names = {"missed_small": "failure_missed_small.png", "duplicate": "failure_duplicates.png", "confusion": "failure_confusion.png"}
    summary = {"score_thr": f["score_thr"], "iou_thr": f["iou_thr"], "n_gt": int(sum(len(v) for v in any_hit.values()))}
    for category in titles:
        subset = [r for r in rows if r["category"] == category]
        pairs = {}
        for r in subset:
            pairs[f"{r['gt_class']}->{r['pred_class']}"] = pairs.get(f"{r['gt_class']}->{r['pred_class']}", 0) + 1
        summary[category] = {
            "count": len(subset),
            "by_size": {s: sum(r["size"] == s for r in subset) for s in SIZE_NAMES},
            "by_gt_class": {c: sum(r["gt_class"] == c for r in subset) for c in data.CLASS_NAMES if any(r["gt_class"] == c for r in subset)},
            "top_pairs": sorted(pairs.items(), key=lambda x: -x[1])[:5] if category == "confusion" else [],
            "occluded_share": round(float(np.mean([r["occlusion"] > 0 for r in subset])), 3) if subset else None,
            "truncated_share": round(float(np.mean([r["truncation"] > 0 for r in subset])), 3) if subset else None,
        }
        try:
            if subset:
                _plot_failures(category, subset, frames_by_id, preds, f["score_thr"], out_dir / "plots" / names[category],
                               f["examples"], f"{titles[category]}, всього {len(subset)} на {len(frames)} кадрах")
            else:
                print(f"failure_analysis: {category}: фейлів немає, графік пропущено")
        except Exception as e:  # графік не повинен губити решту виходів
            import traceback

            traceback.print_exc()
            print(f"failure_analysis: {category}: графік не намальовано: {e!r}")
    (out_dir / "failures_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("failures:", json.dumps({k: v["count"] for k, v in summary.items() if isinstance(v, dict)}))
    return any_hit
