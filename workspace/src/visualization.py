"""Runner `visualization` (B2): що моделі вивчили. Не тренує, лише читає готові ваги.

Три частини, кожна пропускається з повідомленням у log.txt, якщо немає ваг відповідного рану
(`common.find_weights`, у SMOKE із results/_smoke/); metrics.json пишеться завжди:
  1. Grad-CAM Faster R-CNN (`frcnn_run`) на рівнях FPN P2-P5: цільова детекція це топ-1 за score.
     RoI береться з одного рівня FPN (MultiScaleRoIAlign), тож інші рівні дають порожню карту.
  2. Grad-CAM RT-DETR (`rtdetr_run`) на трьох входах декодера (P3-P5 після гібридного енкодера); ціль топ-1 запит.
  3. RoI-ембедінги Faster R-CNN (вихід box_head, 1024-вимірний) для GT-рамок val, t-SNE, колір це клас.
Grad-CAM написаний вручну (градієнти через torch.autograd.grad): бібліотека pytorch-grad-cam не потрібна.
Інтерпретацію пише автор. W&B: картинки логуються, якщо є ключ (без нього no-op).
Ран короткий, відновлення посеред рану немає.

Конфіг: frcnn_run, rtdetr_run, n_frames, embed_frames, embed_per_class, tsne_perplexity, display_side.
"""
import json
import traceback
from pathlib import Path

import numpy as np

from . import common, data

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib є в requirements
    plt = None

CAM_ALPHA = 0.5


def _pick(frames, n):
    """n кадрів, рівномірно по списку."""
    if not frames:
        return []
    return frames[:: max(1, len(frames) // n)][:n]


def _display(frame, side):
    """RGB-кадр, зменшений до довгої сторони side, і множник оригінал -> показ."""
    from PIL import Image

    image = Image.fromarray(data.read_image_rgb(frame["path"]))
    scale = min(1.0, side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    return np.asarray(image), scale


def _cam_map(act, grad, size):
    """Grad-CAM: ReLU(sum_c(mean_hw(grad_c) * act_c)), до розміру size=(h, w), нормовано на максимум."""
    import torch
    import torch.nn.functional as F

    if grad is None:
        return np.zeros(size, np.float32)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0].detach().float()
    top = float(cam.max())
    return (cam / top).cpu().numpy() if top > 0 else np.zeros(size, np.float32)


# ---------------------------------------------------------------- цілі Grad-CAM

def _target_frcnn(predictor, bgr, size, conf):
    """Топ-1 детекція Faster R-CNN і Grad-CAM по P2-P5. size=(h, w) показу. None, якщо детекцій немає."""
    import torch

    from .models import tv_common

    net = predictor.model
    net.eval()
    tv_common.set_inference_limits(net, conf, 100)
    tensor, (sx, sy) = predictor.prepare(bgr)
    tensor = tensor.detach().requires_grad_(True)
    feats = {}
    handle = net.backbone.register_forward_hook(lambda _m, _i, out: feats.update(out))
    try:
        with torch.enable_grad():
            out = net([tensor])[0]
            if len(out["scores"]) == 0:
                return None
            score = out["scores"][0]  # детекції відсортовані за score
            names = [k for k in feats if k != "pool"]  # P6 (pool) RoI-пулінг не бачить
            grads = torch.autograd.grad(score, [feats[k] for k in names], allow_unused=True)
    finally:
        handle.remove()
    cams = {f"P{int(k) + 2}": _cam_map(feats[k], g, size) for k, g in zip(names, grads)}
    box = out["boxes"][0].detach().cpu().numpy() / np.array([sx, sy, sx, sy], np.float32)
    return {"box": box, "label": int(out["labels"][0]) - 1, "score": float(score), "cams": cams}


def _rtdetr_net(predictor):
    import torch

    net = predictor.model.model  # nn.Module Ultralytics (RTDETRDetectionModel)
    net.float().eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return net


def _target_rtdetr(predictor, bgr, size, imgsz):
    """Топ-1 запит RT-DETR і Grad-CAM по входах декодера. Кадр розтягується до imgsz x imgsz, як у predictor."""
    import cv2
    import torch

    net = _rtdetr_net(predictor)
    device = next(net.parameters()).device
    height, width = bgr.shape[:2]
    resized = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1])).permute(2, 0, 1)[None].float().div(255)
    tensor = tensor.to(device).requires_grad_(True)
    feats = []
    handle = net.model[-1].register_forward_pre_hook(lambda _m, args: feats.extend(args[0]))
    try:
        with torch.enable_grad():
            out = net(tensor)
            y = out[0] if isinstance(out, (tuple, list)) else out  # eval: (y, x), y = (1, nq, 6) cx cy w h score cls
            top = int(torch.argmax(y[0, :, 4]))
            score = y[0, top, 4]
            grads = torch.autograd.grad(score, feats, allow_unused=True)
    finally:
        handle.remove()
    cx, cy, w, h = [float(v) for v in y[0, top, :4].detach().cpu()]
    if max(cx, cy, w, h) > 1.5:  # піксельні координати входу замість нормованих
        cx, cy, w, h = cx / imgsz, cy / imgsz, w / imgsz, h / imgsz
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32) * np.array(
        [width, height, width, height], np.float32
    )
    cams = {f"P{3 + i}": _cam_map(f, g, size) for i, (f, g) in enumerate(zip(feats, grads))}
    return {"box": box, "label": int(y[0, top, 5]), "score": float(score), "cams": cams}


# ---------------------------------------------------------------- малювання

def _draw_box(ax, box, scale, color="lime"):
    x1, y1, x2, y2 = [v * scale for v in box]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2))


def _plot_cams(rows, title, out_path):
    """rows: [{frame, image, scale, target}]. Колонки: кадр із рамкою цілі, потім рівні."""
    levels = list(rows[0]["target"]["cams"])
    ncols = 1 + len(levels)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.3 * ncols, 2.0 * len(rows) + 0.9), squeeze=False)
    for r, row in enumerate(rows):
        t = row["target"]
        axes[r][0].imshow(row["image"])
        _draw_box(axes[r][0], t["box"], row["scale"])
        axes[r][0].set_title(f"{row['frame']['image_id']}\n{data.CLASS_NAMES[t['label']]} {t['score']:.2f}", fontsize=7)
        for c, name in enumerate(levels, start=1):
            cam = t["cams"][name]
            axes[r][c].imshow(row["image"])
            axes[r][c].imshow(cam, cmap="jet", alpha=CAM_ALPHA, vmin=0, vmax=1)
            _draw_box(axes[r][c], t["box"], row["scale"], "white")
            axes[r][c].set_title(name + ("" if cam.max() > 0 else " (порожньо)"), fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_embeddings(points, labels, method, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, name in enumerate(data.CLASS_NAMES):
        mask = labels == cls
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.6, label=f"{name} ({int(mask.sum())})", color=f"C{cls}")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"{title}\n{method}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _project(embeddings, perplexity, seed):
    """t-SNE (scikit-learn); без нього PCA на numpy. Повертає (точки Nx2, назва методу)."""
    try:
        from sklearn.manifold import TSNE

        perplexity = max(2.0, min(perplexity, (len(embeddings) - 1) / 3))
        points = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(embeddings)
        return points, f"t-SNE, perplexity {perplexity:.0f}"
    except ImportError:
        centered = embeddings - embeddings.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, "PCA (scikit-learn недоступний)"


# ---------------------------------------------------------------- ембедінги

def _roi_embeddings(predictor, frames, per_class, seed):
    """Вихід box_head для GT-рамок (не ignore) val-кадрів: (N x 1024, labels N)."""
    import torch

    net = predictor.model
    net.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for frame in frames:
            if len(frame["boxes"]) == 0:
                continue
            tensor, (sx, sy) = predictor.prepare(data.read_image_bgr(frame["path"]))
            images, _ = net.transform([tensor])
            height, width = images.image_sizes[0]
            rx, ry = width / tensor.shape[2], height / tensor.shape[1]
            boxes = frame["boxes"] * np.array([sx * rx, sy * ry, sx * rx, sy * ry], np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
            if not keep.any():
                continue
            feats = net.backbone(images.tensors)
            pooled = net.roi_heads.box_roi_pool(feats, [torch.as_tensor(boxes[keep]).to(images.tensors.device)], images.image_sizes)
            embeddings.append(net.roi_heads.box_head(pooled).float().cpu().numpy())
            labels.append(frame["labels"][keep])
    if not embeddings:
        return np.zeros((0, 1)), np.zeros(0, np.int64)
    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    rng = np.random.default_rng(seed)
    chosen = np.concatenate(
        [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in range(data.NUM_CLASSES)]
    ).astype(int)
    return embeddings[chosen], labels[chosen]


# ---------------------------------------------------------------- ран

def _load(run_id, smoke):
    """(base_cfg, module, weights) або None з повідомленням, якщо рану чи ваг ще немає."""
    import importlib

    from . import pipeline

    if not run_id:
        print("visualization: ран не заданий у конфігу, частину пропущено")
        return None
    try:
        weights = common.find_weights(run_id, smoke)
    except FileNotFoundError as e:
        print(f"visualization: {run_id}: ваг немає, частину пропущено ({e})")
        return None
    base = pipeline.load_config(run_id)
    module = importlib.import_module(f".{base['runner']}", __package__)
    print(f"visualization: {run_id}: ваги {weights}")
    return base, module, weights


def _log_wandb(run, paths):
    try:
        import wandb

        run.log({Path(p).stem: wandb.Image(str(p)) for p in paths})
    except Exception as e:  # без ключа run це no-op, а wandb.Image не потрібен
        print(f"wandb: картинки не залогувано ({e!r})")


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if plt is None:
        raise RuntimeError("matplotlib недоступний")
    wb = common.wandb_init(cfg, out_dir, tags=["visualization"])
    frames = data.split_frames(cfg, data_root, "val", smoke=smoke)
    side = cfg.get("display_side", 640)
    status, images = {}, []

    def part(name, fn):
        try:
            result = fn()
            status[name] = result or "ok"
        except Exception as e:
            traceback.print_exc()
            status[name] = f"помилка: {e!r}"
            if smoke:
                raise
        print(f"visualization: {name}: {status[name]}")

    def gradcam(predictor_fn, target_fn, title, file_name, frames_n):
        rows = []
        for frame in _pick(frames, frames_n):
            image, scale = _display(frame, side)
            target = target_fn(predictor_fn, data.read_image_bgr(frame["path"]), image.shape[:2])
            if target is None:
                print(f"visualization: {frame['image_id']}: детекцій немає, кадр пропущено")
                continue
            rows.append({"frame": frame, "image": image, "scale": scale, "target": target})
        if not rows:
            return "пропущено: жодної детекції"
        path = plots / file_name
        _plot_cams(rows, title, path)
        images.append(path)
        return f"ok: {len(rows)} кадрів"

    # 1 і 3: Faster R-CNN
    loaded = _load(cfg.get("frcnn_run"), smoke)
    if loaded is None:
        status["gradcam_frcnn"] = status["embeddings_frcnn"] = f"пропущено: немає ваг {cfg.get('frcnn_run')}"
    else:
        base, module, weights = loaded
        predictor = module.load_predictor(base, weights)
        part(
            "gradcam_frcnn",
            lambda: gradcam(
                predictor,
                lambda p, bgr, size: _target_frcnn(p, bgr, size, cfg["conf"]),
                f"Grad-CAM Faster R-CNN ({cfg['frcnn_run']}): ціль топ-1 детекція кадру, рівні FPN. "
                "Порожній рівень: RoI цієї детекції береться з іншого рівня.",
                "gradcam_frcnn.png",
                cfg["n_frames"],
            ),
        )

        def embeddings():
            picked = _pick(frames, cfg["embed_frames"])
            emb, labels = _roi_embeddings(predictor, picked, cfg["embed_per_class"], cfg["seed"])
            if len(emb) < 10:
                return f"пропущено: лише {len(emb)} рамок"
            points, method = _project(emb, cfg["tsne_perplexity"], cfg["seed"])
            path = plots / "embeddings_frcnn.png"
            _plot_embeddings(
                points, labels, method,
                f"RoI-ембедінги Faster R-CNN ({cfg['frcnn_run']}): GT-рамки val ({len(picked)} кадрів), колір це клас",
                path,
            )
            images.append(path)
            return f"ok: {len(emb)} рамок"

        part("embeddings_frcnn", embeddings)
        del predictor

    # 2: RT-DETR
    loaded = _load(cfg.get("rtdetr_run"), smoke)
    if loaded is None:
        status["gradcam_rtdetr"] = f"пропущено: немає ваг {cfg.get('rtdetr_run')}"
    else:
        base, module, weights = loaded
        predictor = module.load_predictor(base, weights)
        part(
            "gradcam_rtdetr",
            lambda: gradcam(
                predictor,
                lambda p, bgr, size: _target_rtdetr(p, bgr, size, base["imgsz"]),
                f"Grad-CAM RT-DETR ({cfg['rtdetr_run']}): ціль топ-1 запит, входи декодера P3-P5",
                "gradcam_rtdetr.png",
                cfg["n_frames"],
            ),
        )

    _log_wandb(wb, images)
    url = getattr(wb, "url", None)
    metrics = {
        "parts": status,
        "images": [p.name for p in images],
        "n_val_frames_pool": len(frames),
        "wandb_url": url if isinstance(url, str) else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
