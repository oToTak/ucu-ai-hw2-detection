"""EDA: класи, розміри об'єктів, густина, відмінності між кліпами, ignore-регіони, роздільності.

Runner `eda` (T1b). CPU, без torch: pandas/matplotlib/numpy/PIL. Читає розмітку через `src.data`.
Виходи в out_dir: metrics.json, per_clip.csv, plots/*.png.
Розміри "після ресайзу" це довга сторона кадру -> imgsz (масштаб imgsz/long, зі збільшенням
для малих кадрів, як letterbox в Ultralytics). Порогові "малі" об'єкти: площа < 32² (COCO).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common, data

SPLITS = ["train", "val", "test"]
LABELLED = ["train", "val"]
COLORS = {"train": "#3b6fb6", "val": "#e07b39", "test": "#7a7a7a"}
SMALL_AREA, MEDIUM_AREA = 32**2, 96**2
SIZE_EDGES = [0, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 10**6]
GRID = (96, 54)  # сітка для частки ignore (приблизно, об'єднання регіонів)
REGISTRY_N_ANCHORS = 9


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def _save(plt, fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "plots" / name, dpi=110)
    plt.close(fig)


def _pct(values, qs=(50, 95, 99)):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def _round(obj, digits=4):
    """Рекурсивно в JSON-сумісні типи з округленням."""
    if isinstance(obj, dict):
        return {str(k): _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _ignore_fraction(frame):
    """Частка площі кадру під ignore-регіонами (об'єднання на грубій сітці)."""
    if not len(frame["ignore"]):
        return 0.0
    gx, gy = GRID
    mask = np.zeros((gy, gx), bool)
    for x1, y1, x2, y2 in frame["ignore"]:
        mask[int(y1 / frame["height"] * gy):int(np.ceil(y2 / frame["height"] * gy)),
             int(x1 / frame["width"] * gx):int(np.ceil(x2 / frame["width"] * gx))] = True
    return float(mask.mean())


def build_tables(frames_by_split, imgsz, step):
    """frame_df (рядок на кадр) і obj_df (рядок на об'єкт, лише train/val)."""
    selected = {s: {f["image_id"] for f in data.select_frames(frames_by_split[s], step if s == "train" else 1)}
                for s in SPLITS}
    frame_rows, obj_parts = [], []
    for split in SPLITS:
        for index, f in enumerate(frames_by_split[split]):
            long_side = max(f["width"], f["height"])
            n_obj = len(f["boxes"])
            kept = len(data.training_targets(f)[0]) if n_obj else 0
            frame_rows.append({
                "split": split, "clip": f["clip"], "image_id": f["image_id"], "width": f["width"],
                "height": f["height"], "long": long_side, "scale": imgsz / long_side, "n_obj": n_obj,
                "n_kept": kept, "n_ignore": len(f["ignore"]), "ignore_frac": _ignore_fraction(f),
                "selected": f["image_id"] in selected[split],
            })
            if n_obj:
                wh = f["boxes"][:, 2:] - f["boxes"][:, :2]
                obj_parts.append(pd.DataFrame({
                    "split": split, "clip": f["clip"], "frame_idx": len(frame_rows) - 1, "label": f["labels"],
                    "w": wh[:, 0], "h": wh[:, 1], "long": long_side, "scale": imgsz / long_side,
                    "occ": f["occlusion"], "trunc": f["truncation"],
                }))
    frame_df = pd.DataFrame(frame_rows)
    obj_df = pd.concat(obj_parts, ignore_index=True) if obj_parts else pd.DataFrame()
    if len(obj_df):
        obj_df["area"] = obj_df["w"] * obj_df["h"]
        obj_df["size"] = np.sqrt(obj_df["area"])            # sqrt(площі) у пікселях оригіналу
        obj_df["size_in"] = obj_df["size"] * obj_df["scale"]  # те саме після ресайзу до imgsz
        obj_df["rel"] = obj_df["size"] / obj_df["long"]       # відносний розмір (частка довгої сторони)
        obj_df["small_orig"] = obj_df["area"] < SMALL_AREA
        obj_df["small_in"] = obj_df["size_in"] < 32
    return frame_df, obj_df


def _size_shares(df, col="size"):
    n = max(len(df), 1)
    return {
        "n": int(len(df)),
        "small": float((df["area"] < SMALL_AREA).sum() / n),
        "medium": float(((df["area"] >= SMALL_AREA) & (df["area"] < MEDIUM_AREA)).sum() / n),
        "large": float((df["area"] >= MEDIUM_AREA).sum() / n),
        "small_after_resize": float((df["size_in"] < 32).sum() / n),
        "median_size_px": float(df[col].median()) if len(df) else None,
        "median_size_after_resize": float(df["size_in"].median()) if len(df) else None,
    }


def kmeans_anchors(wh, k, seed, iters=60, max_n=200_000):
    """IoU-k-means (як у YOLOv2/3) на (w, h) у пікселях входу. Повертає (anchors за площею, середній найкращий IoU)."""
    rng = np.random.default_rng(seed)
    if len(wh) > max_n:
        wh = wh[rng.choice(len(wh), max_n, replace=False)]
    wh = np.maximum(wh, 1.0)

    def iou(a, c):
        inter = np.minimum(a[:, None, 0], c[None, :, 0]) * np.minimum(a[:, None, 1], c[None, :, 1])
        return inter / (a[:, 0:1] * a[:, 1:2] + (c[:, 0] * c[:, 1])[None] - inter)

    centers = wh[rng.choice(len(wh), k, replace=False)].copy()
    for _ in range(iters):
        assign = iou(wh, centers).argmax(1)
        new = np.array([np.median(wh[assign == j], axis=0) if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    centers = centers[np.argsort(centers.prod(1))]
    return centers, float(iou(wh, centers).max(1).mean())


def projected_test_small(obj_df, frame_df):
    """Очікувана частка COCO-малих (<32² у пікселях оригіналу) на test.

    Тест без розмітки. Беремо відносні розміри об'єктів train+val (sqrt(площі)/довга сторона)
    й множимо на довгу сторону кадрів test. Припущення: ті самі сцени за відносним розміром.
    Подвійне зважування: по об'єктах (густі кліпи домінують) і по кліпах (кожен кліп однаково).
    """
    test = frame_df[frame_df["split"] == "test"]
    if test.empty or obj_df.empty:
        return None
    per_clip_rel = [g["rel"].to_numpy() for _, g in obj_df.groupby("clip")]
    pooled = obj_df["rel"].to_numpy()
    by_res = {}
    total_frames = len(test)
    obj_w = clip_w = 0.0
    for long_side, g in test.groupby("long"):
        weight = len(g) / total_frames
        o = float((pooled * long_side < 32).mean())
        c = float(np.mean([(r * long_side < 32).mean() for r in per_clip_rel]))
        by_res[int(long_side)] = {"test_frames": int(len(g)), "small_object_weighted": o, "small_clip_weighted": c}
        obj_w += weight * o
        clip_w += weight * c
    return {"small_object_weighted": obj_w, "small_clip_weighted": clip_w, "by_long_side": by_res}


def per_clip_table(frame_df, obj_df):
    rows = []
    objs = {c: g for c, g in obj_df.groupby("clip")} if len(obj_df) else {}
    for (split, clip), g in frame_df.groupby(["split", "clip"], sort=True):
        row = {"split": split, "clip": clip, "width": int(g["width"].iloc[0]), "height": int(g["height"].iloc[0]),
               "frames": int(len(g))}
        o = objs.get(clip)
        if o is not None:
            counts = np.bincount(o["label"], minlength=data.NUM_CLASSES) / len(o)
            row.update({
                "objects": int(len(o)), "obj_per_frame": float(g["n_obj"].mean()), "obj_p95": float(g["n_obj"].quantile(0.95)),
                "obj_max": int(g["n_obj"].max()), "median_size_px": float(o["size"].median()),
                "median_size_after_resize": float(o["size_in"].median()), "small_orig": float(o["small_orig"].mean()),
                "small_after_resize": float(o["small_in"].mean()), "ignore_frac": float(g["ignore_frac"].mean()),
                "frames_with_ignore": float((g["n_ignore"] > 0).mean()),
                **{f"share_{name}": float(counts[i]) for i, name in enumerate(data.CLASS_NAMES)},
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- графіки

def plot_classes(plt, obj_df, out_dir):
    names = data.CLASS_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(names))
    for i, split in enumerate(LABELLED):
        d = obj_df[obj_df["split"] == split]
        counts = np.bincount(d["label"], minlength=len(names))
        axes[0].bar(x + (i - 0.5) * 0.4, counts, 0.4, label=f"{split} ({len(d):,})", color=COLORS[split])
        axes[1].bar(x + (i - 0.5) * 0.4, counts / counts.sum() * 100, 0.4, label=split, color=COLORS[split])
    axes[0].set_yscale("log")
    axes[0].set_title("Об'єкти за класами (лог. шкала)")
    axes[1].set_title("Частка класу, %")
    for ax in axes:
        ax.set_xticks(x, names, rotation=30)
        ax.legend()
    _save(plt, fig, out_dir, "classes.png")


def plot_sizes(plt, obj_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.logspace(0, 3, 60)
    for split in LABELLED:
        d = obj_df[obj_df["split"] == split]
        axes[0].hist(d["size"], bins, histtype="step", density=True, label=f"{split}, оригінал", color=COLORS[split])
        axes[0].hist(d["size_in"], bins, histtype="step", density=True, linestyle="--",
                     label=f"{split}, після ресайзу", color=COLORS[split], alpha=0.7)
    for ax in axes[:1]:
        ax.axvline(32, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("sqrt(площа рамки), px")
        ax.set_title("Розмір об'єктів (вертикаль 32 px = поріг малого)")
        ax.legend(fontsize=7)
    train = obj_df[obj_df["split"] == "train"]
    data_by_class = [train.loc[train["label"] == i, "size_in"].to_numpy() for i in range(data.NUM_CLASSES)]
    axes[1].boxplot(data_by_class, tick_labels=data.CLASS_NAMES, showfliers=False, whis=(5, 95))
    axes[1].axhline(32, color="k", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title("train: sqrt(площі) після ресайзу за класами (5-95 %)")
    axes[1].tick_params(axis="x", rotation=30)
    _save(plt, fig, out_dir, "sizes.png")


def plot_resolutions(plt, frame_df, out_dir):
    table = frame_df.groupby(["split", "width", "height"]).size().reset_index(name="frames")
    table["res"] = table["width"].astype(str) + "x" + table["height"].astype(str)
    order = table.sort_values("width")["res"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(order))
    for i, split in enumerate(SPLITS):
        t = table[table["split"] == split].set_index("res").reindex(order).fillna(0)
        total = max(t["frames"].sum(), 1)
        axes[0].bar(x + (i - 1) * 0.28, t["frames"] / total * 100, 0.28, label=f"{split} ({int(total):,} кадрів)", color=COLORS[split])
        clips = frame_df[frame_df["split"] == split].drop_duplicates("clip").groupby(["width", "height"]).size()
        c = pd.Series({f"{w}x{h}": n for (w, h), n in clips.items()}).reindex(order).fillna(0)
        axes[1].bar(x + (i - 1) * 0.28, c, 0.28, label=split, color=COLORS[split])
    for ax, title in zip(axes, ["Частка кадрів за роздільністю, %", "Кількість кліпів за роздільністю"]):
        ax.set_xticks(x, order, rotation=30)
        ax.set_title(title)
        ax.legend(fontsize=8)
    _save(plt, fig, out_dir, "resolutions.png")


def plot_density(plt, frame_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split in LABELLED:
        d = frame_df[frame_df["split"] == split]["n_obj"]
        axes[0].hist(d, np.arange(0, d.max() + 10, 5), histtype="step", density=True,
                     label=f"{split}: p95={d.quantile(.95):.0f}, max={d.max()}", color=COLORS[split])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("об'єктів на кадр")
    axes[0].set_title("Густина об'єктів на кадр")
    axes[0].legend()
    clips = frame_df[frame_df["split"].isin(LABELLED)].groupby(["split", "clip"])["n_obj"].mean().reset_index()
    clips = clips.sort_values("n_obj")
    axes[1].bar(range(len(clips)), clips["n_obj"], color=[COLORS[s] for s in clips["split"]])
    axes[1].set_xlabel("кліпи (відсортовано)")
    axes[1].set_title("Середня кількість об'єктів на кадр по кліпах")
    _save(plt, fig, out_dir, "density.png")


def plot_clips(plt, clip_df, out_dir):
    d = clip_df[clip_df["split"].isin(LABELLED)].dropna(subset=["objects"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in LABELLED:
        s = d[d["split"] == split]
        axes[0].scatter(s["obj_per_frame"], s["median_size_after_resize"], s=14 + s["frames"] / 40, alpha=0.7,
                        color=COLORS[split], label=split)
        axes[1].scatter(s["median_size_px"], s["small_orig"] * 100, s=14 + s["frames"] / 40, alpha=0.7,
                        color=COLORS[split], label=split)
    axes[0].axhline(32, color="k", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("об'єктів на кадр (середнє)")
    axes[0].set_ylabel("медіана sqrt(площі) після ресайзу, px")
    axes[0].set_title("Кліпи: густина проти розміру")
    axes[1].set_xlabel("медіана sqrt(площі), px оригіналу")
    axes[1].set_ylabel("частка <32², %")
    axes[1].set_title("Кліпи: розмір проти частки малих (за оригіналом)")
    axes[0].legend()
    _save(plt, fig, out_dir, "clips_scatter.png")
    share_cols = [f"share_{n}" for n in data.CLASS_NAMES]
    heat = d.sort_values(["split", "clip"])
    fig, ax = plt.subplots(figsize=(7, 0.17 * len(heat) + 1.5))
    im = ax.imshow(heat[share_cols].to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(share_cols)), data.CLASS_NAMES, rotation=30)
    ax.set_yticks(range(len(heat)), [f"{s[:1]} {c[-14:]}" for s, c in zip(heat["split"], heat["clip"])], fontsize=5)
    ax.grid(False)
    ax.set_title("Склад класів по кліпах (t=train, v=val)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(plt, fig, out_dir, "clips_class_mix.png")


def plot_ignore(plt, frame_df, obj_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split in LABELLED:
        d = frame_df[(frame_df["split"] == split) & (frame_df["n_ignore"] > 0)]
        axes[0].hist(d["ignore_frac"] * 100, np.linspace(0, 100, 41), histtype="step", density=True,
                     color=COLORS[split], label=f"{split}: кадрів з ignore {len(d) / (frame_df['split'] == split).sum():.0%}")
    axes[0].set_xlabel("частка площі кадру під ignore, %")
    axes[0].set_title("Ignore-регіони (лише кадри, де вони є)")
    axes[0].legend()
    train = frame_df[frame_df["split"] == "train"]
    dropped = (train["n_obj"] - train["n_kept"]).sum()
    train_objs = obj_df[obj_df["split"] == "train"]
    # частка відкинутих за класами рахується окремо в metrics; тут лише густина n_ignore
    axes[1].hist(frame_df[frame_df["split"].isin(LABELLED)]["n_ignore"], np.arange(0, 12) - 0.5, color="#888")
    axes[1].set_xlabel("ignore-регіонів на кадр")
    axes[1].set_title(f"train: відкинуто з тренування {dropped:,} ({dropped / max(len(train_objs), 1):.1%})")
    _save(plt, fig, out_dir, "ignore.png")


def plot_small_share(plt, obj_df, frame_df, projection, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = []
    for split in LABELLED:
        for long_side, g in obj_df[obj_df["split"] == split].groupby("long"):
            groups.append((f"{split}\n{int(long_side)}", g["small_orig"].mean() * 100, g["small_in"].mean() * 100, split, len(g)))
    if projection:
        for long_side, v in sorted(projection["by_long_side"].items(), key=lambda kv: int(kv[0])):
            groups.append((f"test*\n{long_side}", v["small_object_weighted"] * 100, None, "test", v["test_frames"]))
    x = np.arange(len(groups))
    axes[0].bar(x, [g[1] for g in groups], color=[COLORS[g[3]] for g in groups])
    axes[0].set_xticks(x, [g[0] for g in groups], fontsize=7)
    axes[0].set_ylabel("частка <32² у пікселях оригіналу, %")
    axes[0].set_title("Малі об'єкти за роздільністю кадру\n(test* = проєкція з відносних розмірів train+val)")
    resized = [(g[0], g[2], g[3]) for g in groups if g[2] is not None]
    axes[1].bar(range(len(resized)), [r[1] for r in resized], color=[COLORS[r[2]] for r in resized])
    axes[1].set_xticks(range(len(resized)), [r[0] for r in resized], fontsize=7)
    axes[1].set_ylabel("частка <32² після ресайзу до imgsz, %")
    axes[1].set_title("Те саме після ресайзу довгої сторони до imgsz")
    _save(plt, fig, out_dir, "small_share.png")


def plot_anchors(plt, obj_df, anchors, mean_iou, imgsz, out_dir):
    train = obj_df[obj_df["split"] == "train"]
    sample = train.sample(min(len(train), 200_000), random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hb = axes[0].hexbin(sample["w"] * sample["scale"], sample["h"] * sample["scale"], gridsize=60, bins="log",
                        extent=(0, 150, 0, 150), cmap="Blues")
    axes[0].scatter(anchors[:, 0], anchors[:, 1], color="#d62728", s=24, zorder=3, label=f"k-means, k={len(anchors)}, mIoU={mean_iou:.2f}")
    axes[0].set_xlabel(f"w, px при imgsz={imgsz}")
    axes[0].set_ylabel("h, px")
    axes[0].set_title("train: розміри рамок (w,h) і IoU-k-means анкери")
    axes[0].legend(fontsize=8)
    fig.colorbar(hb, ax=axes[0], fraction=0.04)
    counts = [np.histogram(obj_df.loc[obj_df["split"] == s, "size_in"], SIZE_EDGES)[0] for s in LABELLED]
    labels = [f"{a}-{b}" if b < 10**6 else f">{a}" for a, b in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]
    x = np.arange(len(labels))
    for i, (split, c) in enumerate(zip(LABELLED, counts)):
        axes[1].bar(x + (i - 0.5) * 0.4, c / c.sum() * 100, 0.4, label=split, color=COLORS[split])
    axes[1].set_xticks(x, labels, rotation=45, fontsize=7)
    axes[1].set_xlabel("sqrt(площі) при imgsz, px")
    axes[1].set_ylabel("частка об'єктів, %")
    axes[1].set_title("Гістограма розмірів (для анкерів / вибору рівнів FPN)")
    axes[1].legend()
    _save(plt, fig, out_dir, "anchors.png")


def _load_preview(frame, max_side=1100):
    from PIL import Image

    with Image.open(frame["path"]) as im:
        im.draft("RGB", (max_side, max_side))
        return np.array(im.convert("RGB"))


def plot_box_check(plt, frames_by_split, seed, out_dir):
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(seed)
    picks = []
    for split in LABELLED:
        by_res = {}
        for f in frames_by_split[split]:
            if len(f["boxes"]) >= 15:
                by_res.setdefault((f["width"], f["height"]), []).append(f)
        for res in sorted(by_res, key=lambda r: -r[0]):
            if len(picks) < 6 and (split == "val" or res[0] in (1344, 1904, 2688)):
                picks.append(by_res[res][int(rng.integers(len(by_res[res])))])
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#17becf", "#a65628", "#f781bf"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.6))
    for ax, f in zip(axes.ravel(), picks):
        ax.imshow(_load_preview(f), extent=(0, f["width"], f["height"], 0))
        for (x1, y1, x2, y2), label in zip(f["boxes"], f["labels"]):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.9, edgecolor=palette[label]))
        for x1, y1, x2, y2 in f["ignore"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="red", alpha=0.25, edgecolor="red", lw=1))
        ax.set_title(f"{f['image_id']}  {f['width']}x{f['height']}  об'єктів {len(f['boxes'])}, ignore {len(f['ignore'])}", fontsize=8)
        ax.grid(False)
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, label=n) for c, n in zip(palette, data.CLASS_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8)
    _save(plt, fig, out_dir, "boxes_check.png")
    return [f["image_id"] for f in picks]


def plot_box_zoom(plt, frames_by_split, seed, out_dir, crop=(560, 320)):
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(seed + 1)
    picks = []
    for split, minimum_width in (("train", 1800), ("val", 3000)):
        pool = [f for f in frames_by_split[split] if f["width"] >= minimum_width and len(f["boxes"]) >= 30]
        if pool:
            picks.append(pool[int(rng.integers(len(pool)))])
    if not picks:
        return []
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#17becf", "#a65628", "#f781bf"]
    fig, axes = plt.subplots(1, len(picks), figsize=(7.5 * len(picks), 4.6), squeeze=False)
    for ax, f in zip(axes[0], picks):
        centers = (f["boxes"][:, :2] + f["boxes"][:, 2:]) / 2
        cx, cy = np.median(centers, axis=0)
        x0 = int(np.clip(cx - crop[0] / 2, 0, f["width"] - crop[0]))
        y0 = int(np.clip(cy - crop[1] / 2, 0, f["height"] - crop[1]))
        image = data.read_image_rgb(f["path"])[y0:y0 + crop[1], x0:x0 + crop[0]]
        ax.imshow(image, extent=(x0, x0 + crop[0], y0 + crop[1], y0))
        for (x1, y1, x2, y2), label in zip(f["boxes"], f["labels"]):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.1, edgecolor=palette[label]))
        ax.set_xlim(x0, x0 + crop[0])
        ax.set_ylim(y0 + crop[1], y0)
        ax.set_title(f"{f['image_id']} {f['width']}x{f['height']}: вирізка {crop[0]}x{crop[1]} у піксельному масштабі", fontsize=8)
        ax.grid(False)
        ax.axis("off")
    _save(plt, fig, out_dir, "boxes_zoom.png")
    return [f["image_id"] for f in picks]


# ---------------------------------------------------------------------------- run

def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    imgsz, step, seed = cfg["imgsz"], cfg["frame_step"], cfg["seed"]
    frames_by_split = {s: data.load_frames(data_root, s) for s in SPLITS}
    if smoke:
        frames_by_split = {s: common.smoke_subset(fr, 2, 64) for s, fr in frames_by_split.items()}
    plt = _plt()
    print(f"EDA imgsz={imgsz} frame_step={step} smoke={smoke}")

    frame_df, obj_df = build_tables(frames_by_split, imgsz, step)
    clip_df = per_clip_table(frame_df, obj_df)
    clip_df.round(4).to_csv(out_dir / "per_clip.csv", index=False)

    metrics = {"imgsz": imgsz, "frame_step": step, "smoke": smoke, "small_threshold_area": SMALL_AREA}

    # 1. склад сплітів і роздільності
    splits = {}
    for split in SPLITS:
        d = frame_df[frame_df["split"] == split]
        res = d.groupby(["width", "height"]).agg(frames=("clip", "size"), clips=("clip", "nunique")).reset_index()
        splits[split] = {
            "clips": int(d["clip"].nunique()), "frames": int(len(d)),
            "frames_selected": int(d["selected"].sum()),
            "objects": int(d["n_obj"].sum()),
            "max_width": int(d["width"].max()), "min_width": int(d["width"].min()),
            "resolutions": {f"{r.width}x{r.height}": {"frames": int(r.frames), "clips": int(r.clips)} for r in res.itertuples()},
            "frame_scale_to_imgsz": {"min": float(d["scale"].min()), "max": float(d["scale"].max()),
                                      "upscaled_frames_share": float((d["scale"] > 1).mean())},
        }
    metrics["splits"] = splits

    # 2. класи
    classes = {}
    for split in LABELLED:
        d = obj_df[obj_df["split"] == split]
        counts = np.bincount(d["label"], minlength=data.NUM_CLASSES)
        classes[split] = {n: {"count": int(c), "share": float(c / max(counts.sum(), 1))} for n, c in zip(data.CLASS_NAMES, counts)}
    metrics["classes"] = classes

    # 3. розміри: медіани по класах, частки малих, процентилі
    sizes = {"by_split": {}, "by_class": {}, "by_resolution": {}, "percentiles_size_after_resize": {}}
    for split in LABELLED:
        d = obj_df[obj_df["split"] == split]
        sizes["by_split"][split] = _size_shares(d)
        sizes["percentiles_size_after_resize"][split] = _pct(d["size_in"], (1, 5, 25, 50, 75, 95))
        sizes["by_class"][split] = {}
        for i, name in enumerate(data.CLASS_NAMES):
            g = d[d["label"] == i]
            if len(g):
                sizes["by_class"][split][name] = {
                    "n": int(len(g)), "median_w_px": float(g["w"].median()), "median_h_px": float(g["h"].median()),
                    "median_w_after_resize": float((g["w"] * g["scale"]).median()),
                    "median_h_after_resize": float((g["h"] * g["scale"]).median()),
                    "median_size_after_resize": float(g["size_in"].median()),
                    "small_orig": float(g["small_orig"].mean()), "small_after_resize": float(g["small_in"].mean()),
                }
        for (long_side), g in d.groupby("long"):
            sizes["by_resolution"][f"{split}_{int(long_side)}"] = _size_shares(g)
    metrics["sizes"] = sizes

    # 4. густина: для num_queries і max_det
    density = {}
    for split in LABELLED:
        d = frame_df[frame_df["split"] == split]
        sel = d[d["selected"]]
        n = d["n_obj"]
        density[split] = {
            "mean": float(n.mean()), "median": float(n.median()), "p95": float(n.quantile(0.95)),
            "p99": float(n.quantile(0.99)), "max": int(n.max()),
            "frames_over_300": int((n > 300).sum()), "frames_over_500": int((n > 500).sum()),
            "selected_frames_p95": float(sel["n_obj"].quantile(0.95)), "selected_frames_max": int(sel["n_obj"].max()),
        }
    metrics["objects_per_frame"] = density

    # 5. ignore
    ignore = {}
    for split in LABELLED:
        d = frame_df[frame_df["split"] == split]
        regions = d["n_ignore"]
        ignore[split] = {
            "frames_with_ignore": float((regions > 0).mean()), "regions_per_frame_mean": float(regions.mean()),
            "regions_max": int(regions.max()), "ignore_frac_mean_all_frames": float(d["ignore_frac"].mean()),
            "ignore_frac_mean_where_present": float(d.loc[regions > 0, "ignore_frac"].mean()) if (regions > 0).any() else 0.0,
            "ignore_frac_p95": float(d["ignore_frac"].quantile(0.95)),
            "objects_dropped_from_training": int((d["n_obj"] - d["n_kept"]).sum()),
            "objects_dropped_share": float((d["n_obj"] - d["n_kept"]).sum() / max(d["n_obj"].sum(), 1)),
        }
    metrics["ignore"] = ignore

    # 6. occlusion / truncation
    metrics["occlusion"] = {s: {str(k): float(v) for k, v in
                                 obj_df[obj_df["split"] == s]["occ"].value_counts(normalize=True).sort_index().items()} for s in LABELLED}
    metrics["truncation"] = {s: {str(k): float(v) for k, v in
                                  obj_df[obj_df["split"] == s]["trunc"].value_counts(normalize=True).sort_index().items()} for s in LABELLED}

    # 7. анкери й гістограма площ (train, розміри при imgsz)
    train = obj_df[obj_df["split"] == "train"]
    wh = np.stack([(train["w"] * train["scale"]).to_numpy(), (train["h"] * train["scale"]).to_numpy()], axis=1)
    anchors, mean_iou = kmeans_anchors(wh, REGISTRY_N_ANCHORS, seed)
    metrics["anchors"] = {
        "k": REGISTRY_N_ANCHORS, "space": f"w,h у пікселях при imgsz={imgsz} (довга сторона)",
        "anchors_wh": anchors.round(1).tolist(), "mean_best_iou": mean_iou,
    }
    metrics["size_histogram_after_resize"] = {
        "edges_sqrt_area_px": SIZE_EDGES[:-1] + [None],
        **{s: np.histogram(obj_df.loc[obj_df["split"] == s, "size_in"], SIZE_EDGES)[0].tolist() for s in LABELLED},
    }

    # 8. перенос AP_small на test
    projection = projected_test_small(obj_df, frame_df)
    metrics["test_projection"] = {
        "note": "test без розмітки: частка <32² у пікселях оригіналу, оцінена з відносних розмірів train+val і довгих сторін кадрів test",
        **(projection or {}),
    }
    val_small = sizes["by_split"]["val"]["small"]
    train_small = sizes["by_split"]["train"]["small"]
    metrics["small_share_summary"] = {
        "train_small_orig": train_small, "val_small_orig": val_small,
        "test_small_orig_projected_object_weighted": projection["small_object_weighted"] if projection else None,
        "test_small_orig_projected_clip_weighted": projection["small_clip_weighted"] if projection else None,
        "train_small_after_resize": sizes["by_split"]["train"]["small_after_resize"],
        "val_small_after_resize": sizes["by_split"]["val"]["small_after_resize"],
    }

    # 9. графіки
    plot_classes(plt, obj_df, out_dir)
    plot_sizes(plt, obj_df, out_dir)
    plot_resolutions(plt, frame_df, out_dir)
    plot_density(plt, frame_df, out_dir)
    plot_clips(plt, clip_df, out_dir)
    plot_ignore(plt, frame_df, obj_df, out_dir)
    plot_small_share(plt, obj_df, frame_df, projection, out_dir)
    plot_anchors(plt, obj_df, anchors, mean_iou, imgsz, out_dir)
    metrics["box_check_frames"] = plot_box_check(plt, frames_by_split, seed, out_dir)
    metrics["box_zoom_frames"] = plot_box_zoom(plt, frames_by_split, seed, out_dir)

    (out_dir / "metrics.json").write_text(json.dumps(_round(metrics), ensure_ascii=False, indent=1), encoding="utf-8")
    print("готово")
