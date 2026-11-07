"""Runner `temporal` (B1): часова постобробка збережених передбачень. Лише CPU, без torch і без зображень.

Крок 1: у кожному кліпі зв'язуємо детекції сусідніх кадрів (той самий клас, IoU >= link_iou,
жадібне відповідність 1-до-1 за спаданням IoU): простий трекер, що дає треки.
Крок 2: score по треку згладжується вікном ±window кадрів (середнє чи максимум), змішується з
власним score з вагою alpha, а короткі треки (< min_len кадрів) штрафуються множником.
Крок 3 (необов'язковий): впевнені детекції (score >= prop_thr) прокидаються в сусідній кадр, якщо там
немає рамки того ж класу з IoU >= prop_iou; новий score = score * prop_decay.

Ідеї: Seq-NMS (Han et al. 2016, arXiv:1602.08465, зв'язування рамок між кадрами й перерахунок score
по послідовності) і T-CNN (Kang et al. 2016, arXiv:1604.02532, прокидання детекцій у сусідні кадри).
Компенсацію руху камери не робимо: вона потребує зображень, а тут лише передбачення.

Конфіг:
  source: <ран>        ран із preds_val.json.gz (повний val, кроки кадрів 1) і, якщо predict_test, submission.csv
  grid: {параметр: [значення, ...]}   декартів добуток; ключі з PARAMS, відсутні беруть значення за замовчуванням
  select: mAP          за чим вибирати найкращу точку
Кожна точка сітки оцінюється спільним оцінювачем на тому ж val; baseline це source без змін.
Кадри в кліпі беруться за номером `frame`; зв'язуються лише сусідні кадри з різницею номерів <= max_gap.
Тому preds мають бути з повного val: з val_step=5 кадри віддалені на 5 і за max_gap=1 нічого не зв'яжеться.
"""
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import common, data
from .submission import write_submission

PARAMS = {
    "link_iou": 0.5,  # поріг IoU зв'язування
    "max_gap": 1,  # найбільша різниця номерів кадрів між сусідніми кадрами, що зв'язуються
    "link_min_score": 0.01,  # детекції нижче не беруть участі в треках і лишаються без змін
    "window": 2,  # напівширина вікна згладжування в кадрах; 0 вимикає згладжування
    "alpha": 0.5,  # частка згладженого score у новому
    "agg": "mean",  # mean | max
    "min_len": 2,  # треки коротші за це отримують short_penalty
    "short_penalty": 1.0,  # 1.0 вимикає штраф
    "prop_thr": None,  # score, з якого детекція прокидається в сусідній кадр; None вимикає
    "prop_decay": 0.8,
    "prop_iou": 0.5,  # у цільовому кадрі вже є така рамка того ж класу: не прокидаємо
    "prop_dir": "both",  # both | forward (лише з минулого)
}
KEY_BIG = 10**7  # key = track * KEY_BIG + frame: вікно не виходить за межі треку
ROWS_FILE = "_grid_rows.json"
METRIC_KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR", "n_det"]


# ---------------------------------------------------------------- дані кліпу

def iou_matrix(a, b):
    """IoU xyxy-рамок a (n,4) і b (m,4) -> (n,m)."""
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(right_bottom - left_top, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class Clip:
    """Плоскі масиви детекцій кліпу в порядку кадрів: offsets[k]:offsets[k+1] це кадр k."""

    def __init__(self, image_ids, frame_numbers, preds):
        self.image_ids = list(image_ids)
        self.frame_numbers = np.asarray(frame_numbers, np.int64)
        parts = [preds.get(i) for i in self.image_ids]
        counts = [0 if p is None else len(p["scores"]) for p in parts]
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        filled = [p for p in parts if p is not None and len(p["scores"])]
        self.boxes = np.concatenate([np.asarray(p["boxes"], np.float32).reshape(-1, 4) for p in filled]) if filled else np.zeros((0, 4), np.float32)
        self.scores = np.concatenate([np.asarray(p["scores"], np.float32) for p in filled]) if filled else np.zeros(0, np.float32)
        self.labels = np.concatenate([np.asarray(p["labels"], np.int64) for p in filled]) if filled else np.zeros(0, np.int64)
        self.frame_of = np.repeat(self.frame_numbers, np.diff(self.offsets))
        self.n_frames = len(self.image_ids)

    def frame_slice(self, k):
        return slice(self.offsets[k], self.offsets[k + 1])


def build_clips(frames, preds):
    """frames: кадри одного split-а. Повертає {clip: Clip}, кадри за номером."""
    by_clip = {}
    for frame in sorted(frames, key=lambda f: (f["clip"], f["frame"])):
        by_clip.setdefault(frame["clip"], []).append(frame)
    return {
        clip: Clip([f["image_id"] for f in items], [f["frame"] for f in items], preds)
        for clip, items in by_clip.items()
    }


# ---------------------------------------------------------------- трекер

def link_clip(clip, link_iou, max_gap, min_score):
    """Повертає (track_id, eligible): track_id (N,) для детекцій зі score >= min_score, інші -1."""
    n = len(clip.scores)
    eligible = clip.scores >= min_score
    tid = np.full(n, -1, np.int64)
    next_id = 0
    prev_idx = np.zeros(0, np.int64)  # плоскі індекси придатних детекцій попереднього кадру
    prev_k = None
    for k in range(clip.n_frames):
        sl = clip.frame_slice(k)
        cur_idx = np.flatnonzero(eligible[sl]) + sl.start
        linked = np.full(len(cur_idx), -1, np.int64)  # позиція у prev_idx
        if prev_k is not None and len(prev_idx) and len(cur_idx) and clip.frame_numbers[k] - clip.frame_numbers[prev_k] <= max_gap:
            iou = iou_matrix(clip.boxes[prev_idx], clip.boxes[cur_idx])
            iou[clip.labels[prev_idx][:, None] != clip.labels[cur_idx][None, :]] = 0.0
            rows, cols = np.nonzero(iou >= link_iou)
            if len(rows):
                order = np.argsort(-iou[rows, cols], kind="stable")
                used_prev = np.zeros(len(prev_idx), bool)
                for o in order:
                    i, j = rows[o], cols[o]
                    if not used_prev[i] and linked[j] < 0:
                        used_prev[i] = True
                        linked[j] = i
        fresh = linked < 0
        tid[cur_idx[~fresh]] = tid[prev_idx[linked[~fresh]]]
        tid[cur_idx[fresh]] = np.arange(next_id, next_id + int(fresh.sum()))
        next_id += int(fresh.sum())
        prev_idx, prev_k = cur_idx, k
    return tid


# ---------------------------------------------------------------- перерахунок score

def smooth_scores(clip, tid, p):
    """Нові score: згладжування по треку й штраф коротких треків. Детекції без треку не змінюються."""
    scores = clip.scores.astype(np.float64).copy()
    members = np.flatnonzero(tid >= 0)
    if not len(members):
        return scores
    key = tid[members] * KEY_BIG + clip.frame_of[members]
    order = np.argsort(key, kind="stable")
    members, key = members[order], key[order]
    s = scores[members]
    new = s.copy()
    window = int(p["window"])
    if window > 0 and p["alpha"] > 0:
        lo = np.searchsorted(key, key - window, side="left")
        hi = np.searchsorted(key, key + window, side="right")
        if p["agg"] == "mean":
            cs = np.concatenate([[0.0], np.cumsum(s)])
            agg = (cs[hi] - cs[lo]) / (hi - lo)
        elif p["agg"] == "max":
            agg = s.copy()
            for shift in range(1, window + 1):  # позиції в треку відрізняються щонайменше на 1 кадр
                for sign in (-1, 1):
                    idx = np.arange(len(s)) + sign * shift
                    valid = (idx >= 0) & (idx < len(s))
                    idx = np.clip(idx, 0, len(s) - 1)
                    near = valid & (np.abs(key[idx] - key) <= window)
                    agg = np.where(near, np.maximum(agg, s[idx]), agg)
        else:
            raise ValueError(f"agg: mean або max, отримано {p['agg']!r}")
        new = (1.0 - p["alpha"]) * s + p["alpha"] * agg
    if p["short_penalty"] != 1.0:
        length = np.bincount(tid[members])[tid[members]]
        new = np.where(length < p["min_len"], new * p["short_penalty"], new)
    scores[members] = np.clip(new, 0.0, 1.0)
    return scores


def propagate(clip, scores, p):
    """Додаткові детекції: [(k_цілі, boxes, scores, labels)], прокинуті з сусідніх кадрів за початковим score."""
    extras = []
    directions = [(-1, 0)] if p["prop_dir"] == "forward" else [(-1, 0), (1, 0)]
    if p["prop_dir"] not in ("both", "forward"):
        raise ValueError(f"prop_dir: both або forward, отримано {p['prop_dir']!r}")
    for k in range(clip.n_frames):
        target = clip.frame_slice(k)
        for step, _ in directions:
            src_k = k + step
            if src_k < 0 or src_k >= clip.n_frames:
                continue
            if abs(clip.frame_numbers[k] - clip.frame_numbers[src_k]) > p["max_gap"]:
                continue
            src = clip.frame_slice(src_k)
            strong = np.flatnonzero(clip.scores[src] >= p["prop_thr"]) + src.start
            if not len(strong):
                continue
            keep = np.ones(len(strong), bool)
            if target.stop > target.start:
                iou = iou_matrix(clip.boxes[strong], clip.boxes[target])
                same = clip.labels[strong][:, None] == clip.labels[target][None, :]
                keep = ~((iou >= p["prop_iou"]) & same).any(axis=1)
            strong = strong[keep]
            if len(strong):
                extras.append((k, clip.boxes[strong], clip.scores[strong] * p["prop_decay"], clip.labels[strong]))
    return extras


def apply_clip(clip, tid, p, max_det):
    """preds {image_id: {boxes, scores, labels}} кліпу після temporal-кроку з параметрами p."""
    scores = smooth_scores(clip, tid, p) if (p["window"] > 0 and p["alpha"] > 0) or p["short_penalty"] != 1.0 else clip.scores
    extra_by_frame = {}
    if p["prop_thr"] is not None:
        for k, boxes, sc, labels in propagate(clip, scores, p):
            extra_by_frame.setdefault(k, []).append((boxes, sc, labels))
    out = {}
    for k, image_id in enumerate(clip.image_ids):
        sl = clip.frame_slice(k)
        boxes, sc, labels = clip.boxes[sl], np.asarray(scores[sl], np.float32), clip.labels[sl]
        if k in extra_by_frame:
            boxes = np.concatenate([boxes] + [e[0] for e in extra_by_frame[k]])
            sc = np.concatenate([sc] + [np.asarray(e[1], np.float32) for e in extra_by_frame[k]])
            labels = np.concatenate([labels] + [e[2] for e in extra_by_frame[k]])
        order = np.argsort(-sc, kind="stable")[:max_det]
        out[image_id] = {"boxes": boxes[order], "scores": sc[order], "labels": labels[order]}
    return out


class Temporal:
    """Temporal-крок над набором кліпів з кешем зв'язування між точками сітки."""

    def __init__(self, clips, max_det=500):
        self.clips, self.max_det = clips, max_det
        self._links = {}

    def _tracks(self, name, clip, p):
        key = (name, p["link_iou"], p["max_gap"], p["link_min_score"])
        if key not in self._links:
            self._links[key] = link_clip(clip, p["link_iou"], p["max_gap"], p["link_min_score"])
        return self._links[key]

    def apply(self, params):
        p = {**PARAMS, **params}
        preds = {}
        for name, clip in self.clips.items():
            preds.update(apply_clip(clip, self._tracks(name, clip, p), p, self.max_det))
        return preds


# ---------------------------------------------------------------- вхід/вихід

def read_submission(path, frames):
    """submission.csv -> preds {image_id: {boxes xyxy, scores, labels}} для кадрів frames."""
    wanted = {f["image_id"] for f in frames}
    preds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["image_id"] not in wanted:
                continue
            values = np.array(row["PredictionString"].split(), np.float64).reshape(-1, 6)
            boxes = values[:, 2:6].copy()
            boxes[:, 2:] += boxes[:, :2]
            preds[row["image_id"]] = {
                "boxes": boxes.astype(np.float32),
                "scores": values[:, 1].astype(np.float32),
                "labels": values[:, 0].astype(np.int64),
            }
    return preds


def expand_grid(grid):
    """Декартів добуток сітки -> список словників параметрів (відсутні ключі беруть PARAMS)."""
    unknown = set(grid) - set(PARAMS)
    if unknown:
        raise ValueError(f"Невідомі параметри сітки: {sorted(unknown)}; дозволені {sorted(PARAMS)}")
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]


def _row(label, params, result):
    row = {"name": label, **{k: params.get(k, PARAMS[k]) for k in PARAMS}}
    row.update({k: result[k] for k in METRIC_KEYS})
    return row


def _write_grid(path, rows, baseline):
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + ["d_mAP", "d_AP50", "d_AP_small"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**row, **{f"d_{k}": (row[k] or 0.0) - (baseline[k] or 0.0) for k in ("mAP", "AP50", "AP_small")}}
            )


def _plot(rows, baseline, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.5 * len(rows) + 6), 4.5))
    for ax, key in zip(axes, ("mAP", "AP_small")):
        ax.bar(labels, [r[key] or 0.0 for r in rows])
        ax.axhline(baseline[key] or 0.0, color="red", linestyle="--", label="без temporal")
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(cfg, out_dir, data_root, smoke):
    from . import evaluator, pipeline  # evaluator тягне faster-coco-eval: імпорт лише тут

    out_dir = Path(out_dir)
    started = time.time()
    state = common.load_progress(out_dir) or {}
    minutes0 = float(state.get("minutes", 0.0))
    wb = common.wandb_init(cfg, out_dir, tags=["temporal"])
    source = cfg["source"]
    max_det = cfg["max_det"]

    frames = data.split_frames(cfg, data_root, "val", smoke, full=True)
    val_preds = data.load_preds(common.find_preds(source, smoke))
    covered = [f for f in frames if f["image_id"] in val_preds]
    if len(covered) < len(frames):
        print(f"temporal: у передбаченнях {source} {len(covered)} із {len(frames)} кадрів val; оцінка на покритих")
    frames = covered
    clips = build_clips(frames, val_preds)
    engine = Temporal(clips, max_det)
    scorer = evaluator.Evaluator(frames, max_det)

    rows_path = out_dir / ROWS_FILE
    done = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else {}
    variants = [("baseline", None)] + [
        (f"g{i:02d}", params) for i, params in enumerate(expand_grid(cfg.get("grid") or {}))
    ]
    results = {}
    for name, params in variants:
        if name in done:
            results[name] = done[name]
            continue
        preds = val_preds if params is None else engine.apply(params)
        result = scorer.evaluate({i: preds[i] for i in (f["image_id"] for f in frames) if i in preds})
        results[name] = {"params": params, "result": {k: result[k] for k in (*METRIC_KEYS, "n_val_frames", "n_gt")} | {
            "per_size": result["per_size"], "per_class": result["per_class"]}}
        done[name] = results[name]
        rows_path.write_text(json.dumps(done, default=float), encoding="utf-8")
        common.save_progress(out_dir, {"stage": "grid", "minutes": minutes0 + (time.time() - started) / 60}, files=[rows_path])
        print(f"{name}: mAP={result['mAP']:.4f} AP_small={result['AP_small']:.4f} {params}")

    baseline = results["baseline"]["result"]
    rows = [_row(name, entry["params"] or {}, entry["result"]) for name, entry in results.items()]
    _write_grid(out_dir / "temporal_grid.csv", rows, baseline)
    _plot(rows, baseline, out_dir / "plots" / "temporal_grid.png")

    candidates = [n for n in results if n != "baseline"] or ["baseline"]
    best = max(candidates, key=lambda n: results[n]["result"][cfg.get("select", "mAP")] or 0.0)
    best_params, best_result = results[best]["params"] or {}, results[best]["result"]
    start = time.perf_counter()
    best_preds = val_preds if best == "baseline" else engine.apply(best_params)
    seconds = time.perf_counter() - start

    base_imgsz = pipeline.load_config(source)["imgsz"]
    extra = {
        "source": source,
        "best": best,
        "best_params": {**PARAMS, **best_params},
        "baseline_mAP": baseline["mAP"],
        "baseline_AP_small": baseline["AP_small"],
        "delta_mAP": (best_result["mAP"] or 0.0) - (baseline["mAP"] or 0.0),
        "delta_AP_small": (best_result["AP_small"] or 0.0) - (baseline["AP_small"] or 0.0),
        "postprocess_seconds_val": seconds,
        "n_grid": len(candidates),
    }
    metrics = evaluator.write_outputs(out_dir, best_result, None, 0.0, 0, base_imgsz, extra)
    data.save_preds(best_preds, out_dir / "preds_val.json.gz")

    if cfg.get("predict_test"):
        test_frames = data.split_frames(cfg, data_root, "test", smoke)
        sub_path = common._find_in_runs(source, ["submission.csv"], "submission.csv", smoke)
        test_preds = read_submission(sub_path, test_frames)
        if best != "baseline":
            test_preds = Temporal(build_clips(test_frames, test_preds), max_det).apply(best_params)
        write_submission(test_preds, data.load_frames(data_root, "test"), out_dir / "submission.csv", max_det)

    wb.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    rows_path.unlink(missing_ok=True)
    print(json.dumps(metrics, ensure_ascii=False, default=float))
