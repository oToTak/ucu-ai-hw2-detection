"""Runner `summary`: зводить усі results/*/metrics.json у summary.csv і два графіки.

Конфіг, наприклад configs/59_summary.yaml:
    runner: summary
    title: "Усі моделі"   # необов'язково

Збирає теки результатів, імена яких не починаються з "_" (тобто без _smoke).
У SMOKE беремо results/_smoke/*. Теки без metrics.json пропускаються з повідомленням.

Виходи в out_dir:
    summary.csv               рядок на ран, відсортовано за mAP (метрики + git/started з meta.json)
    plots/summary_map.png     mAP за ранами (горизонтальні бари)
    plots/summary_speed.png   mAP проти fps (точки з підписами)

Нічого не падає, якщо результатів ще немає: порожній summary.csv і повідомлення в print.
"""
import csv
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRIC_COLUMNS = [
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR",
    "fps", "train_minutes", "epochs", "imgsz", "n_val_frames",
]
META_COLUMNS = ["git", "started", "smoke"]


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_folders(smoke):
    root = RESULTS / "_smoke" if smoke else RESULTS
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _load_run(folder):
    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        print(f"summary: {folder.name}: немає metrics.json, пропускаю")
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"summary: {folder.name}: не прочитав metrics.json ({e!r}), пропускаю")
        return None
    meta = {}
    meta_path = folder / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    return {"run": folder.name, "metrics": metrics, "meta": meta}


def _sort_key(row):
    value = _float(row["mAP"])
    return (0, -value) if value is not None else (1, 0.0)


def _write_csv(rows, out_dir):
    columns = ["run", *METRIC_COLUMNS, *META_COLUMNS]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: таблиця → {out_dir / 'summary.csv'} ({len(rows)} ранів)")


def _plot_map(rows, path, title):
    rows = [r for r in rows if _float(r["mAP"]) is not None]
    if not rows:
        print("summary: немає mAP, summary_map.png пропущено")
        return
    rows = sorted(rows, key=lambda r: _float(r["mAP"]))  # найкращий зверху
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(rows) + 1.5)))
    ax.barh([r["run"] for r in rows], [_float(r["mAP"]) for r in rows])
    ax.set_xlabel("mAP (COCO)")
    ax.set_xlim(left=0)
    ax.set_title(f"mAP за ранами{': ' + title if title else ''}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_speed(rows, path, title):
    points = [r for r in rows if _float(r["fps"]) is not None and _float(r["mAP"]) is not None]
    if not points:
        print("summary: немає пар fps/mAP, summary_speed.png пропущено")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in points:
        x, y = _float(r["fps"]), _float(r["mAP"])
        ax.scatter([x], [y])
        ax.annotate(r["run"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("fps")
    ax.set_ylabel("mAP (COCO)")
    ax.set_title(f"Швидкість проти точності{': ' + title if title else ''}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _safe_plot(name, fn, *args):
    try:
        fn(*args)
    except Exception as e:  # графік не має валити ран
        print(f"summary: {name} не намальовано: {e!r}")
        if plt is not None:
            plt.close("all")


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    loaded = [r for r in (_load_run(f) for f in _run_folders(smoke)) if r]
    rows = []
    for item in loaded:
        row = {"run": item["run"]}
        row.update({c: item["metrics"].get(c) for c in METRIC_COLUMNS})
        row.update({c: item["meta"].get(c) for c in META_COLUMNS})
        rows.append(row)
    rows.sort(key=_sort_key)
    print(f"summary: знайдено {len(rows)} ранів з metrics.json")
    _write_csv(rows, out_dir)
    if plt is None:
        print("summary: matplotlib недоступний, графіків не буде")
        return
    title = cfg.get("title", "")
    _safe_plot("summary_map", _plot_map, rows, out_dir / "plots" / "summary_map.png", title)
    _safe_plot("summary_speed", _plot_speed, rows, out_dir / "plots" / "summary_speed.png", title)
