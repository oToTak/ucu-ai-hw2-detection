"""Runner `compare`: порівнює кілька ранів з cfg["runs"] (таблиця, бари за розміром, криві по епохах).

Конфіг, наприклад configs/19_compare.yaml:
    runner: compare
    runs: [10_frcnn_base, 11_frcnn_rpn]   # імена ранів; відсутні пропускаються з повідомленням
    title: "Two-stage"                    # необов'язково, заголовок графіків

Вхід: results/<ран>/metrics.json, per_size.csv (або AP_small/... з metrics.json), train_log.csv.
У SMOKE читає results/_smoke/<ран>.

Виходи в out_dir:
    compare.csv              рядок на ран (метрики з metrics.json)
    compare.md               та сама таблиця markdown, фрагмент для звіту
    plots/compare_sizes.png  AP small/medium/large за ранами (з per_size.csv)
    plots/compare_curves.png val_mAP за епохами (з train_log.csv)

Якщо в конфізі `density: true`: mAP за щільністю сцени (density.csv, plots/compare_density.png).
    density_bins: [e1, e2]   # межі за кількістю GT на кадр; за замовчуванням терцилі повного val
Кадри повного val ділимо на low/mid/high, кожну групу оцінює спільний Evaluator.

Нічого не падає, якщо частини ранів ще немає: такий ран пропускається, причина йде в print (log.txt).
"""
import csv
import json
import math
from pathlib import Path

from . import common

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib є в requirements; якщо ні, таблиці все одно пишемо
    plt = None

from .common import RESULTS, SMOKE_RESULTS

METRIC_COLUMNS = [
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR",
    "fps", "train_minutes", "epochs", "imgsz", "n_val_frames",
]
SIZES = ["small", "medium", "large"]


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_dir(run, smoke):
    return SMOKE_RESULTS / run if smoke else RESULTS / run


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_run(run, smoke):
    """Метрики, AP за розмірами і крива val_mAP одного рану. None, якщо метрик ще немає."""
    folder = _run_dir(run, smoke)
    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        print(f"compare: {run}: немає {metrics_path.name} у {folder}, пропускаю")
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sizes_rows = _read_csv(folder / "per_size.csv")
        curve_rows = _read_csv(folder / "train_log.csv")
    except (OSError, ValueError) as e:
        print(f"compare: {run}: не прочитав результати ({e!r}), пропускаю")
        return None
    if sizes_rows:
        sizes = {row["size"]: _float(row.get("AP")) for row in sizes_rows}
    else:
        sizes = {s: _float(metrics.get(f"AP_{s}")) for s in SIZES}
    curve, curve_min = [], []
    for row in curve_rows:
        epoch, val = _float(row.get("epoch")), _float(row.get("val_mAP"))
        if epoch is not None and val is not None:
            curve.append((epoch, val))
        minutes = _float(row.get("minutes"))  # стовпця може не бути або він порожній
        if minutes is not None and val is not None:
            curve_min.append((minutes, val))
    if not curve_min:
        print(f"compare: {run}: немає minutes у train_log.csv, у графіку за часом пропущено")
    return {"run": run, "metrics": metrics, "sizes": sizes, "curve": curve, "curve_min": curve_min}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else str(value)


def _write_table(runs, out_dir):
    columns = ["run", *METRIC_COLUMNS]
    rows = [{"run": r["run"], **{c: r["metrics"].get(c) for c in METRIC_COLUMNS}} for r in runs]
    with open(out_dir / "compare.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    if not rows:
        lines.append("| (немає готових ранів) |")
    (out_dir / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"compare: таблиця → {out_dir / 'compare.csv'} ({len(rows)} рядків)")


def _plot_sizes(runs, path, title):
    runs = [r for r in runs if any(v is not None for v in r["sizes"].values())]
    if not runs:
        print("compare: немає даних AP за розмірами, compare_sizes.png пропущено")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(runs)
    for i, r in enumerate(runs):
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(SIZES))]
        ys = [r["sizes"].get(s) or 0.0 for s in SIZES]
        ax.bar(xs, ys, width, label=r["run"])
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_ylabel("AP (COCO)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"AP за розміром об'єкта{': ' + title if title else ''}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_curves(runs, path, title):
    runs = [r for r in runs if r["curve"]]
    if not runs:
        print("compare: немає train_log з val_mAP, compare_curves.png пропущено")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        xs, ys = zip(*r["curve"])
        ax.plot(xs, ys, marker="o", label=r["run"])
    ax.set_xlabel("епоха")
    ax.set_ylabel("val mAP")
    ax.set_title(f"Крива val mAP{': ' + title if title else ''}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_curves_minutes(runs, path, title):
    runs = [r for r in runs if r["curve_min"]]
    if not runs:
        print("compare: немає minutes у train_log, compare_curves_minutes.png пропущено")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        xs, ys = zip(*r["curve_min"])
        ax.plot(xs, ys, marker="o", label=r["run"])
    ax.set_xlabel("хвилини тренування")
    ax.set_ylabel("val mAP")
    ax.set_title(f"Крива val mAP за часом{': ' + title if title else ''}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _safe_plot(name, fn, *args):
    try:
        fn(*args)
    except Exception as e:  # графік не має валити ран
        print(f"compare: {name} не намальовано: {e!r}")
        if plt is not None:
            plt.close("all")


DENSITY_COLUMNS = ["run", "bin", "n_frames", "mean_objects", "mAP", "AP50", "AP_small"]


def _present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _num(value):
    if not _present(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _density_edges(cfg, counts):
    """Межі low/mid/high за GT на кадр: density_bins з конфігу або терцилі повного val."""
    import numpy as np

    terciles = [float(np.quantile(counts, 1 / 3)), float(np.quantile(counts, 2 / 3))]
    given = cfg.get("density_bins")
    if given is None:
        return terciles
    try:
        edges = [float(x) for x in given]
    except (TypeError, ValueError):
        edges = []
    if len(edges) == 2 and edges[0] < edges[1]:
        return edges
    print(f"compare: density_bins має бути двома зростаючими числами, беру терцилі: {terciles}")
    return terciles


def _write_density_csv(rows, out_dir):
    with open(out_dir / "density.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DENSITY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _num(row[c]) for c in DENSITY_COLUMNS})
    print(f"compare: density → {out_dir / 'density.csv'} ({len(rows)} рядків)")


def _plot_density(rows, path, title):
    rows = [r for r in rows if _present(r["mAP"])]
    if not rows:
        print("compare: немає mAP за щільністю, compare_density.png пропущено")
        return
    bins = list(dict.fromkeys(r["bin"] for r in rows))
    runs = list(dict.fromkeys(r["run"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(runs)
    for i, run_name in enumerate(runs):
        values = {r["bin"]: r["mAP"] for r in rows if r["run"] == run_name}
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(bins))]
        ys = [values.get(b) if _present(values.get(b)) else 0.0 for b in bins]
        ax.bar(xs, ys, width, label=run_name)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_xlabel("щільність: GT-об'єктів на кадр")
    ax.set_ylabel("mAP (COCO)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"mAP за щільністю сцени{': ' + title if title else ''}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _density(cfg, names, smoke, data_root, out_dir):
    """mAP кожного рану на трьох групах кадрів повного val за кількістю GT-об'єктів."""
    import numpy as np

    from . import data, evaluator

    frames = data.split_frames(cfg, data_root, "val", smoke, full=True)
    counts = np.array([len(f["labels"]) for f in frames], np.float64)
    rows = []
    if counts.size:
        e1, e2 = _density_edges(cfg, counts)
        bins = [f"low (<={e1:g})", f"mid ({e1:g}..{e2:g}]", f"high (>{e2:g})"]
        group_idx = np.where(counts <= e1, 0, np.where(counts <= e2, 1, 2))
        groups = [[f for f, i in zip(frames, group_idx) if i == k] for k in range(3)]
        means = [float(counts[group_idx == k].mean()) if groups[k] else None for k in range(3)]
        # Evaluator будуємо один раз на групу: GT спільні для всіх ранів
        evaluators = [
            evaluator.Evaluator(g, max_det=cfg.get("max_det", 500)) if g else None for g in groups
        ]
        for name in names:
            try:
                preds = data.load_preds(common.find_preds(name, smoke))
            except FileNotFoundError as e:
                print(f"compare: density: {name}: немає preds_val, пропускаю ({e})")
                continue
            for k in range(3):
                row = {"run": name, "bin": bins[k], "n_frames": len(groups[k]), "mean_objects": means[k],
                       "mAP": None, "AP50": None, "AP_small": None}
                if evaluators[k] is not None:
                    res = evaluators[k].evaluate(preds)
                    row.update(mAP=res["mAP"], AP50=res["AP50"], AP_small=res["AP_small"])
                rows.append(row)
    else:
        print("compare: density: у val немає кадрів")
    _write_density_csv(rows, out_dir)
    if plt is None:
        print("compare: matplotlib недоступний, compare_density.png не буде")
    else:
        _safe_plot("compare_density", _plot_density, rows, out_dir / "plots" / "compare_density.png",
                   cfg.get("title", ""))


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    names = cfg.get("runs") or []
    if isinstance(names, str):
        names = [names]
    runs = [r for r in (_load_run(n, smoke) for n in names) if r]
    print(f"compare: знайдено {len(runs)} з {len(names)} ранів")
    _write_table(runs, out_dir)
    if cfg.get("density"):
        try:
            _density(cfg, names, smoke, data_root, out_dir)
        except Exception as e:  # щільність не має валити ран
            print(f"compare: density не порахований: {e!r}")
    title = cfg.get("title", "")
    if plt is None:
        print("compare: matplotlib недоступний, графіків не буде")
        return
    _safe_plot("compare_sizes", _plot_sizes, runs, out_dir / "plots" / "compare_sizes.png", title)
    _safe_plot("compare_curves", _plot_curves, runs, out_dir / "plots" / "compare_curves.png", title)
    _safe_plot(
        "compare_curves_minutes", _plot_curves_minutes, runs,
        out_dir / "plots" / "compare_curves_minutes.png", title,
    )
