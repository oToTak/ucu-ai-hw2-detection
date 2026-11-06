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

Нічого не падає, якщо частини ранів ще немає: такий ран пропускається, причина йде в print (log.txt).
"""
import csv
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib є в requirements; якщо ні, таблиці все одно пишемо
    plt = None

RESULTS = Path(__file__).resolve().parents[1] / "results"
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
    return RESULTS / "_smoke" / run if smoke else RESULTS / run


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
    curve = []
    for row in curve_rows:
        epoch, val = _float(row.get("epoch")), _float(row.get("val_mAP"))
        if epoch is not None and val is not None:
            curve.append((epoch, val))
    return {"run": run, "metrics": metrics, "sizes": sizes, "curve": curve}


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


def _safe_plot(name, fn, *args):
    try:
        fn(*args)
    except Exception as e:  # графік не має валити ран
        print(f"compare: {name} не намальовано: {e!r}")
        if plt is not None:
            plt.close("all")


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    names = cfg.get("runs") or []
    if isinstance(names, str):
        names = [names]
    runs = [r for r in (_load_run(n, smoke) for n in names) if r]
    print(f"compare: знайдено {len(runs)} з {len(names)} ранів")
    _write_table(runs, out_dir)
    title = cfg.get("title", "")
    if plt is None:
        print("compare: matplotlib недоступний, графіків не буде")
        return
    _safe_plot("compare_sizes", _plot_sizes, runs, out_dir / "plots" / "compare_sizes.png", title)
    _safe_plot("compare_curves", _plot_curves, runs, out_dir / "plots" / "compare_curves.png", title)
