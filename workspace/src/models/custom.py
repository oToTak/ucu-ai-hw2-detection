"""B3, своя архітектура: абляції над 30_yolo_base через ul_common (протокол, відновлення, оцінка ті самі).

Конфіг задає рівно одну зміну відносно `base`:
    arch: {name, nc, scales, backbone, head}   архітектура; ім'я дає масштаб (yolo11s-p2 -> s)
    nwd:  {ratio, constant}                    домішок NWD у box-loss
    experiment: {change, why, hypothesis}      для журналу experiments.md

Голова P2. Ultralytics передає ваги за іменем і формою, а Detect у новій архітектурі має інший індекс шару,
тому голова P3-P5 не передалась би. _build_seed переносить її явно й пише seed.pt, який ul_common навчає так само,
як yolo11s.pt. Відновлення не залежить від seed.pt: resume бере модель і yaml із last.pt.

NWD. Міняється BboxLoss.forward на час рану (метод класу, у чекпоінт не потрапляє, після рану повертається).
Рамки в BboxLoss у клітинках свого рівня, тому NWD рахується в пікселях входу через stride.

Виходи на додачу до контракту: custom_checks.json, experiments.md.
"""

import contextlib
import json
import math
from pathlib import Path

from .. import common, data
from . import ul_common

NWD_EPS = 1e-7
JOURNAL_METRICS = ("mAP", "AP50", "AP_small", "AP_medium", "AP_large", "fps", "train_minutes", "epochs")
LOG_MARKERS = {
    "oom_batch_reduced": "Reducing to batch",
    "oom_assigner": "out of memory in TaskAlignedAssigner",
    "nms_time_limit": "NMS time limit",
    "resume_ignored_args": "Resume ignores",
}
CONFIG_ONLY_KEYS = {"runner", "run_id", "base", "experiment"}


def _build_seed(cfg, out_dir):
    """Модель з cfg["arch"] із вагами cfg["model"]; новий рівень лишається свіжим. Повертає (шлях, факти)."""
    import yaml
    from ultralytics import YOLO

    arch = dict(cfg["arch"])
    folder = Path(out_dir) / "dataset"  # не йде в git, light.zip і RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    yaml_path = folder / f"{arch.pop('name')}.yaml"
    yaml_path.write_text(yaml.safe_dump(arch, sort_keys=False), encoding="utf-8")

    source = YOLO(cfg["model"]).model
    built = YOLO(str(yaml_path))
    target = built.model
    src_head, dst_head = len(source.model) - 1, len(target.model) - 1
    level = target.model[-1].nl - 1  # новий рівень стоїть останнім у from Detect

    src_state, dst_state = source.state_dict(), target.state_dict()
    mapped = {}
    for key, value in src_state.items():
        parts = key.split(".")
        if parts[1] == str(src_head):
            parts[1] = str(dst_head)
        new_key = ".".join(parts)
        if new_key not in dst_state or dst_state[new_key].shape != value.shape:
            raise ValueError(f"seed: {key} -> {new_key} не передається: форма чи ім'я не збігаються")
        mapped[new_key] = value

    new_prefixes = tuple(f"model.{i}." for i in range(src_head, dst_head)) + (
        f"model.{dst_head}.cv2.{level}.",
        f"model.{dst_head}.cv3.{level}.",
    )
    unexpected = [key for key in dst_state if key not in mapped and not key.startswith(new_prefixes)]
    if unexpected:
        raise ValueError(f"seed: без ваг лишились старі шари: {unexpected[:5]}")
    target.load_state_dict(mapped, strict=False)
    # Рядки класів передає вже model.load у тренері (_remap_cls_by_names), як і в базі, тому імена COCO лишаємо.
    target.names = dict(source.names)

    head = target.model[-1]
    strides = [float(s) for s in head.stride]
    # Як bias_init для наших класів, інакше рядки нового рівня стартували б із пріора для 80 класів.
    head.cv3[level][-1].bias.data[:] = math.log(5 / data.NUM_CLASSES / (640 / strides[level]) ** 2)

    seed_path = folder / "seed.pt"
    built.save(seed_path)
    facts = {
        "name": cfg["arch"]["name"],
        "transferred": len(mapped),
        "source_tensors": len(src_state),
        "new_tensors": len(dst_state) - len(mapped),
        "strides": strides,
        "detect_inputs": [m[0].conv.in_channels for m in head.cv2],
        "box_hidden": head.cv2[0][-1].in_channels,
        "cls_hidden": head.cv3[0][-1].in_channels,
        "tal_stride_val": strides[1],  # utils/tal.py: stride[1], підлога сторони GT при виборі кандидатів
    }
    print(f"seed: {json.dumps(facts)}")
    return seed_path, facts


def _nwd(pred, target, constant):
    """Normalized Gaussian Wasserstein Distance для рамок xyxy у пікселях, (n, 1)."""
    import torch

    center = (pred[:, :2] + pred[:, 2:]) / 2 - (target[:, :2] + target[:, 2:]) / 2
    half_size = ((pred[:, 2:] - pred[:, :2]) - (target[:, 2:] - target[:, :2])) / 2
    squared = (center * center).sum(-1, keepdim=True) + (half_size * half_size).sum(-1, keepdim=True)
    return torch.exp(-torch.sqrt(squared + NWD_EPS) / constant)


@contextlib.contextmanager
def _nwd_loss(spec, stats):
    """На час рану box-loss = (1 - ratio) * CIoU + ratio * NWD. Без spec нічого не міняє."""
    if not spec:
        yield
        return
    from ultralytics.utils.loss import BboxLoss

    ratio, constant = float(spec["ratio"]), float(spec["constant"])
    original = BboxLoss.forward
    stats.update(ratio=ratio, constant=constant, calls=0)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                imgsz, stride):
        loss_iou, loss_dfl = original(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
            imgsz, stride,
        )
        fg = fg_mask.nonzero(as_tuple=True)
        scale = stride[fg[1]].float()  # (n, 1): у BboxLoss рамки поділені на stride свого рівня
        weight = target_scores[fg].sum(-1, keepdim=True).float()
        nwd = _nwd(pred_bboxes[fg].float() * scale, target_bboxes[fg].float() * scale, constant)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
        stats["calls"] += 1
        if stats["calls"] % 200 == 1 and fg[0].numel():
            stats["last_mean_nwd"] = float(nwd.mean())
            stats["last_loss_ciou"] = float(loss_iou)
            stats["last_loss_nwd"] = float(loss_nwd)
            if stats["calls"] == 1:
                print(f"NWD активний: {json.dumps(stats)}")
        return (1.0 - ratio) * loss_iou + ratio * loss_nwd, loss_dfl

    BboxLoss.forward = forward
    try:
        yield
    finally:
        BboxLoss.forward = original


def _flatten(cfg, prefix=""):
    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and key != "arch":
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _config_diff(cfg):
    """Ключі, якими конфіг відрізняється від бази: доказ, що абляція міняє одну річ."""
    from ..pipeline import load_config

    this = _flatten({k: v for k, v in cfg.items() if k not in CONFIG_ONLY_KEYS})
    base = _flatten({k: v for k, v in load_config(cfg["base"]).items() if k not in CONFIG_ONLY_KEYS})
    return sorted(k for k in set(this) | set(base) if this.get(k) != base.get(k))


def _log_facts(out_dir):
    """Відхилення з ultralytics.log; лог лише поточної сесії, бо відновлення його не переносить."""
    path = Path(out_dir) / "ultralytics.log"
    if not path.exists():
        return {name: None for name in LOG_MARKERS}
    text = path.read_text(encoding="utf-8", errors="replace")
    facts = {name: text.count(marker) for name, marker in LOG_MARKERS.items()}
    facts["transferred_line"] = next((line.strip() for line in text.splitlines() if "Transferred" in line), None)
    return facts


def _model_facts(out_dir, cfg):
    """Що насправді лежить у best.pt."""
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model, _ = load_checkpoint(Path(out_dir) / "weights" / "best.pt")
    head = model.model[-1]
    return {
        "layers": len(model.model),
        "strides": [float(s) for s in head.stride],
        "detect_inputs": [m[0].conv.in_channels for m in head.cv2],
        "cls_hidden": head.cv3[0][-1].in_channels,
        "params": get_num_params(model),
        "gflops_at_imgsz": get_flops(model, imgsz=cfg["imgsz"]),
    }


def _read_metrics(folder):
    path = Path(folder) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _number(value):
    return "-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _write_journal(cfg, out_dir, metrics, base_metrics, checks):
    """experiments.md: що, чому, результат, фейли. Висновок лишається автору."""
    experiment = cfg.get("experiment") or {}
    logs = checks.get("ultralytics_log") or {}
    lines = [
        f"# {cfg['run_id']}",
        "",
        f"- **База:** {cfg['base']}",
        f"- **Що змінено:** {experiment.get('change', '-')}",
        f"- **Чому:** {experiment.get('why', '-')}",
        f"- **Гіпотеза:** {experiment.get('hypothesis', '-')}",
        f"- **Ключі конфігу, відмінні від бази:** {', '.join(checks.get('config_diff') or []) or '-'}",
        "",
        "## Результат (повний val)",
        "",
        "| Метрика | База | Цей ран | Різниця |",
        "|---|---|---|---|",
    ]
    for key in JOURNAL_METRICS:
        mine, base = metrics.get(key), (base_metrics or {}).get(key)
        numeric = isinstance(mine, (int, float)) and isinstance(base, (int, float))
        lines.append(f"| {key} | {_number(base)} | {_number(mine)} | {_number(float(mine - base)) if numeric else '-'} |")
    if base_metrics is None:
        lines += ["", "База на момент рану не порахована: порівняння в `89_compare/compare.md`."]

    deviations = [f"{name}: {count}" for name, count in logs.items() if name in LOG_MARKERS and count]
    if metrics.get("epochs") != cfg["epochs"] and not checks.get("smoke"):
        deviations.append(f"епох {metrics.get('epochs')} замість {cfg['epochs']} (сторож бюджету)")
    if base_metrics and metrics.get("epochs") != base_metrics.get("epochs"):
        deviations.append(f"епох не порівну з базою: {metrics.get('epochs')} проти {base_metrics.get('epochs')}")
    if "nwd" in checks and not checks["nwd"].get("calls"):
        deviations.append("NWD не викликався в цій сесії (ран міг відновитись одразу на фінальну оцінку)")
    lines += ["", "## Фейли й відхилення", ""]
    lines += [f"- {item}" for item in deviations] or ["- у цій сесії не зафіксовано"]
    lines += [
        "",
        "## Перевірка, що зміна застосована",
        "",
        "```json",
        json.dumps({k: v for k, v in checks.items() if k != "config_diff"}, ensure_ascii=False, indent=1),
        "```",
        "",
        "## Висновок (пише автор)",
        "",
    ]
    (Path(out_dir) / "experiments.md").write_text("\n".join(lines), encoding="utf-8")


def _report(cfg, out_dir, smoke, checks):
    """custom_checks.json і experiments.md. Метрики вже записані, тож збій тут не валить ран."""
    checks["smoke"] = smoke
    checks["config_diff"] = _config_diff(cfg)
    checks["ultralytics_log"] = _log_facts(out_dir)
    checks["best_pt"] = _model_facts(out_dir, cfg)
    (Path(out_dir) / "custom_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")

    metrics = _read_metrics(out_dir)
    if cfg.get("arch"):  # ul_common записав шлях seed.pt; імені архітектури досить
        metrics["model"] = cfg["arch"]["name"]
        (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    root = common.SMOKE_RESULTS if smoke else common.RESULTS
    _write_journal(cfg, out_dir, metrics, _read_metrics(root / cfg["base"]), checks)


def run(cfg, out_dir, data_root, smoke):
    out_dir = Path(out_dir).resolve()
    run_cfg = dict(cfg)
    checks = {}
    if cfg.get("arch"):
        seed_path, checks["arch"] = _build_seed(cfg, out_dir)
        run_cfg["model"] = str(seed_path)
    nwd_stats = {}
    with _nwd_loss(cfg.get("nwd"), nwd_stats):
        ul_common.run(run_cfg, out_dir, data_root, smoke)
    if cfg.get("nwd"):
        checks["nwd"] = nwd_stats
    try:
        _report(cfg, out_dir, smoke, checks)
    except Exception as e:
        print(f"custom: звіт не записано: {e!r}")


def load_predictor(cfg, weights):
    """Архітектуру визначає чекпоінт; cfg["model"] тут лише обирає клас YOLO."""
    return ul_common.load_predictor(cfg, weights)
