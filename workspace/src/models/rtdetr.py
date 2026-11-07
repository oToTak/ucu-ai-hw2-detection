"""T6, query-based: RT-DETR (Ultralytics) через ul_common. Конфіги 40-42.

ul_common.run робить усе (train з відновленням, оцінка, submission). Тут лише те, чого
Ultralytics 8.4.203 не дає аргументами тренування (звірено з вихідниками wheel):
  - num_queries: аргумент `nq` голови RTDETRDecoder, лише в yaml моделі. Пишемо копію rtdetr-l.yaml
    в out_dir з потрібним nq, ваги беремо з rtdetr-l.pt через train_args.pretrained (форми тензорів
    не залежать від nq, бо learnt_init_query=False, тож переноситься все, окрім голів класів 80->8).
  - окремий LR backbone: аргументу немає, тому підклас RTDETRTrainer.build_optimizer ділить кожну
    групу параметрів на backbone/решту. initial_lr береться планувальником із lr групи, тож розклад
    і warmup зберігають співвідношення. Структура груп детермінована, resume з чекпоінта коректний.
Ключі конфігу (блок `rtdetr`, усе необов'язкове):
    num_queries: int | null          # null = 300 (дефолт rtdetr-l)
    backbone_lr_mult: float          # 1.0 = один LR на всю модель
    converged_min_map: float         # поріг для `converged` у metrics.json (за val_mAP на val_step)
Класи loss/matching (loss_gain, cost_gain) у 8.4.203 жорстко задані в DETRLoss.__init__ і не є
аргументами; не міняємо.
"""

import copy
import csv
import json
from pathlib import Path

from . import ul_common


def _write_queries_yaml(base_model, num_queries, out_dir):
    """Копія yaml архітектури rtdetr-l з іншим nq. Повертає шлях (ім'я містить 'rtdetr' для _model_class)."""
    import ultralytics
    import yaml

    stem = Path(str(base_model)).stem  # rtdetr-l
    src = Path(ultralytics.__file__).parent / "cfg" / "models" / "rt-detr" / f"{stem}.yaml"
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    head = d["head"][-1]
    assert head[2] == "RTDETRDecoder", head
    head[3] = ["nc", 256, int(num_queries)]  # nc, (ch вставляє parse_model), hd=256, nq
    path = Path(out_dir) / f"rtdetr_nq{int(num_queries)}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return path


def _backbone_lr_trainer(mult):
    """RTDETRTrainer із build_optimizer, що дає backbone lr * mult."""
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class BackboneLRTrainer(RTDETRTrainer):
        def build_optimizer(self, *args, **kwargs):
            optimizer = super().build_optimizer(*args, **kwargs)
            net = unwrap_model(self.model)
            n_backbone = len(net.yaml["backbone"])
            backbone = {id(p) for p in net.model[:n_backbone].parameters()}
            assert backbone, "backbone порожній"
            for group in list(optimizer.param_groups):
                inside = [p for p in group["params"] if id(p) in backbone]
                if not inside:
                    continue
                outside = [p for p in group["params"] if id(p) not in backbone]
                if not outside:
                    group["lr"] *= mult
                    continue
                new = {k: v for k, v in group.items() if k != "params"}
                new["lr"] = group["lr"] * mult
                new["params"] = inside
                group["params"] = outside
                optimizer.add_param_group(new)
            print(f"backbone_lr_mult={mult}: груп оптимізатора {len(optimizer.param_groups)}")
            return optimizer

    return BackboneLRTrainer


def _converged(out_dir, rt, smoke):
    """Дописує converged і пояснення в metrics.json (ран зберігається в будь-якому разі)."""
    out_dir = Path(out_dir)
    path = out_dir / "metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    vals = []
    log = out_dir / "train_log.csv"
    if log.exists():
        with open(log, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    vals.append(float(r["val_mAP"]))
                except (KeyError, ValueError):
                    pass
    threshold = float(rt.get("converged_min_map", 0.02))
    best = max(vals) if vals else float("nan")
    ok = bool(vals) and best >= threshold and vals[-1] >= 0.5 * best
    if smoke:
        why = "SMOKE: 1 епоха, висновку про збіжність немає"
    elif ok:
        why = f"найкращий val_mAP {best:.4f} >= {threshold}, остання епоха не обвалилась"
    else:
        why = (f"найкращий val_mAP {best:.4f} (поріг {threshold}), остання {vals[-1] if vals else float('nan'):.4f}; "
               f"за {len(vals)} епох не збіглась або обвалилась")
    metrics.update(
        {
            "converged": ok and not smoke,
            "converged_note": why,
            "num_queries": rt.get("num_queries") or 300,
            "backbone_lr_mult": rt.get("backbone_lr_mult", 1.0),
        }
    )
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run(cfg, out_dir, data_root, smoke):
    cfg = copy.deepcopy(cfg)
    rt = cfg.get("rtdetr") or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nq = rt.get("num_queries")
    if nq:
        base_model = cfg["model"]
        cfg["model"] = str(_write_queries_yaml(base_model, nq, out_dir))
        cfg.setdefault("train_args", {})["pretrained"] = str(base_model)  # ваги з rtdetr-l.pt

    mult = float(rt.get("backbone_lr_mult", 1.0))
    patched = None
    if mult != 1.0:
        import ultralytics.models.rtdetr.model as rtdetr_model

        patched = (rtdetr_model, rtdetr_model.RTDETRTrainer)
        rtdetr_model.RTDETRTrainer = _backbone_lr_trainer(mult)  # task_map читає це ім'я при кожному train()
    try:
        ul_common.run(cfg, out_dir, data_root, smoke)
    finally:
        if patched:
            patched[0].RTDETRTrainer = patched[1]
    _converged(out_dir, rt, smoke)


def load_predictor(cfg, weights):
    return ul_common.load_predictor(cfg, weights)
