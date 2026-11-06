# Контракт коду

`pipeline.py` читає `configs/_protocol.yaml`, зверху накладає `configs/<ран>.yaml` і викликає модуль `src.<runner>`, де `runner` задано в конфігу.

Кожен runner має функцію:

```python
def run(cfg: dict, out_dir: Path, data_root: Path, smoke: bool) -> None: ...
```

Runner пише в `out_dir`:
- `metrics.json`: для детекторів обов'язково mAP, AP50, AP_small, fps;
- `plots/*.png`: їх показує `show()`;
- усе інше, що потрібно (`per_size.csv`, `per_class.csv`, `train_log.csv`, `submission.csv`).

Усі детектори оцінюються одним спільним оцінювачем, щоб порівняння було чесним. Друк runner-а йде в `log.txt`, а не в ноутбук.

## Розширення контракту (оркестрація)

Власник цього розділу і спільних модулів: лейн F. Решта лише читає. Деталі виконання: `tasks/_common.md`.

**Спільні модулі** (пише F у T1a): `common.py` (`seed_everything`, `smoke_subset`, `find_weights(run_id)`, `wandb_init`, `Predictor`, `pack_results`), `data.py` (читання split-ів, вибірка кадрів за кроком, ignore-регіони, датасети для Ultralytics і torchvision; усе, що треба писати, лягає в `out_dir`), `evaluator.py`, `submission.py`. `pipeline.py` приймає шаблони в `compute`/`show`/`COMPUTE` (`"1?_*"`), глибоко зливає конфіги, підтримує `extends: <ран>` і друкує хвіст `log.txt` при помилці.

**Обов'язкові виходи кожного детектора в `out_dir`:**
- `metrics.json`: `mAP, AP50, AP_small, AP_medium, AP_large, fps, train_minutes, epochs, imgsz, n_val_frames` (ключі саме такі);
- `train_log.csv`: рядок на епоху `epoch, train_loss, val_mAP, val_AP50, minutes`;
- `per_size.csv` (small/medium/large), `per_class.csv`;
- `preds_val.json.gz`: передбачення на val (формат фіксує F у `data.py`); `weights/best.*` (у git не йдуть);
- `submission.csv` для test (`predict_test: true` у конфігу; у SMOKE кілька кадрів).

**Predictor.** Кожен модуль моделі має `load_predictor(cfg, weights) -> Predictor`; `predict(images_bgr, conf)` повертає для кожного кадру `boxes` (xyxy, пікселі оригіналу), `scores`, `labels` (0–7). На ньому стоять SAHI/TTA/WBF, temporal, візуалізація.

**Координати й класи.** Оцінка завжди в пікселях оригінального кадру, незалежно від розміру входу моделі. Класи 0–7 усюди; 1–8 живуть лише всередині `evaluator`.

**SMOKE.** ≤2 кліпи, ≤64 кадри, ≤1 епоха, ≤3 хв, але всі виходи записані. Ран, що читає ваги іншого рану, у SMOKE бере їх із `results/_smoke/`.

**Залежності.** `requirements.txt` лише дописується (піновані версії). `torch`, `torchvision`, `numpy` не додавати.
