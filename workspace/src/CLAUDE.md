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
