# T2. Бейзлайн і перший сабміт

Найпростіша модель проходить весь ланцюжок: клонування на Kaggle → тренування → оцінка → `submission.csv` → скор на лідерборді. Мета не скор, а робочий ланцюжок.

**Готово, коли:** сабміт прийнято на Kaggle, а результати повернулись у `workspace/results/`.

## Оркестрація

- **Лейн UL · контекстний блок 1 (YOLO) · модель sonnet · хвиля W1 · залежить від T1a.**
- **Власні файли:** `src/models/ul_common.py` (спільна обгортка Ultralytics: підготовка датасету з `data.py`, тренування, збір `metrics.json`/`train_log.csv`/`per_size.csv` через `src.evaluator`, `Predictor`), `src/models/baseline.py`, `configs/05_baseline.yaml`.
- **Ідея:** найдешевша модель, що проходить увесь ланцюжок до `submission.csv` (наприклад, найменший YOLO зі спільним протоколом). Мета: перевірити Kaggle-ланцюжок, а не скор. Оцінка тільки через `src.evaluator`; власний mAP Ultralytics можна лише логувати.
- **Критично:** `ul_common` буде основою T5 і T6, тож пиши його без прив'язки до YOLO (модель з конфігу: `model: yolo11s.pt` або `rtdetr-l.pt`), а per-epoch val mAP бери з evaluator на кроці `val_step`.
- **Виходи:** `handoff/UL.md`, ран `05_baseline`.
- **GPU:** ≈0.5 год. **Smoke:** обов'язковий, ≤3 хв.
