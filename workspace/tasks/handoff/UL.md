# Handoff UL (T2)

**Зроблено:** `src/models/ul_common.py`, `src/models/baseline.py`, `configs/05_baseline.yaml`; `ultralytics==8.4.203  # UL` у requirements. Перевірено лише `py_compile` і `load_config("05_baseline")`. Сигнатури звірені з вихідним кодом wheel 8.4.203 (trainer, predictor, default.yaml), не лише з docs. Перший реальний запуск: Kaggle smoke.

**API `ul_common` (для T5 YOLO, T6 RT-DETR):**
- `run(cfg, out_dir, data_root, smoke)`: весь цикл. Рант-модуль = 2 функції-делегати, як у `baseline.py` (`run`, `load_predictor`). Для T5/T6 достатньо скопіювати `baseline.py` у `yolo.py`/`rtdetr.py` і покласти yaml з `runner: models.yolo`.
- `load_predictor(cfg, weights) -> UltralyticsPredictor` (`predict(images_bgr, conf)`, класи 0-7, рамки в пікселях оригіналу; `imgsz`, `max_det`, `device`, `predict_args` з cfg). Для SAHI/TTA/WBF/temporal.
- Клас моделі за назвою: містить `rtdetr` -> `RTDETR`, інакше `YOLO`. Власний `.yaml` теж іде в YOLO.
- Ключі конфігу: `model, epochs, batch, device, train_args{}, predict_args{}`; решта з `_protocol.yaml`. `train_args` кидається в `model.train(**)` останнім і перекриває все (lr0, optimizer, mosaic, fraction...). Deep merge: додаючи ключ у `extends`-ран, решта лишається.

**Що робить `run`:** `data.build_ultralytics_dataset` -> `model.train` (callback `on_fit_epoch_end`: predictor з `trainer.last` -> `predict_frames` -> `Evaluator` на `val_step` -> `append_train_log` + W&B) -> `best.pt` = last.pt з найкращим НАШИМ val_mAP (копія в `weights/best.pt`, потім `strip_optimizer`) -> `evaluator.finish_run` (повний val, fps, submission.csv) -> `plots/train_curves.png`. Нативний mAP Ultralytics (val=True) іде лише в `metrics.json` як `mAP_native_last` і в W&B.
- Робочий каталог Ultralytics: `out_dir/dataset/ultralytics/train` (поза git і light.zip). Лог Ultralytics дублюється в `out_dir/ultralytics.log`.
- Ultralytics wandb вимкнено (`settings.update`), свій через `common.wandb_init`.

**Рішення (автор може змінити):**
- yolo11n.pt, 8 епох, batch 16, imgsz 960 з протоколу; `close_mosaic: 2` (дефолт 10 > epochs і мозаїка б не вимикалась). Оцінка ≈0.5 GPU-год (епоха ≈2-3 хв + наш val ≈0.5 хв): ЗАПЛАНОВАНО, не заміряно, глянь час на smoke/першому запуску.
- `deterministic=False` (швидше, без падінь на деяких ops), `plots=False` у Ultralytics.
- Подвійна валідація щоепохи (нативна Ultralytics потрібна її тренеру для fitness/stopper, наша для кривої). Дорожче на ~20-30 с/епоху.

**Ризики:**
- Перший запуск на Kaggle: AMP-check тягне yolo11n.pt з інтернету (потрібен Internet On).
- Колбек відпрацьовує і на `final_eval` (epoch == epochs): відсікається умовою `trainer.epoch >= trainer.epochs`.
- На 2xT4 `device=None` -> лише GPU 0 (мультиGPU DDP не вмикали).
- NMS time limit-warning при conf 0.001 у Ultralytics (рання епоха, багато боксів) може обрізати детекції: лише на старті; гляди в `ultralytics.log`.
- Predictor тримає `YOLO(...)` на GPU; у колбеку після eval `del` + `empty_cache`; для RT-DETR-l (T6) пам'ять може бути тісною: тоді зменшити `eval_batch`.
- Для RT-DETR `iou`/NMS-аргументи не діють; `predict_args.iou` там прибрати.

**Після review (RV.md):** виправлено B1 (`tloss` dict), UL-1 (stream-handler на живий stdout), UL-2 (вимкнено raytune/mlflow/comet/clearml/dvc), UL-3 (бюджет: колбек ставить `trainer.stop`, якщо наступна епоха не вкладається в `budget_minutes - reserve_minutes` (дефолт 12); сторож через `state["finished"]`). Свідомо не робив UL-4 (multi-label NMS) і UL-5 (PIL vs cv2 resize): обидва умовні, спершу заміряти різницю на `05_baseline` (`mAP` проти `mAP_native_last`). Сигнатури не змінились.

**Правки чужих файлів:** немає.
**requirements:** `ultralytics==8.4.203`.
**Автору:** запустити `SMOKE = True` з `COMPUTE=["05_baseline"]`, потім справжній ран, здати `results/05_baseline/submission.csv` на Kaggle і покласти light.zip у `workspace/results/`.
