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

**Colab-відновлення (вбудовано в `ul_common.run`):** на початку `common.load_progress(out_dir)`; колбек кожної епохи після `append_train_log` кличе `common.save_progress(out_dir, {epoch, minutes, stage, best_mAP}, files=[last.pt, weights/best.pt])`, `force` при завершенні навчання. Стани: `stage="train"` і є `dataset/ultralytics/train/weights/last.pt` -> `YOLO(last.pt).train(resume=True, data=<той самий data.yaml>, device, workers)` (решта аргументів з чекпоінта, шляхи абсолютні й однакові на новій VM; data.yaml і кеш перебудовуються детерміновано); `stage="trained"` і є `best.pt` -> `train` не викликається, одразу фінальна оцінка й сабміт; інакше старт з нуля (старий `train_log.csv` видаляється). Хвилини накопичуються (`start = now - minutes`), сторож бюджету рахує сумарно, повтор епохи (`epoch <= epochs_done`) у лог не пишеться. Фінальна оцінка/сабміт при дисконекті на цьому етапі просто перераховуються. Не перевірено реально: Ultralytics `resume` зі звірених вихідників (`check_resume`, `Model.train`), запуску на Colab не було. Ризик: `resume` ігнорує зміни `epochs`/`train_args` (бере з чекпоінта); для зміни починати новий ран.

**Правки чужих файлів:** немає.
**requirements:** `ultralytics==8.4.203`.
**Автору:** запустити `SMOKE = True` з `COMPUTE=["05_baseline"]`, потім справжній ран, здати `results/05_baseline/submission.csv` на Kaggle і покласти light.zip у `workspace/results/`.

## T5 (YOLO, W2)
**Зроблено:** `src/models/yolo.py` (делегат у `ul_common`, як `baseline.py`; `ul_common` не чіпав), конфіги `30_yolo_base`, `31_yolo_imgsz` (1280), `31_yolo_imgsz_640`, `32_yolo_loss`, `39_compare`. Перевірено лише py_compile і `load_config`. Відновлення посеред рану те саме, що в baseline (вбудоване в `ul_common.run`).
- `30_yolo_base`: yolo11s, 10 епох, batch 16, imgsz 960, дефолтні gains виписані явно. Оцінка ≈1 год на T4: НЕ заміряно, підігнати `epochs`.
- Експеримент 1, роздільність (діапазон 640 / 960 / 1280): `31_yolo_imgsz_640`, база 960, `31_yolo_imgsz` (1280, batch 8, `budget_minutes: 130`). Однакові `epochs` для чесності, тож бюджет 1280 піднято замість зменшення епох (≈1.8x вартість). Другий файл `31_*_640` доданий понад ТЗ (один yaml = один експеримент). Це перевіряє рішення про imgsz 960 зі STATE.md.
- Експеримент 2, loss: `32_yolo_loss`: `cls` 0.5->1.0, `cls_pw` 0->0.5 (ваги класів за зворотною частотою). Діапазон cls 0.5-1.5, cls_pw 0-1. `tal_topk` у 8.4.203 не аргумент тренування (жорстко 10 в `utils/loss.py`), тому assigner не міняли.
- `39_compare`: 05, 30, 31x2, 32 + 10/11/12 + 20_retina_base; відсутні пропускаються. Додати інші `2x`, коли з'являться.
- Орієнтир GPU: ≈1 + 1.8 + 0.45 + 1 одиниці по ≈1 год, тобто ≈4 год (ТЗ ≈3.75).
- Ризик: 1280 batch 8 на T4 може не вміститись у пам'ять (тоді batch 4). Рани 3x читають `AP_small` на val (19% малих), не переноситься на test (STATE.md).
- UL-4/UL-5 лишаються відкладеними до виміру `mAP` проти `mAP_native_last`.

## T6 (RT-DETR, W3)
**Зроблено:** `src/models/rtdetr.py` (обгортка над `ul_common.run`, `ul_common` не чіпав), конфіги `40_rtdetr_base`, `41_rtdetr_queries`, `42_rtdetr_lr`, `49_compare`. Перевірено лише py_compile і `load_config`; не запускалось. Відновлення посеред рану те саме (вбудоване в `ul_common.run`).
- Що Ultralytics 8.4.203 реально застосовує (за вихідниками): `num_queries` НЕ аргумент тренування, лише `nq` голови в yaml моделі; окремого LR backbone немає; ваги loss/matching (class 1, bbox 5, giou 2; cost 2/5/2) жорстко в `DETRLoss.__init__`, тож не міняємо. Тому блок конфігу `rtdetr: {num_queries, backbone_lr_mult, converged_min_map}` обробляє `rtdetr.py`.
- `41`: пише копію `rtdetr-l.yaml` з `nq=150` в `out_dir/rtdetr_nq150.yaml`, ваги з `rtdetr-l.pt` через `train_args.pretrained` (форми не залежать від nq). Обґрунтування з EDA: об'єктів на кадр train mean 46, p95 118, p99 142, max 180.
- `42`: на час рану підміняє `ultralytics.models.rtdetr.model.RTDETRTrainer` підкласом із `build_optimizer`, що ділить групи параметрів на backbone (`yaml["backbone"]`, шари 0-9) і решту; backbone lr x0.1. Підміна знімається в `finally`.
- `metrics.json` отримує `converged`, `converged_note`, `num_queries`, `backbone_lr_mult` (поріг: найкращий val_mAP ≥ 0.02 і остання ≥ половини найкращої; у SMOKE завжди false).
- Гіперпараметри (рецепт RT-DETR, не auto): AdamW lr0 1e-4, wd 1e-4, warmup 0.5 епохи, warmup_bias_lr 0, close_mosaic 1, nbs 16 (при batch 4 це x4 акумуляція, ≈505 кроків/епоху).
- `40/41/42`: rtdetr-l, 4 епохи, batch 4, imgsz 960, `eval_batch` 4, бюджет 75 хв (протокол). ЕПОХА НЕ ЗАМІРЯНА: оцінка 15-25 хв на T4; ≈1.25 год на ран, ≈3.75 год разом. Якщо епоха >20 хв, сторож бюджету зупинить раніше, а LR не доспаде (тоді зменшити `epochs`). `49_compare`: 10, 20, 30, 40, 41, 42 (відсутні пропускаються).
**Ризики:** (1) RT-DETR у Ultralytics розтягує кадр у квадрат 960x960 (scale_fill і в тренуванні, і в predictor), тобто дорожче за 960x540 і з іншою геометрією, ніж у CNN. (2) batch 16 не влізе при 960; 4 обрано консервативно, 8 можна спробувати. (3) AMP + Hungarian matching: можливі NaN (є автовідновлення до 3 разів, далі ран падає). (4) Макс. детекцій на кадр = nq (300 або 150 < 500). (5) Подвійна валідація щоепохи ≈2 хв, з 4-ма епохами це ≈10% бюджету. (6) `resume` ігнорує зміну epochs/train_args. (7) `predict_args.iou` не задано (NMS немає). (8) CPU Colab (2 vCPU) може бути вузьким місцем: мозаїка 960 і Hungarian на CPU.
**Правка чужих файлів:** `src/compare.py` (AG): графік `compare_curves` по епохах не порівнює 4 епохи RT-DETR з 10 епохами YOLO; бажано додати x = `minutes` з `train_log.csv`.
**requirements:** без змін. **Автору:** `COMPUTE=["40_rtdetr_base"]` у SMOKE, глянути час епохи і пам'ять, підігнати `epochs`/`batch`, потім 40, 41, 42, 49.
