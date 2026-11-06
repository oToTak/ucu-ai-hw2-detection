# Handoff F (T1a + Colab)

**Зроблено:** `src/common.py`, `data.py`, `evaluator.py`, `submission.py`, `pipeline.py`, `colab.py`; `workspace/colab/*.ipynb`, `make_data_zip.py`; `_protocol.yaml`; `.gitignore`; `requirements.txt`; правила в `src/CLAUDE.md`. Selfcheck оцінювача пройшов (571 кадр val): GT → AP = 1; порожні → 0; детекції в ignore не штрафуються, поза ignore штрафуються. Colab-код перевірено лише `py_compile`, json і `load_config`: на Colab не запускався.

**API** (імпорт у runner: `from src import common, data, evaluator, submission`; класи 0–7 скрізь):
- `data.load_frames(data_root, split)`, `split_frames(cfg, data_root, split, smoke=False, full=False)`, `select_frames(frames, step)` → кадри-dict: `image_id, clip, frame, path, width, height, boxes (xyxy px), labels, ignore (xyxy px), occlusion, truncation, track_id`. `read_image_bgr(path)`, `read_image_rgb(path)`, `training_targets(frame, drop_ioa=0.5)`, `mask_ignore(image, frame, fill=114, drop_ioa=0.5)`, `CLASS_NAMES`, `NUM_CLASSES`.
- `data.build_ultralytics_dataset(cfg, out_dir, data_root, smoke=False) -> yaml Path`; `TorchDetectionDataset(frames, max_side=None, mask=True, fill=114, drop_ioa=0.5, label_offset=0, transforms=None)`, `collate_detection` (Faster R-CNN `label_offset=1`, RetinaNet 0); `save_preds(preds, path)`, `load_preds(path)`.
- `common.Predictor` (`predict(images_bgr, conf)`), `predict_frames(predictor, frames, conf, batch_size=8, max_det=500, workers=4) -> preds`, `clean_prediction(pred, width, height, max_det=500)`, `measure_fps(predictor, frames, conf, n=100)`, `seed_everything(seed)` (pipeline кличе сам), `seed_worker`, `smoke_subset(frames, max_clips=2, max_frames=64)`, `append_train_log(out_dir, epoch, train_loss, val_mAP=None, val_AP50=None, minutes=None)`, `find_weights(run_id, smoke=None)`, `find_preds(run_id, smoke=None)`, `wandb_init(cfg, out_dir, tags=None) -> run`, `pack_results()`.
- `evaluator.Evaluator(frames, max_det=500).evaluate(preds) -> dict`. `evaluator.finish_run(predictor, cfg, out_dir, data_root, smoke, train_minutes, epochs, extra=None) -> metrics`: **викликай наприкінці кожного детектора**. Лишається runner-у: `train_log.csv`, `weights/best.*`, `plots/`.
- `submission.predict_test(predictor, cfg, out_dir, data_root, smoke=False)`, `write_submission(preds, test_frames, path, max_det=500)`.
- `pipeline.load_config(run_id)`; ключі протоколу: `seed, imgsz, frame_step, val_step, mask_ignore, ignore_fill, ignore_drop_ioa, num_workers (обмежено cpu_count), budget_minutes, ckpt_sync_minutes, conf, max_det, eval_batch, fps_frames, predict_test, smoke_limits{...}`.

**Відновлення посеред рану (для UL і TV; вбудувати у свої цикли):**
- `common.load_progress(out_dir) -> dict | None`: на початку `run()`. None означає починати спочатку. Інакше це стан **останнього чекпоінта, що доїхав на Drive**, а не найновіший; файли чекпоінта вже лежать на тих самих шляхах в `out_dir`, а `train_log.csv` обрізано до `state["epoch"]`.
- `common.save_progress(out_dir, state, files=(), force=False) -> None`: після кожної епохи, **після** `append_train_log`. `state` JSON-сумісний: домовленість `{"epoch": int, "minutes": float, "stage": str}`, ключі на `_` зайняті. `files`: шляхи всередині `out_dir` (абсолютні чи відносні), інакше ValueError. Щоразу пише `state.json`, `train_log.csv`, `meta.json`, `config.yaml`, `log.txt`; `files` у фоні, не частіше ніж раз на `ckpt_sync_minutes`. `force=True` негайно: кличте при зміні етапу (кінець тренування перед оцінкою). Помилка Drive лише друкується.
- `common.wait_progress()`: pipeline кличе сам. `common.progress_selfcheck` теж: у SMOKE на Colab зберігає й відтворює чекпоінт у `_smoke/_wip`.
- Час: `minutes = state["minutes"] + (time.time() - session_start) / 60`; саме це порівнювати з `budget_minutes` і передавати як `train_minutes`. Епохи після останнього чекпоінта перераховуються, їхні хвилини не враховані.
- Чекпоінт має містити все для продовження (модель, EMA, оптимізатор, scaler, епоху). Файли, передані у `files`, у `RESULTS` не йдуть, крім `best.*`. Не клади `best` окремо від `weights/best.*`.
- Поза Colab (немає `HW2_WIP`) і локальний SMOKE: `save_progress` нічого не робить, `load_progress` дає None. Жодних гілок у ранері не треба.

**Colab-контракт:** ноутбуки `1_baseline_twostage`, `2_anchor_custom`, `3_yolo_temporal_viz`, `4_detr_final`; логіка в `colab.setup()`, `colab.run(compute_list, smoke=False, force=False)`, `colab.finish()`. `setup` виставляє env: `HW2_RESULTS=HUB/results`, `HW2_WIP=HUB/wip`, `HW2_PACK_DIR=HUB`, `HW2_DATA=/content/data`, `HW2_WORK=/content/hw2_work`, `HW2_SMOKE_RESULTS=/content/results/_smoke`. `common.RESULTS`, `common.SMOKE_RESULTS` єдине джерело шляхів. Ран рахується в `HW2_WORK/<ран>`, успішний їде в `RESULTS/<ран>` через `_partial_<ран>`, упалий лишає логи в `RESULTS/_failed/<ран>` і свій `wip`. `force=True` або список шаблонів: рахувати заново, `wip` стирається. На Colab pip не змінює наявні `numpy, pandas, matplotlib, torch, torchvision, pyyaml, wandb` (constraints). `light.zip` у HUB без `preds*` і `submission*`: сабміти брати з `hw2/results/<ран>/submission.csv`. У SMOKE на Drive йде лише `light_smoke.zip`.

**data.zip:** `python workspace/colab/make_data_zip.py <шлях поза репо>`; `--dry-run` рахує розмір (при `frame_step=3`: 5.60 GB, 61323 файли). У train лише кадри `frame_step` (кратні кроки теж працюють); `data.py` читає список кадрів з анотацій, а не з диска, тож відсутні кадри йому не заважають, доки їх не вибрали. Зміна `frame_step` на некратний вимагає нового zip.

**Рішення (автор може змінити):**
- Пропуск готових ранів лише з `HW2_WIP`: на Kaggle й локально явний `COMPUTE` рахує заново, як раніше, бо `hw2.ipynb` не може передати `force`.
- Оцінка: `faster-coco-eval==1.8.0`. Об'єкти в ignore ≥50 % не тренуються, регіони заливаються сірим. Кеш залитих кадрів у системному temp. `preds_val.json.gz`, `submission.csv` не в git.

**Факти:** `frame_step=3` дає 8090 кадрів train, `val_step=5` дає 571 кадр val; повний val 2846. Кадри: train до 2720×1530, val до 3840×2160, test до 1920×1080, тож `AP_small` на val не переноситься на test напряму.

**Ризики:**
- Drive синхронізується асинхронно: після дисконекту останні записи можуть не доїхати, порядок не гарантований. `load_progress` звіряє розміри і при розбіжності починає спочатку.
- Не перевірено: чи йдуть перезаписані й видалені файли Drive у Кошик і чи з'їдають квоту. Глянути Кошик після першого довгого рану; важкі `last.pt` це головний кандидат.
- `metadata.accelerator` і `colab.gpuType`: офіційної схеми не знайшов, лише приклади. `setup` зупиняється без GPU, тож помилка буде видна.
- `drive.mount` питає дозвіл у кожному новому runtime: один клік автора. `finish()` відмонтовує Drive після справжніх ранів.
- Після `git pull` у живому kernel потрібен Restart session (`setup` попереджає).
- Падіння рану не зупиняє Run All і статус успішний: дивись «УВАГА, впали рани» у виводі `pack_results`. У SMOKE виняток летить одразу.
- Ultralytics: UL-1 у `handoff/RV.md`. Опис метрики Kaggle не перевірено.

**Правки чужих файлів:** `src/compare.py`, `src/summary.py` (AG): `RESULTS`, `SMOKE_RESULTS` тепер із `common`, за прямою вимогою координатора. Оркестратору: `KAGGLE_RUNBOOK.md`, `PLAN.md` про Colab і `FAILED`.
