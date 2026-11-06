# Handoff F (T1a)

**Зроблено:** `src/common.py`, `data.py`, `evaluator.py`, `submission.py`; `pipeline.py` (шаблони, deep merge, `extends`, хвіст логу, traceback у `log.txt`); `hw2.ipynb` (шаблони + `pack_results()`); `_protocol.yaml`; `.gitignore`; `requirements.txt`; правила в `src/CLAUDE.md`. Selfcheck пройшов (faster-coco-eval 1.8.0, реальний val, 571 кадр): GT → усі AP = 1; порожні → 0; 805 детекцій у ignore зі score 0.99 → AP = 1; ті самі поза ignore → mAP 0.991. Перевірка йшла у venv у scratchpad, глобальне середовище не змінене. Решту перевірить Kaggle smoke.

**API** (імпорт у runner: `from src import common, data, evaluator, submission`; порядок кадрів і класи 0–7 скрізь):
- `data.load_frames(data_root, split)`, `data.split_frames(cfg, data_root, split, smoke=False, full=False)`, `data.select_frames(frames, step)` → кадри-dict: `image_id, clip, frame, path, width, height, boxes (xyxy px), labels, ignore (xyxy px), occlusion, truncation, track_id`.
- `data.read_image_bgr(path)`, `read_image_rgb(path)`, `training_targets(frame, drop_ioa=0.5)`, `mask_ignore(image, frame, fill=114, drop_ioa=0.5)`, `CLASS_NAMES`, `NUM_CLASSES`.
- `data.build_ultralytics_dataset(cfg, out_dir, data_root, smoke=False) -> yaml Path` (передавати в `data=`).
- `data.TorchDetectionDataset(frames, max_side=None, mask=True, fill=114, drop_ioa=0.5, label_offset=0, transforms=None)`, `data.collate_detection`. Faster R-CNN: `label_offset=1`, RetinaNet: 0.
- `data.save_preds(preds, path)`, `data.load_preds(path)`; `preds = {image_id: {"boxes", "scores", "labels"}}`.
- `common.Predictor` (`predict(images_bgr, conf)`), `common.predict_frames(predictor, frames, conf, batch_size=8, max_det=500, workers=4) -> preds`, `common.clean_prediction(pred, width, height, max_det=500)`, `common.measure_fps(predictor, frames, conf, n=100)`.
- `common.seed_everything(seed)` (pipeline вже кликає перед runner-ом), `common.seed_worker`, `common.smoke_subset(frames, max_clips=2, max_frames=64)`, `common.append_train_log(out_dir, epoch, train_loss, val_mAP=None, val_AP50=None, minutes=None)`.
- `common.find_weights(run_id, smoke=None)`, `common.find_preds(run_id, smoke=None)`: у SMOKE спершу `results/_smoke`, далі `results`, далі `/kaggle/input/**/<run_id>/`.
- `common.wandb_init(cfg, out_dir, tags=None) -> run` (`.log`, `.finish`, `.summary`; без ключа no-op). `pipeline` сам завершує ран. `common.pack_results()`.
- `evaluator.Evaluator(frames, max_det=500).evaluate(preds) -> dict` (`mAP, AP50, AP75, AP_small, AP_medium, AP_large, AR, n_val_frames, n_gt, n_det, per_size, per_class`). Створюй один раз і клич щоепохи; кадри без preds рахуються порожніми.
- `evaluator.finish_run(predictor, cfg, out_dir, data_root, smoke, train_minutes, epochs, extra=None) -> metrics`: **викликай наприкінці кожного детектора**: повний val, fps, `metrics.json`, `per_size.csv`, `per_class.csv`, `preds_val.json.gz`, `submission.csv`. Лишається runner-у: `train_log.csv`, `weights/best.*`, `plots/`.
- `submission.predict_test(predictor, cfg, out_dir, data_root, smoke=False)`, `write_submission(preds, test_frames, path, max_det=500)`, `prediction_string(pred)`.
- `pipeline.load_config(run_id)`: у `cfg` є `run_id`. Ключі протоколу: `seed, imgsz, frame_step, val_step, mask_ignore, ignore_fill, ignore_drop_ioa, num_workers, budget_minutes, conf, max_det, eval_batch, fps_frames, predict_test, smoke_limits{train_clips, train_frames, val_clips, val_frames, test_clips, test_frames, fps_frames, epochs, max_iters}`.

**Kaggle:** власних `run_id` немає (GPU 0). Перша перевірка `pipeline` і модулів: smoke першого готового рану (K0).

**Рішення (автор може змінити):**
- Один бекенд оцінки, `faster-coco-eval==1.8.0`, бо повний val з 500 детекціями на кадр у pycocotools задовгий для кривої по епохах. Є wheels для Python 3.10–3.13.
- Об'єкти в ignore ≥50 % не йдуть у тренування; регіони заливаються сірим (деталі в `src/CLAUDE.md`).
- Кеш залитих кадрів у системному temp, а не в `out_dir`.
- `preds_val.json.gz` і `submission.csv` не йдуть у git, бо важать десятки MB, але є в `light.zip`. Якщо для здачі треба закомітити фінальний сабміт, це `git add -f`.
- `pack_results` пропускає `weights/`, `dataset/`, `wandb/`.

**Факти для наступних:** `frame_step=3` дає 8090 кадрів train, `val_step=5` дає 571 кадр val; повний val 2846. Розміри кадрів: train до 2720×1530, val до 3840×2160, test лише до 1920×1080. У пікселях оригіналу test дрібніший за val, тож `AP_small` на val не переноситься на test напряму (EDA варто глянути). Кожен ignore-регіон у COCO повторений для 8 категорій: `data` лишає одну копію.

**Ризики і застереження:**
- UL: куди пише логер Ultralytics, залежить від місця першого імпорту. `ul_common` імпортує його всередині `redirect_stdout`, тож див. UL-1 у `handoff/RV.md`.
- Ultralytics не може писати `.cache` в `/kaggle/input` для val: лише warning, ран не падає (перевірено за кодом 8.4.203).
- **Поведінка змінена (M1):** без SMOKE упалий ран (і помилка конфігу чи імпорту runner-а) не зупиняє Run All: друкується хвіст логу, ран іде в `common.FAILED`, сесія йде далі, статус ноутбука успішний. Дивись «УВАГА, впали рани» у виводі `pack_results`. `light.zip` оновлюється після кожного рану, атомарно. У SMOKE виняток летить одразу, Ctrl+C завжди зупиняє.
- Кадри в батчі `Predictor` можуть мати різний розмір (межа кліпів).
- Опис метрики Kaggle так і не перевірено: пункт автора у `DECISIONS.md`.

**Правки чужих файлів:** `KAGGLE_RUNBOOK.md`: успішний статус версії не означає, що всі рани пройшли, треба дивитись `FAILED`. `tasks/STATE.md` змінений не мною.
**requirements:** `pyyaml==6.0.3`, `faster-coco-eval==1.8.0`, `pandas==2.3.3`, `matplotlib==3.10.8`, `wandb==0.26.1`.
