# Handoff PP (T7 етап 1, код)

**Зроблено:** `src/infer_tricks.py`, `src/final.py`, `configs/50_final.yaml` (плейсхолдер). Перевірено: `py_compile`, `pipeline.load_config("50_final")`, одноразова перевірка WBF/NMS/Soft-NMS/нарізання/TrickPredictor на фейковому Predictor (без torch; WBF дає за статтею 0.85 і 0.25). На Colab і Kaggle не запускалось.

**API `infer_tricks`** (лише numpy, класи 0-7, пікселі оригіналу):
- `nms(pred, iou_thr, metric)`, `soft_nms(pred, method, sigma, iou_thr, score_thr)`, `wbf(sources, weights, iou_thr, skip_box_thr, conf_type)`, `merge_predictions(sources, cfg, weights, max_det)`; `slice_boxes(w, h, size, overlap)`; `input_size(predictor, size)`; pred це dict `boxes, scores, labels`.
- `TrickPredictor(predictors, scales, flip, sahi, merge, model_weights, max_det)` і `.from_config(predictors, tricks, max_det, model_weights)`: сам є `Predictor`. Джерела злиття: модель x (масштаб x флип) по цілому кадру, плюс один набір тайлів на модель.
- Масштаб і `slice_imgsz` міняють `predictor.imgsz` та `model.transform.min_size/max_size` (torchvision) тимчасово. Predictor без `imgsz` падає з ValueError.

**Runner `final`** (`src/final.py`): `weights_from: <ран> | [<ран>, ...]`; конфіг базового рану береться через `pipeline.load_config`, модуль через його `runner`, ваги через `common.find_weights(run_id, smoke)`. Список ранів = ансамбль (WBF між моделями, `model_weights` необов'язково). Виходи: `metrics.json` (`train_minutes=0, epochs=0`, `extra`: `weights_from, tricks, infer_minutes, val_full`), `per_size/per_class.csv`, `preds_val.json.gz`, `submission.csv`, `plots/final_ap.png`. `train_log.csv` і `weights/` немає (не тренує). Відновлення: `_val_partial.json.gz`, `_test_partial.json.gz` через `save_progress` не рідше ніж раз на `ckpt_sync_minutes`; після успіху видаляються.

**Run_id для Colab:** `50_final` (плейсхолдер на `10_frcnn_base`, усі трюки ввімкнені: TTA флип + масштаби 1.0 і 1.25, SAHI 640/0.2, WBF 0.6). Спершу smoke. Оцінка часу не відома: SAHI дає десятки тайлів на кадр, val повний (2846 кадрів, до 3840x2160) плюс test, тож орієнтир години, а не хвилини. Зменшити можна `val_full: false`.

**Етап 2 (не робив):** `51_final_sahi`, `52_final_wbf` робляться як `extends: 50_final` зі зміною `tricks.*.enabled`; для ефекту моделі різні `weights_from`. `59_summary` і вибір двох сабмітів (D9) після результатів 1x-4x.

**Ризики:**
- Менш за все перевірено: `input_size` для torchvision (у frcnn `min_size=max_size=imgsz` вбудовано в `model.transform`, тому міняю саме їх) і для Ultralytics (лише `imgsz`). Smoke покаже.
- WBF на python-циклі: до секунд на кадр при тисячах рамок, виконується на CPU між викликами GPU. Якщо повільно, підняти `merge.skip_box_thr` (наприклад 0.01) або `source_max_det`.
- Розмір кадрів різний (val до 3840, test до 1920), тож `slice_size` у пікселях оригіналу діє по-різному; `AP_small` на val не переноситься на test напряму.
- Тайл-краї: `edge_px` відкидає рамки на внутрішньому краю тайла; об'єкти більші за перекриття (128 px) покриває лише прохід по цілому кадру.
- `val_full: true` у SMOKE обмежується `smoke_limits`, як усюди.

**Правки чужих файлів:** немає. Залежності в `requirements.txt` не додавав.

---

# B1 temporal (той самий лейн)

**Зроблено:** `src/temporal.py`, `configs/60_temporal.yaml`. Перевірено `py_compile`, `load_config` і логіку на фейкових кліпах (згладжування, штраф коротких треків, прокидання, розрив за max_gap, читання submission.csv); `evaluator` не імпортується локально (немає faster-coco-eval), ран не запускався.
- **Алгоритм** (CPU, лише preds): IoU-трекер між сусідніми кадрами (той самий клас, жадібно 1-до-1), згладжування score по треку вікном ±`window` (mean/max, частка `alpha`), штраф `short_penalty` для треків коротших за `min_len`, прокидання впевнених детекцій (`prop_thr`, `prop_decay`) у сусідній кадр. Ідеї: Seq-NMS (arXiv:1602.08465), T-CNN (arXiv:1604.02532). Компенсації руху камери немає (потрібні зображення).
- **Ран:** `60_temporal`, `source: 10_frcnn_base` (плейсхолдер; потрібні повний `preds_val.json.gz` і `submission.csv` цього рану). Сітка 16 точок у конфігу, кожна точка це повна оцінка val (≈ хвилина CPU, орієнтовно 20-30 хв разом). Виходи: `temporal_grid.csv` (параметри, метрики, різниця з baseline), `plots/temporal_grid.png`, `metrics.json` найкращої точки (`fps` = null, у `extra`: `baseline_mAP`, `delta_mAP`, `best_params`), `preds_val.json.gz`, `submission.csv` (найкращі параметри на test-submission джерела). Відновлення через `_grid_rows.json`.
- **Кроки кадрів:** у val, test і train номери `frame` суцільні (різниця завжди 1; перевірено за анотаціями й `test_images.csv`). Відмінність лише в довжині: val 184-978 кадрів на кліп, test 35-148 (медіана 74), тож вікно ±2 на test охоплює більшу частку кліпу. Передбачення з `val_step=5` не годяться: за `max_gap=1` нічого не зв'яжеться. Потрібні повні val-preds (їх дає `finish_run`; у SMOKE кадри розкидані по кліпу (до 30 з двох кліпів), тож треків майже не буде, це очікувано).
- **Ризик/зауваження:** параметри вибираються на тому ж val, на якому звітуємо (мало параметрів, але слід назвати в звіті). Submission джерела уже обрізаний до 500 і округлений до 5 знаків. `common._find_in_runs` викликається напряму для `submission.csv`. Правок чужих файлів не потрібно.
