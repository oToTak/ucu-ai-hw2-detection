# Handoff RV: review перед K0 (06.11)

**Підсумок:** 1 BLOCKER (UL), 1 MAJOR (F), 9 MINOR (UL 5, TV 2, F 1, оркестратор 1). AG: 0. Після B1 і M1 можна в K0 smoke; решта MINOR не блокує, але TV-1 краще зробити до K1, а UL-3 до K2b і K3, бо потім доведеться перераховувати рани.

## BLOCKER

**B1 · UL · `src/models/ul_common.py:145`.** У 8.4.203 `trainer.tloss` це dict (`engine/trainer.py:559-562`, `utils/loss.py:489`), тому `trainer.tloss.sum()` дає AttributeError у колбеку наприкінці першої епохи. Падають усі рани Ultralytics, включно зі smoke, а `weights/best.pt` не створюється. Правка:
```python
tloss = getattr(trainer, "tloss", None)
loss = sum(float(v) for v in tloss.values()) if tloss else float("nan")
```

## MAJOR

**M1 · F · `src/pipeline.py:150-152`, `hw2.ipynb` остання клітинка.** Виняток зупиняє Run All до `pack_results()`: `light.zip` немає, наступні рани сесії не стартують. Падіння `12_frcnn_loss` на 4-й годині K1 лишає три готові рани (05, 10, 11) без архіву і без `compare`. Чи зберігає Kaggle Output для версії зі статусом Error, я не перевіряв, тому не варто на це покладатися. Правка: у `common.py` додати `FAILED = []`, а в `compute`:
```python
except Exception:
    _print_log_tail(log_path)
    common.FAILED.append(run_id)
    common.pack_results()
    if _smoke:   # у smoke падати одразу
        raise
    continue
except BaseException:
    _print_log_tail(log_path)
    raise
```
Після `print(f"{run_id}: готово ...")` теж кликати `common.pack_results()`, а `pack_results` нехай друкує `FAILED`. Тоді Run All доходить до кінця зі статусом успіху, а архів свіжий після кожного рану.

## MINOR

**UL-1 · `ul_common.py:79-83`.** Ultralytics уперше імпортується всередині `redirect_stdout` (модулі рану не імпортують його зверху; твердження в handoff F про «ноутбук» хибне), тому `StreamHandler` прив'язується до `log.txt` першого рану й після його закриття пише в закритий файл. Не падає, але кожен рядок логера в наступних UL-ранах сесії друкує traceback у stderr (K2b: 30, 31, 32). Правка після імпорту LOGGER (`setStream` не годиться, він робить flush закритого файлу; потрібен `import sys`):
```python
for h in LOGGER.handlers:
    if type(h) is logging.StreamHandler:
        h.stream = sys.stdout
```

**UL-2 · `ul_common.py:126`.** У налаштуваннях `raytune` увімкнено за замовчуванням, а `ray` є в образі (`kaggle_requirements.txt:126`): тренер імпортує його щоразу і додає колбек. Правка: `settings.update({"wandb": False, "raytune": False, "mlflow": False, "comet": False, "clearml": False, "dvc": False})`.

**UL-3 · `ul_common.py:141`, `:167-183`.** `budget_minutes` в Ultralytics не діє; для 05 це неважливо, для RT-DETR (T6) це ризик спалити GPU. Сторож `trainer.epoch >= trainer.epochs` вірний лише для повного прогону: при ранній зупинці `final_eval` (`trainer.py:1029`) викличе колбек ще раз із зайвою оцінкою і дублем у `train_log`. Правка: на початку колбека `if state.get("finished"): return` замість теперішнього сторожа; наприкінці порахувати `over = not smoke and minutes + (minutes - state.get("prev", 0)) > cfg["budget_minutes"] - cfg.get("reserve_minutes", 12)`, далі `state["prev"] = minutes`, `if over: trainer.stop = True`, `state["finished"] = bool(trainer.stop)`. `trainer.stop` читається після колбека (`trainer.py:695`).

**UL-4 · `ul_common.py:47-56`.** `predict` бере один клас на бокс (`multi_label=False`, `utils/nms.py:19`), нативний val і Faster R-CNN дають кілька класів на бокс. Наш YOLO-mAP буде трохи нижчим за `mAP_native_last`, і порівняння родин злегка нечесне. Спершу глянути різницю на `05_baseline`. Якщо вона суттєва: підклас `DetectionPredictor` з `postprocess`, що кличе `nms.non_max_suppression(..., multi_label=True)` і `construct_results`, передати через `model.predict(..., predictor=...)` лише для YOLO.

**UL-5 · `data.py:185` проти Ultralytics `data/augment.py:1714,1843`.** Кеш train зменшений PIL BILINEAR (зі згладжуванням), а інференс зменшує кадр через `cv2.INTER_LINEAR` (без нього). На val 3840 (×4, 34 % кадрів) вхід YOLO відрізняється від тренувального; torchvision використовує PIL в обох місцях. На test (×2 максимум) вплив менший. Вплив на mAP не заміряно. Правка за потреби: у `UltralyticsPredictor.predict` зменшувати кадр через `data._resized(im, imgsz)` і множити рамки назад на масштаб.

**TV-1 · `tv_common.py:81`.** Інференс під autocast: фічі fp16, тому якорі fp16 (`anchor_utils.py`), а `decode_single` робить `boxes.to(rel_codes.dtype)`, і центри рамок квантуються до 0.25–0.5 px при входу 960. YOLO рахує у fp32. Симуляція на val для ідеальних рамок: середній IoU 0.979 для об'єктів <16 px при 960 (7 % нижче 0.95), 0.989 для 16–32 px. Це б'є переважно по високих порогах, але упереджує порівняння. Референсний `evaluate` torchvision, наскільки пам'ятаю, теж без autocast. Правка: прибрати `with torch.autocast` у `predict` (fps трохи впаде). Зробити до K1.

**TV-2 · `frcnn.py:26-32`, `11_frcnn_rpn.yaml`.** Якорі лишаються дефолтні (від 32 px), а 87 % об'єктів при 960 мають 8–48 px (EDA). Більше пропозалів зі старими якорями може дати слабкий сигнал за 75 хв GPU. Це рішення, а не баг. Варіант: ключ `model.anchor_sizes` і `rpn_anchor_generator=AnchorGenerator(tuple((s,) for s in sizes), ((0.5, 1.0, 2.0),) * 5)` у `build_model`; по 3 якорі на локацію, тож COCO-ваги RPN завантажаться.

**F-1 · `hw2.ipynb`, клітинка 2.** Якщо тека вже є, код не оновлюється: в інтерактивній сесії після пушу запуститься старий код. Правка: `else: subprocess.run(["git", "-C", "ucu-ai-hw2-detection", "pull", "--ff-only"], check=True)`. Save & Run All стартує з чистого каталогу, тож його це не стосується.

**ORC-1 · `KAGGLE_RUNBOOK.md:7`.** «T4 або P100»: база образу, за Colab freeze, має `torch 2.14.1+cu130`; CUDA 13, наскільки я знаю, не підтримує Pascal (це з пам'яті, на Kaggle не перевірено). Написати: лише T4.

## Чекліст G1
Пункти 1–9 пройдено за кодом, крім п. 2 для UL: до B1 ран не допише виходи. `SMOKE` ≤3 хв лише оцінка.

## Перевірено, правок не треба
- **Requirements:** образ Kaggle це Python 3.13 (`Dockerfile.tmpl`), за Colab freeze numpy 2.4.6, pandas 2.2.3, matplotlib 3.10.0, wandb 0.28.1. Dry-run pip для py3.13 розв'язується із закріпленими версіями freeze (numpy 2.4.6, opencv 5.0.0.93, polars, pydantic, protobuf, pillow), wheels cp313 є. Пін `wandb==0.26.1` лише понижує версію, це не ламає. Імпорт пінених pandas 2.3.3 і matplotlib 3.10.8 з numpy 2.4.6 перевірено на py3.14 (не 3.13), на ньому ж пройшов смоук EDA.
- **Прогін без torch** (venv, вивід у scratchpad; torch лише заглушка для `failure_analysis`): EDA smoke, `build_ultralytics_dataset` (train-кеш 960×540, лейбли поруч, val-лейбли на місці), `finish_run` із фейковим Predictor (усі виходи, `submission.csv` на 3811 рядків), `failure_analysis`, `pack_results`, `compare`, `summary`. Розміри кадрів у JSON/CSV збігаються з файлами (111 кліпів), рамки ≥4 px, класи 0–7.
- **Класи й координати:** `label_offset` 1 у датасеті і −1 у predictor; бокси в пікселях оригіналу в обох обгортках.
- **API:** `get_save_dir` з абсолютним project, `load_checkpoint` з last.pt (EMA), `strip_optimizer`, `device=None` дає один GPU на 2×T4, змішані розміри в батчі (`pre_transform`), kwargs FasterRCNN у torchvision 0.29.1. torchvision NMS завжди активний для YOLO (`validator.py:124`, `autobackend.py:379`). `extends`/deep merge перевірено на 11 і 12.
- **Пам'ять і час:** кадри зменшуються до GPU, 8 кадрів 3840×2160 це ≈200 MB RAM. Оцінювач на 2846×500 детекціях: 13 с; `preds_val.json.gz` 22 MB за 13 с; `submission.csv` 70 MB за 4 с (випадкові рамки, лише на CPU). Ліміт розміру сабміту на Kaggle не перевірено. `reserve_minutes: 12` вистачить на 6657 кадрів при ≈15 fps, але fps лише моя оцінка.

## Не перевірено (покаже smoke)
Усе, що потребує torch/GPU: швидкість `с/it`, AMP, `torch 2.14`; `opencv-python 5.0` з Ultralytics 8.4.203 (пакет задовольняє вимоги, запускати не міг); фактична збірка Kaggle може відрізнятись від Colab freeze. На smoke глянути фактичні версії в рядку `Ultralytics ... torch-...` у `log.txt`; у `ultralytics.log` на K1 грепнути `NMS time limit` (перевищення лишає решту кадрів батчу порожніми, поріг 2.4 с на батч 8).
