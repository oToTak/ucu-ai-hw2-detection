# Handoff TV (T3, T4 ще ні)

**Зроблено:** `src/models/tv_common.py`, `src/models/frcnn.py`, `configs/10_frcnn_base.yaml`, `11_frcnn_rpn.yaml`, `12_frcnn_loss.yaml`, `19_compare.yaml`. Перевірено лише `py_compile` і `pipeline.load_config` для 4 конфігів. Сигнатури torchvision звірені з вихідним кодом на GitHub (faster_rcnn.py, roi_heads.py, rpn.py) і докою AMP; torch локально немає, перший запуск = Kaggle smoke.

**run_id для Kaggle:** `10_frcnn_base`, `11_frcnn_rpn`, `12_frcnn_loss` (GPU, ≈75 хв кожен, ≈3.75 год разом), `19_compare` (CPU, після трьох). Спершу SMOKE = True на `10_frcnn_base`. Потрібен Internet (COCO-ваги Faster R-CNN ≈160 MB з download.pytorch.org).

**tv_common (для T4):**
- `run_detector(cfg, out_dir, data_root, smoke, build_model, label_offset, tags=()) -> (metrics, predictor)`: датасет, SGD+AMP, косинус із warmup, val mAP щоепохи, `train_log.csv`, `weights/best.pt`, далі `evaluator.finish_run`.
- `build_model(cfg, pretrained) -> nn.Module`: пише модуль моделі (pretrained=True з вагами й новою головою; False без завантажень). Ваги завантажує `load_weights` із `strict=False` лише через `num_batches_tracked` (FrozenBN проти BatchNorm).
- `load_predictor(cfg, build_model, weights, label_offset)`, `TorchvisionPredictor` (зменшує кадр до imgsz PIL-ом, боксі в пікселях оригіналу, `.prepare(bgr)`, `.model`), `set_inference_limits` (працює і для RetinaNet: `score_thresh`, `detections_per_img` на моделі).
- `failure_analysis(cfg, out_dir, data_root, smoke) -> any_hit`: `failures.csv`, `failures_summary.json`, `plots/failure_{missed_small,duplicates,confusion}.png`. Модель-незалежний, читає `preds_val.json.gz`.
- Секції конфігу `train` (усі ключі обов'язкові, див. 10_frcnn_base) і `failure`. Для RetinaNet: `label_offset=0`, `num_classes=8`; у `train.amp` перевірити NaN (якщо loss не скінченний, батч пропускається і рахується).

**Рішення (автор може змінити):**
- Після review: інференс у fp32 (без autocast, RV TV-1); якорі `model.anchor_sizes: [16,32,64,128,256]` у базі (RV TV-2), тож 10/11/12 мають однакові якорі.
- v1 `fasterrcnn_resnet50_fpn`, не v2: швидше, більше епох у 75 хв. 4 епохи, bs 4, lr 0.005, `min_size = max_size = 960`.
- 11 = RPN пропозали 2000/1000 -> 3000/2000 (post-NMS train/test). 12 = `box_positive_fraction` 0.25 -> 0.5 (RoI sampling). GIoU-лосс не робив: `fastrcnn_loss` не бачить пропозалів.
- Зупинка за бюджетом: після епохи, якщо наступна не влізе в `budget_minutes - reserve_minutes (12)`. Тоді косинус не дойде до кінця.
- Фейли (на повному val, score ≥ 0.3, IoU ≥ 0.5): missed_small (GT ≤ 32² без детекції), duplicate (другий збіг із уже зайнятим GT того ж класу), confusion (збіг лише з GT іншого класу). Також `rpn_recall.csv`: частка GT, покрита пропозалами RPN, проти знайдених фінальними детекціями (100 кадрів val).
- Вага зберігається як `weights/best.pt` (conv/linear fp16, решта fp32). `19_compare` читає три ран-теки.

**Ризики:** швидкість невідома: після smoke глянути `с/it` у `log.txt` і підігнати `train.epochs`; OOM при bs 4 малоймовірний (T4/P100 16 GB), інакше bs 2 і lr 0.0025; перший прохід `failure_analysis` і `rpn_recall` не бігав (rpn_recall у try/except, не валить ран); `torch.amp.GradScaler("cuda")` потребує torch ≥ 2.3.

**Правки чужих файлів:** немає. **requirements:** нічого не додано.
