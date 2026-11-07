# Handoff VZ (B2, візуалізація)

**Зроблено:** `src/visualization.py` (runner `visualization`), `configs/70_visualization.yaml` (була заглушка). Перевірено лише `py_compile` і `load_config`; не запускалось, перший запуск = SMOKE.

**run_id:** `70_visualization` (GPU, кілька хвилин на Colab T4; відновлення посеред рану немає, бо ран короткий). Не тренує, читає ваги через `common.find_weights` (у SMOKE з `results/_smoke/`).

**Що робить (3 частини, кожна пропускається з повідомленням у log.txt, якщо немає ваг; `metrics.json` пишеться завжди, ключ `parts` = статус кожної частини):**
- `plots/gradcam_frcnn.png`: Grad-CAM Faster R-CNN (`frcnn_run`) на P2-P5, ціль топ-1 детекція кадру. RoI береться з одного рівня FPN, тож інші рівні порожні (позначено в підписі).
- `plots/gradcam_rtdetr.png`: Grad-CAM RT-DETR (`rtdetr_run`) на трьох входах декодера (P3-P5), ціль топ-1 запит. Це не attention-карти декодера (MSDeformAttn хуками не чіпав), а градієнтний аналог.
- `plots/embeddings_frcnn.png`: RoI-ембедінги (вихід `box_head`, 1024) для GT-рамок val, t-SNE (sklearn; без нього PCA), колір це клас.

**Рішення (D6, найпростіший шлях):** Grad-CAM написаний вручну через `torch.autograd.grad`, бібліотека `pytorch-grad-cam` НЕ потрібна; UMAP не використано. `requirements.txt` не змінювався; `scikit-learn` очікується з образу Colab/Kaggle.

**Конфіг:** `frcnn_run: 10_frcnn_base`, `rtdetr_run: 40_rtdetr_base` (після ноутбука 3 замінити на найкращі з 1x і 4x), `n_frames`, `embed_frames`, `embed_per_class`, `tsne_perplexity`, `display_side`.

**W&B:** картинки логуються через `common.wandb_init`; без ключа (Colab) це no-op. Посилання в `metrics.json` ключ `wandb_url` (null без ключа). **W&B-посилання відсутнє**, поки не налаштований ключ.

**Ризики (не перевірено):** (1) RT-DETR: формат виходу eval-forward у Ultralytics 8.4.203 звірено з `head.py` (`(y, x)`, `y` = `(1, nq, 6)`: cx cy w h score cls), але градієнт через `topk`, `net.model[-1]` pre-hook і `predictor.model.model` лише за вихідниками. (2) Faster R-CNN: padding до кратного 32 не обрізається в картах (зсув <1%). (3) У SMOKE частини з помилкою піднімають виняток; поза SMOKE помилка частини пишеться в `metrics.json` і ран іде далі. (4) Висновки по картинках пише автор.

**Правки чужих файлів:** немає. **requirements:** без змін.
**Автору:** SMOKE = True з `COMPUTE=["70_visualization"]` після smoke-ранів 10_* і 40_*; потім справжній ран, коли ваги є.

## B2-bis: зворотне логування в W&B (`wandb_backfill`)
**Зроблено:** `src/wandb_backfill.py`, `configs/71_wandb_backfill.yaml`. Не перевірено (Python недоступний у сесії): код звірено читанням, перший запуск = SMOKE на Colab. Чужі файли не чіпав; `common._wandb_key` лише викликається.
**Що логує** (один W&B-ран на ран із `RESULTS`, де є `metrics.json`; за замовчуванням префікси 0,1,2,3,4,5,8 без runner-ів compare/summary/eda/visualization/wandb_backfill/temporal): `config.yaml` як config; криві `train_log.csv` (step = epoch); `metrics.json` у summary; таблиці `per_size`, `per_class`; `val_predictions` (8 кадрів, рамки передбачень зі score >= 0.3 і GT як `wandb.Image boxes`); `plots/*.png`.
**Дублікати:** `common.wandb_init` називає ран `run_id` (SMOKE: `_smoke_<run_id>`) зі ВИПАДКОВИМ id. Backfill має id `bf-<run_id>` (SMOKE `bfsmoke-<run_id>`) і `resume="allow"`; перед логуванням через `wandb.Api` шукає ран з тим самим name і іншим id (залогований наживо) і пропускає його. Повторний запуск оновлює той самий ран; точки кривих з уже наявними кроками W&B ігнорує з попередженням, таблиці й картинки додаються знову.
**Без ключа:** повідомлення в log.txt, `metrics.json` має `skipped_no_key` = число кандидатів, ран не падає. У SMOKE: <= 2 рани, 2 кадри, name з префіксом `_smoke_`.
**metrics.json:** `candidates, logged, skipped_live, skipped_no_key, errors, wandb_url` (`https://wandb.ai/<entity>/ucu-ai-hw2-detection`).
**Автору:** (1) Colab: Secrets -> додати `WANDB_API_KEY`, увімкнути доступ ноутбука. (2) `COMPUTE=["71_wandb_backfill"]`, спершу SMOKE. (3) Посилання на проєкт: `wandb_url` у `results/71_wandb_backfill/metrics.json` або `wandb.ai/<логін>/ucu-ai-hw2-detection`. (4) Публічність: wandb.ai -> проєкт -> Settings -> Project visibility = Public (у personal-акаунті доступно; у team залежить від налаштувань команди). Інакше додати викладачів у team або дати посилання на окремі ран-и. Перевірити посилання у вікні інкогніто.
**Ризик:** `wandb.Image(boxes=...)` з `domain: pixel` і числовими `class_labels` за документацією W&B, не запускалось; `api.default_entity` для перевірки дублікатів, при збої перевірка пропускається (можливі дублікати кривих).
