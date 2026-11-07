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
