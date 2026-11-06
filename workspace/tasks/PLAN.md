# План оркестрації

> З 06.11 обчислення на Colab (4 ноутбуки, сесія ≈5 год), див. `COLAB_RUNBOOK.md`. Згадки Kaggle-сесій K0–K4 нижче читати як запуски цих ноутбуків. Кількість годин GPU без змін.

Сьогодні 06.11. Kaggle закривається 10.11 23:55, код і звіт 11.11 23:55. Вузьке місце: GPU на Kaggle, а не написання коду. Тому код пишемо хвилями, а Kaggle-запуски стартують, щойно готовий перший шматок.

## Лейни

Лейн — один воркер, що тримає свій контекст. Контекстний блок — набір тасків, між якими контекст не чистимо. Між блоками: новий воркер (аналог `/clear`) з handoff.

| Лейн | Таски | Модель | Блоки | Чому така модель |
|---|---|---|---|---|
| **F** foundation | T1a: дані, оцінювач, протокол, pipeline | opus | 1 | Помилка в оцінювачі знецінює всі числа й захист. Робиться раз. |
| **EDA** | T1b: аналіз даних | sonnet | 1 | Чистий pandas/matplotlib по готовому `data.py`. CPU. |
| **TV** torchvision | T3, T4 | sonnet | [T3+T4] | Faster R-CNN і RetinaNet ділять один тренувальний цикл. |
| **UL** ultralytics | T2, T5, T6 | sonnet | [T2+T5] YOLO · [T6] RT-DETR | YOLO і RT-DETR ділять обгортку. T6 з новим контекстом: DETR має свої нюанси. |
| **PP** postproc | T7 (код), B1 | sonnet | [T7+B1] | Усе працює над збереженими передбаченнями й `Predictor`. |
| **VZ** viz | B2 | sonnet | 1 | Хуки Grad-CAM/attention. Старт після готових T3 і T6. |
| **CA** custom arch | B3 | opus | 1 | Архітектурні рішення з абляціями: потрібне судження, а не обсяг коду. |
| **AG** aggregator | `compare.py`, `summary.py`; потім T8-підготовка | haiku | [код] · [T8] | Механічні таблиці й графіки з csv/json. |
| **RV** reviewer | review на G1 | opus | разово | Один дорогий погляд до першого Kaggle-запуску дешевший за 3 години GPU на баги. |

## Хвилі та залежності

```
W0   F:T1a ───────────────────────────────────────────────┐ (блокує все)
W1   EDA:T1b ‖ UL:T2 ‖ TV:T3 ‖ AG:код                     │
     └─ G1: RV review → автор commit+push → Kaggle K0 smoke → K1
W2   TV:T4 ‖ UL:T5 ‖ PP:T7-код                            (Kaggle K2…)
W3   UL:T6 ‖ PP:B1 ‖ CA:B3 (після T5) ‖ VZ:B2 (після T3 і T6)
G3   результати 1x–4x в results/ → PP:T7 фінал → G4 → AG:T8
```

Паралельні хвилі безпечні, бо лейни правлять різні файли (таблиця нижче).

## Хто який файл править

| Файли | Власник |
|---|---|
| `src/common.py`, `data.py`, `evaluator.py`, `submission.py`, `pipeline.py`, `configs/_protocol.yaml`, `hw2.ipynb`, `.gitignore`, `src/CLAUDE.md` | F |
| `src/eda.py`, `configs/00_*` | EDA |
| `src/compare.py`, `src/summary.py` | AG |
| `src/models/baseline.py`, `src/models/yolo.py`, `src/models/rtdetr.py`, `src/models/ul_common.py`, `configs/05_*`, `3?_*`, `4?_*` | UL |
| `src/models/frcnn.py`, `src/models/retina.py`, `src/models/tv_common.py`, `configs/1?_*`, `2?_*` | TV |
| `src/final.py`, `src/infer_tricks.py`, `src/temporal.py`, `configs/5?_*`, `6?_*` | PP |
| `src/visualization.py`, `configs/7?_*` | VZ |
| `src/models/custom.py`, `configs/8?_*` | CA |
| `requirements.txt` | усі дописують лише в кінець |

Кожна родина має свій `?9_compare.yaml` (`runner: compare`): порівняльний графік/таблиця живуть як звичайний ран. Його кладе лейн родини, код `compare` пише AG.

## Експерименти (орієнтири, не вимоги)

| Родина | Ран | Що змінюємо |
|---|---|---|
| 0x | `00_eda`, `05_baseline` | EDA (CPU); найпростіша модель, увесь ланцюжок до `submission.csv` |
| 1x | `10_frcnn_base` | Faster R-CNN FPN, pretrained |
| | `11_frcnn_rpn` | кількість RPN-пропозалів (напр. 1000 → 2000 під щільні кадри) |
| | `12_frcnn_loss` | box loss (smooth L1 → GIoU/L1) або RoI sampling |
| | `19_compare` | таблиця, швидкість, 3 фейли (картинки) |
| 2x | `20_retina_base` | RetinaNet |
| | `21_retina_anchors` | anchors k-means з EDA |
| | `22_retina_focal` | focal loss α/γ, IoU-пороги |
| | `29_compare` | базова/тюнінгована vs найкраща two-stage |
| 3x | `30_yolo_base` | сучасний YOLO (Ultralytics) |
| | `31_yolo_imgsz` | роздільність, діапазон пошуку фіксуємо |
| | `32_yolo_loss` | box loss/assigner |
| | `39_compare` | порівняння з anchor-based за розміром об'єктів |
| 4x | `40_rtdetr_base` | RT-DETR, **той самий бюджет** |
| | `41_rtdetr_queries` | кількість queries (щільність з EDA) |
| | `42_rtdetr_lr` | окремий LR backbone / ваги matching |
| | `49_compare` | крива mAP по епохах проти CNN |
| 5x | `50_final` … | найкраща модель + SAHI/TTA/Soft-NMS/WBF; ефект даних і моделі окремо |
| | `59_summary` | зведена таблиця всіх моделей |
| 6x | `60_temporal` | згладжування score по кліпу, прокидання детекцій |
| 7x | `70_visualization` | Grad-CAM (CNN), attention (RT-DETR), ембедінги |
| 8x | `80_custom_arch` | голова високої роздільності (P2) + 1-2 ablation |

Мінімум за рубрикою: у кожної з 1x–4x база + два експерименти. «Тюнінгована» модель = найкращий із двох, комбінація лише за наявності бюджету.

## Бюджет GPU (оцінка, Kaggle ~30 год/тиждень)

| Блок | Год |
|---|---|
| K0 smoke усього | 0.5 |
| 05 | 0.5 |
| 1x · 2x · 3x · 4x (по 3 рани ≈75 хв) | 4 × 3.75 = 15 |
| 5x (трюки інференсу дешеві, 1 довший ран) | 4 |
| 6x · 7x | 1 |
| 8x | 2.5 |
| **Разом** | **≈23.5** |
| Резерв на повтори | ≈6 |

Якщо квоти вже менше: порядок відсікання 8x → 7x → комбіновані ран-и → 6x. Мінімум 0x–5x не чіпати.

## Графік

| День | Ціль |
|---|---|
| 06.11 | W0 (F), старт W1 |
| 07.11 | кінець W1, G1, K0 smoke, K1: 05 + 1x |
| 08.11 | W2, K2: 2x + 3x (два ноутбуки паралельно) |
| 09.11 | W3, K3: 4x, K4: 8x/7x; починається T7 |
| 10.11 | до 18:00 фінальні обчислення; вибір двох сабмітів; Kaggle до 23:55 |
| 11.11 | T8 (таблиці/картинки), автор пише звіт; фінальний коміт до 20:00 |
