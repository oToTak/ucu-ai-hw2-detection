# Colab: інструкція для автора

Важкі рани йдуть на Google Colab (T4), бо Kaggle не має інтернету. Kaggle лишається лише для сабмітів, **завжди з одного основного акаунта**: правила змагання дискваліфікують за сабміти з кількох.

Ноутбуки беруть **запушений** код, тому перед кожною сесією: коміт і пуш.

## Один раз

1. Зібрати дані (шлях поза репозиторієм, файл ≈5.6 ГБ):
   `python workspace/colab/make_data_zip.py D:/hw2/data.zip`
2. На Google Drive в «Мій диск» створити папку `hw2`, покласти туди `data.zip`.
3. Для кожного іншого акаунта: Share папки `hw2` з правами редагування. Потім у цьому акаунті «Доступні мені» → папка → Add shortcut to Drive → «Мій диск». Ярлик має лежати в корені й зватися `hw2`.
4. Опційно W&B: у Colab Secrets додати `WANDB_API_KEY`. Без ключа ран не падає.

## Ноутбуки

| Ноутбук | Рани | ≈ год |
|---|---|---|
| [0_smoke_all](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/0_smoke_all.ipynb) | спільний smoke усіх нових ранів | ≈1 |
| [1_baseline_twostage](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/1_baseline_twostage.ipynb) | `05_baseline`, `1?_*` | 4.25 |
| [2_anchor](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/2_anchor.ipynb) | `2?_*` | 4.25 |
| [3_yolo_custom](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/3_yolo_custom.ipynb) | `3?_*`, `8?_*` | 6.5 |
| [4_detr](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/4_detr.ipynb) | `4?_*` | 4 |
| [0_smoke_final](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/0_smoke_final.ipynb) | smoke фінальних ранів на реальних вагах 41 | ≈0.5 |
| [5_final_temporal_viz](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/5_final_temporal_viz.ipynb) | `50`, `51`, `52`, `55`, `53`, `54` | ≈5 |
| [6_combo_temporal_viz](https://colab.research.google.com/github/oToTak/ucu-ai-hw2-detection/blob/main/workspace/colab/6_combo_temporal_viz.ipynb) | `56`, `6?_*`, `7?_*`, `?9_compare`, `59_summary`, лише після 3 і 5 | ≈1.5 |

Години орієнтовні: їх пораховано за кількома десятками ітерацій зі smoke, а не за повними епохами.

## Запуск

1. Відкрити посилання, Runtime → Run all, дозволити доступ до Drive.
2. Перша клітинка: `COMPUTE` і `SMOKE`. Спершу один раз `0_smoke_all`, бо рани читають smoke-ваги одне одного, а вони живуть лише в цій сесії. Він зупиняється на першому падінні, а `light_smoke.zip` із логами лежить у `hw2` в обох випадках.
3. Вкладку лишати відкритою. Рани дійдуть до кінця або сесія відвалиться: обидва випадки нормальні.

## Якщо відвалилось

Ще раз Run all у тому самому ноутбуці, можна й з іншого акаунта. Готові рани (є `results/<ран>/metrics.json`) пропускаються, перерваний продовжує з останнього чекпоінта в `hw2/wip/<ран>/`. Втрачається лише робота після останнього збереження чекпоінта (не частіше ніж раз на 15 хв).

Перерахувати ран примусово: `colab.run(COMPUTE, SMOKE, force=True)`.

## Після сесії

1. Наприкінці ноутбук кладе `light.zip` у папку `hw2` (без ваг). Зручно: скачати, розпакувати в `workspace/results/`, закомітити.
2. Копія на комп'ютері, якщо хочеться: Google Drive для комп'ютера синхронізує папку `hw2`.
3. Якщо ран упав: надіслати оркестратору останні ~15 рядків, ноутбук друкує їх сам. Упалий ран не зупиняє наступні, тому дивись список збоїв у виводі.
4. Public score записати в `results/<ран>/public_score.txt`.

## Місце на Drive

Чекпоінти перезаписуються на місці, але Drive може тримати старі версії й Кошик, і поки не перевірено, чи вони займають квоту. Файли, створені акаунтом у спільній папці, рахуються на нього. Якщо місце закінчується: очистити Кошик, папки `wip/` готових ранів прибираються самі.
