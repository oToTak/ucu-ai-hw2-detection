# Стан оркестрації

Єдине джерело правди після `/clear`. Оркестратор оновлює після кожної зміни. Формат рядка: `ID · статус · агент · примітка`.

Статуси: `todo` · `running` · `done` · `blocked` · `skipped`.

## Задачі

| ID | Лейн | Хвиля | Статус | Агент | Примітка |
|---|---|---|---|---|---|
| T1a | F | W0 | done | F-1 | selfcheck пройдено; torch-модулі перевірить smoke |
| T1b | EDA | W1 | done | EDA-1 | results/00_eda пораховано локально; Kaggle не потрібен |
| T2 | UL | W1 | done | UL-1 | run_id 05_baseline (yolo11n, 8 еп., ≈0.5 год); для T5 продовжити цього воркера |
| T3 | TV | W1 | done | TV-1 | run_id 10, 11, 12 (≈75 хв кожен), 19_compare CPU; train.epochs=4 підігнати після smoke; для T4 продовжити цього воркера |
| AG-код | AG | W1 | done | AG-1 | формат compare у handoff/AG.md; ?9_compare кладуть лейни родин, 59_summary кладе PP |
| G1 review | RV | | done | RV-1 | handoff/RV.md: 1 BLOCKER (UL), 1 MAJOR (F), MINOR; правки йдуть власникам |
| T4 | TV | W2 | todo | | |
| T5 | UL | W2 | todo | | |
| T7-код | PP | W2 | todo | | |
| T6 | UL | W3 | todo | | новий воркер (блок RT-DETR) |
| B1 | PP | W3 | todo | | |
| B3 | CA | W3 | todo | | після T5 |
| B2 | VZ | W3 | todo | | після T3 і T6 |
| T7-фінал | PP | G3 | todo | | після результатів 1x–4x |
| T8-підготовка | AG | G5 | todo | | текст звіту пише автор |

## Kaggle-сесії

| Сесія | Статус | Результати в results/ | public score |
|---|---|---|---|
| K0 smoke (Colab, ноутбук 1) | чекає на автора | | |
| K1 | todo | | |
| K2a/K2b | todo | | |
| K3 | todo | | |
| K4 | todo | | |

## Чекають на автора

- Commit+push; `make_data_zip.py`, папка `hw2` на Drive; smoke у ноутбуці 1 (`COLAB_RUNBOOK.md`).
- Перевірити на Kaggle: метрика на вкладці Evaluation, ліміт сабмітів.

## Зміни рішень відносно DECISIONS.md

- 06.11 · imgsz лишаємо 960 (73,5% малих на train після ресайзу перевіряє ран `31_yolo_imgsz`). Val за роздільністю не ділимо: зміщення val AP_small відносно test (19% проти оцінки 40–45% малих) автор називає у звіті як обмеження. Вибір фіналів за D9 без змін.

## Правки після review

06.11 · RV-1 завершено. UL (B1 + MINOR), F (M1 + MINOR), TV (TV-1 + MINOR) виправляють через продовження своїх воркерів. ORC-1 виправлено в KAGGLE_RUNBOOK. W2 (T4, T5, T7-код) стартує лише після K0 smoke, щоб не множити код на неперевіреній основі.
06.11 · Правки UL, F, TV готові (py_compile). UL-4, UL-5 відкладено до виміру на 05_baseline. Чекаємо на автора: commit+push, потім K0 smoke. Після K0 старт W2.
06.11 · Перехід на Colab (Kaggle без інтернету, лише сабміти вручну). F: 4 ноутбуки в workspace/colab/, src/colab.py, спільна Drive-папка hw2, save_progress/load_progress, make_data_zip.py (5.60 ГБ). UL і TV вбудовують відновлення. Далі: runbook, commit+push автора, smoke на Colab, потім W2.
06.11 · UL і TV вбудували відновлення (py_compile). COLAB_RUNBOOK написано, CLAUDE.md і PLAN оновлено. Незакомічена чужа правка hw2.ipynb лишається на рішення автора. Після K0 старт W2; T4 продовжує TV, T5 продовжує UL.
