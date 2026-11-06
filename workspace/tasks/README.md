# Таски

Виконуються **оркестровано**: одна керівна сесія запускає воркерів-субагентів паралельно. Кожен таск-файл має блок «Оркестрація» (лейн, модель, власні файли, GPU, smoke).

## Як запустити

Нова сесія в цьому репозиторії, перший промпт:

```
Ти оркестратор HW2. Прочитай workspace/tasks/ORCHESTRATOR.md і виконуй його. Стан: workspace/tasks/STATE.md.
```

Оркестратор сам читає решту й запускає воркерів. Після `/clear` або `/compact` той самий промпт відновлює роботу зі `STATE.md`.

## Файли керування

| Файл | Що |
|---|---|
| [ORCHESTRATOR.md](ORCHESTRATOR.md) | Інструкція оркестратору: запуск, контекст, ворота |
| [PLAN.md](PLAN.md) | Лейни, моделі, хвилі, власники файлів, бюджет GPU, графік |
| [STATE.md](STATE.md) | Живий стан (оркестратор оновлює) |
| [DECISIONS.md](DECISIONS.md) | Рішення за замовчуванням і що перевірити авторові |
| [_common.md](_common.md) | Правила для кожного воркера: межі, тести, Kaggle-готовність, handoff |
| [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md) | Інструкція авторові: як запускати Kaggle, чекліст review |
| `handoff/` | Записки воркерів один одному (`F.md`, `EDA.md`, `TV.md`, `UL.md`, `PP.md`, `VZ.md`, `CA.md`, `AG.md`) |

## Таски

Деталі рубрики: `workspace/teachers-things/overview.txt`.

| Таск | Що | Лейн · модель | Бали | Статус |
|---|---|---|---|---|
| T0 | Каркас | | — | готово |
| [T1](T1_data_and_eval.md) | Дані, EDA, спільний оцінювач | F opus · EDA sonnet | 1.00 | |
| [T2](T2_baseline.md) | Бейзлайн і перший сабміт | UL sonnet | — | |
| [T3](T3_two_stage.md) | Two-stage | TV sonnet | 1.00 | |
| [T4](T4_one_stage_anchor.md) | One-stage anchor-based | TV sonnet | 1.00 | |
| [T5](T5_anchor_free.md) | Anchor-free | UL sonnet | 1.00 | |
| [T6](T6_query_based.md) | Query-based | UL sonnet | 0.75 | |
| [T7](T7_final_pipeline.md) | Фінальний пайплайн | PP sonnet | 1.25 | |
| [B1](B1_temporal.md) | Бонус: temporal | PP sonnet | +1 | |
| [B2](B2_visualization.md) | Бонус: візуалізація | VZ sonnet | +1 | |
| [B3](B3_custom_architecture.md) | Бонус: своя архітектура | CA opus | до +3 | |
| [T8](T8_report.md) | Звіт (текст пише автор) | AG haiku | 2.00 | |

Статуси ведемо у `STATE.md`, не тут.

Порядок: T1a → (T1b ‖ T2 ‖ T3) → G1 review + Kaggle smoke → (T4 ‖ T5 ‖ T7-код) → (T6 ‖ B1 ‖ B3 ‖ B2) → T7 фінал → T8.
