# Стан оркестрації

Єдине джерело правди після `/clear`. Оркестратор оновлює після кожної зміни. Формат рядка: `ID · статус · агент · примітка`.

Статуси: `todo` · `running` · `done` · `blocked` · `skipped`.

## Задачі

| ID | Лейн | Хвиля | Статус | Агент | Примітка |
|---|---|---|---|---|---|
| T1a | F | W0 | todo | | |
| T1b | EDA | W1 | todo | | |
| T2 | UL | W1 | todo | | |
| T3 | TV | W1 | todo | | |
| AG-код | AG | W1 | todo | | compare.py, summary.py |
| G1 review | RV | | todo | | |
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
| K0 smoke | todo | | |
| K1 | todo | | |
| K2a/K2b | todo | | |
| K3 | todo | | |
| K4 | todo | | |

## Чекають на автора

- Перевірити пункти зі «Що має перевірити автор на старті» в `DECISIONS.md`.

## Зміни рішень відносно DECISIONS.md

(поки порожньо)
