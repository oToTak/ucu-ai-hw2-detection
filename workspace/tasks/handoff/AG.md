# Handoff AG (AG-код, W1)

**Зроблено:** `src/compare.py` (runner `compare`), `src/summary.py` (runner `summary`). Перевірка: лише `py_compile` (ок). Нових залежностей немає (matplotlib уже в requirements; без нього пишуться лише csv/md).

**Формат `?9_compare.yaml` (визначено тут):**
```yaml
runner: compare
runs: [10_frcnn_base, 11_frcnn_rpn]   # імена ранів; відсутні пропускаються
title: "Two-stage"                    # необов'язково, заголовок графіків
```
Вхід: `results/<ран>/metrics.json`, `per_size.csv` (fallback: `AP_small/...` з metrics.json), `train_log.csv`. Шляхи беруться з `common.RESULTS` / `common.SMOKE_RESULTS` (з Colab через `HW2_RESULTS`), не з локальної константи. У SMOKE читає `SMOKE_RESULTS/<ран>`.
Виходи в `results/<compare>/`: `compare.csv`, `compare.md` (фрагмент таблиці), `plots/compare_sizes.png` (AP small/medium/large, бари), `plots/compare_curves.png` (val_mAP по епохах), `plots/compare_curves_minutes.png` (val_mAP за стовпцем `minutes` train_log.csv; без minutes — повідомлення в log, без падіння).

**Формат `59_summary.yaml` (пропоную, створює власник F або AG за потреби):**
```yaml
runner: summary
title: "Усі моделі"   # необов'язково
```
Сканує `common.RESULTS/*/metrics.json` (без тек з `_`; у SMOKE `common.SMOKE_RESULTS/*`). Виходи: `summary.csv` (відсортовано за mAP, з git/started/smoke із meta.json), `plots/summary_map.png`, `plots/summary_speed.png` (mAP проти fps).

**Density (`density: true` у `?9_compare.yaml`):** кадри повного val (`data.split_frames(..., "val", full=True)`) за кількістю GT на кадр ділимо на low/mid/high. Межі: `density_bins: [e1, e2]`, за замовчуванням терцилі val (однакові для всіх ранів). Для кожного рану з `common.find_preds` читаю `data.load_preds`, оцінюю групи спільним `evaluator.Evaluator` (max_det з протоколу). Вихід: `density.csv` (`run, bin, n_frames, mean_objects, mAP, AP50, AP_small`) і `plots/compare_density.png`. Ран без preds пропускається з повідомленням. Без `density` поведінка не змінюється.
**Не перевірено (Python недоступний):** density лише читанням коду. Перший smoke на Colab — перевірка.

**Поведінка без даних:** ран без `metrics.json` пропускається з рядком у `log.txt`; порожній список дає csv з заголовком і без падіння; помилка окремого графіка ловиться і друкується.

**Ключі, які читаю (контракт `src/CLAUDE.md`):** `mAP, AP50, AP75, AP_small, AP_medium, AP_large, AR, fps, train_minutes, epochs, imgsz, n_val_frames`; у `per_size.csv` — `size, AP`; у `train_log.csv` — `epoch, val_mAP`.

**Для автора (T8):** `compare.md` і `summary.csv` можна вставляти в звіт як фрагменти; числа бери з цих файлів, не з пам'яті.

**Ризики:**
- Ніхто ще не створив `configs/?9_compare.yaml` і `59_summary.yaml`; без них runner-и не викличеш через `compute`.
- Не перевірено на реальних даних: імена колонок у csv беру з `evaluator.py` і `common.py`; якщо F їх змінить, compare/summary тихо дадуть порожні стовпці.
- Порядок `summary` змінюється з часом (сортування за mAP), тож не порівнюй рядки між різними запусками summary.
- `results/_smoke` у summary не потрапляє в не-smoke ран (навмисно).

**Правки чужих файлів:** немає.
**requirements:** не змінював.
**Контекст:** легкий, GPU 0.
