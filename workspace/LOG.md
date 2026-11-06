# Журнал сесій

Один рядок на сесію: дата · що зроблено.

- 2026-11-04 · Узгодили структуру й підхід (ноутбук як вхід), створили каркас і загальні таски.
- 2026-11-06 · Підготували оркестроване виконання: ORCHESTRATOR/PLAN/STATE/DECISIONS/_common/KAGGLE_RUNBOOK, блоки «Оркестрація» в усіх тасках, розширення контракту в src/CLAUDE.md.
- 2026-11-06 · [F/T1a] Спільні модулі common/data/evaluator/submission, pipeline з шаблонами й extends, ноутбук, протокол, requirements; selfcheck оцінювача пройшов.
- 2026-11-06 · [AG/AG-код] compare.py (runner compare: таблиця, бари AP за розмірами, криві val_mAP) і summary.py (runner summary: summary.csv, mAP і mAP-vs-fps); формат конфігів у handoff/AG.md; перевірка лише py_compile.
- 2026-11-06 · [UL/T2] ul_common (обгортка Ultralytics для T5/T6), baseline.py, 05_baseline.yaml (yolo11n, 8 епох); py_compile і load_config ок, smoke ще не бачив Kaggle.
- 2026-11-06 · [EDA/T1b] src/eda.py і 00_eda.yaml; повний ран 00_eda пораховано локально (results/00_eda: графіки, metrics.json, per_clip.csv), handoff/EDA.md.
- 2026-11-06 · [TV/T3] tv_common + frcnn.py, конфіги 10/11/12/19 (Faster R-CNN R50-FPN, розбір фейлів, RPN recall); перевірено лише py_compile і load_config.
- 2026-11-06 · [RV/G1] Review коду перед K0: 1 BLOCKER (UL, tloss це dict), 1 MAJOR (F, light.zip при падінні), 9 MINOR; список у tasks/handoff/RV.md, код не правив.
- 2026-11-06 · [F/T1a] Правки після review: M1 (упалий ран не зупиняє сесію, light.zip після кожного рану) і F-1 (git pull у ноутбуці).
- 2026-11-06 · [оркестратор] W0 і W1 готові, review G1 і правки внесено; код чекає на коміт автора й Kaggle smoke K0.
