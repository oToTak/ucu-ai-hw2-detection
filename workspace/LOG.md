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
- 2026-11-06 · [F/Colab] Чотири Colab-ноутбуки, src/colab.py, HW2_* env, staging ранів, save_progress/load_progress із wip на Drive, make_data_zip.py; перевірено лише py_compile, json, load_config.
- 2026-11-06 · [оркестратор] Перехід на Colab: 4 ноутбуки, Drive-папка hw2, відновлення посеред рану, COLAB_RUNBOOK; чекає на коміт і smoke.
- 2026-11-06 · [UL/T5] yolo.py, конфіги 30_yolo_base (yolo11s), 31_yolo_imgsz (+_640), 32_yolo_loss (cls/cls_pw), 39_compare; py_compile і load_config ок, не запускалось.
- 2026-11-06 · [TV/T4] retina.py, конфіги 20/21/22/29 (RetinaNet: k-means якорі, focal alpha/gamma); спільний set_bn_eps для відновлених моделей; перевірено лише py_compile і load_config.
- 2026-11-06 · [PP/T7] етап 1: src/infer_tricks.py (NMS, Soft-NMS, WBF за статтею, SAHI-тайли, TTA, TrickPredictor), src/final.py (runner final, відновлення), configs/50_final.yaml плейсхолдер на 10_frcnn_base; перевірено py_compile, load_config і numpy-логіку, не запускалось.
- 2026-11-06 · [UL/T6] rtdetr.py (nq через yaml, backbone LR через підклас тренера), конфіги 40-42, 49_compare; py_compile і load_config ок, не запускалось.
- 2026-11-06 · [AG/AG-код] compare.py: додано крива val mAP за хвилинами (compare_curves_minutes.png) поряд зі старою за епохами; без minutes — повідомлення в log, без падіння.
- 2026-11-06 · [PP/B1] src/temporal.py (IoU-трекер, згладжування score по треку, штраф коротких треків, прокидання детекцій, сітка проти baseline) і configs/60_temporal (source-плейсхолдер 10_frcnn_base); py_compile, load_config, логіка на фейкових кліпах; не запускалось.
- 2026-11-06 · [VZ/B2] src/visualization.py (Grad-CAM Faster R-CNN і RT-DETR власним кодом, RoI-ембедінги + t-SNE) і configs/70_visualization.yaml; відсутні ваги = пропуск частини, metrics.json завжди; py_compile і load_config ок, не запускалось.
- 2026-11-06 · [CA/B3] custom.py, конфіги 80_custom_arch (P2), 81_custom_nwd (NWD), 89_compare, handoff/CA.md; перевірено лише py_compile і load_config, не запускалось.
- 2026-11-06 · [оркестратор] W2 і W3 написані, запусків не було; ноутбуки перебудовано під спільний smoke `0_smoke_all`, заглушки 50/60/70 винесено в окремий ноутбук; чекає на коміт і smoke.
