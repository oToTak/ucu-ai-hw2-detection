# T3. Two-stage (Faster R-CNN або Cascade)

База від pretrained ваг плюс щонайменше 2 експерименти (параметр і loss) за спільним протоколом. Метрики, швидкість, розбір 3 фейлів із правдоподібними причинами.

**Готово, коли:** рани `1x` є в ноутбуці й у `results/`, а фейли показані на графіках.

## Оркестрація

- **Лейн TV · блок [T3+T4] · модель sonnet · хвиля W1 · залежить від T1a.**
- **Власні файли:** `src/models/tv_common.py` (датасет, цикл тренування, eval-хук, `Predictor`, спільні для Faster R-CNN і RetinaNet), `src/models/frcnn.py`, `configs/1?_*`.
- **Ідея:** torchvision `fasterrcnn_resnet50_fpn` (або v2) з pretrained. Тренувальний цикл з AMP, per-epoch val mAP через `src.evaluator`, ignore-регіони за правилом з `src/CLAUDE.md`. `tv_common` пиши без прив'язки до Faster R-CNN: T4 додасть лише RetinaNet.
- **Рани:** `10_frcnn_base`, `11_frcnn_rpn` (кількість RPN-пропозалів, `rpn_*_top_n`), `12_frcnn_loss` (box loss або RoI sampling), `19_compare` (`runner: compare`: таблиця метрик і швидкості). Конфіги `11`/`12` через `extends: 10_frcnn_base`.
- **Failure cases:** runner рахує й малює 3 категорії фейлів на val (наприклад, пропущені дрібні, дублікати в натовпі, плутанина van/car) як `plots/failure_*.png` + `failures.csv` (розмір, occlusion, truncation, схожий клас). Причини формулює автор у звіті; ти даєш факти.
- **Виходи:** `handoff/TV.md`, рани 10–12, 19.
- **GPU:** ≈3.75 год (3 рани по ≈75 хв). **Smoke:** обов'язковий.
