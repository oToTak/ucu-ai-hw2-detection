# T6. Query-based (RT-DETR або Deformable DETR)

Той самий бюджет, що й у CNN-моделей, плюс тюнінг (кількість queries, ваги matching, loss або окремі LR). Крива mAP по епохах проти CNN. Якщо модель не збіглась, це теж результат: фіксуємо його.

**Готово, коли:** рани `4x` є в ноутбуці, а крива порівняння намальована.

## Оркестрація

- **Лейн UL · блок 2 (RT-DETR) · новий воркер з `handoff/UL.md` · модель sonnet · хвиля W3 · залежить від `ul_common`.**
- **Власні файли:** `src/models/rtdetr.py`, `configs/4?_*`.
- **Ідея:** `RTDETR` з Ultralytics через `ul_common`. Бюджет такий самий, як у CNN-ранів (за протоколом). Рекомендація: `num_queries` прив'язати до кількості об'єктів у щільних кадрах з EDA. Якщо не збігся, ран усе одно зберігається, а `metrics.json` має `converged: false` і поле з поясненням.
- **Рани:** `40_rtdetr_base`, `41_rtdetr_queries`, `42_rtdetr_lr` (окремий LR backbone/ваги matching), `49_compare` (крива val mAP по епохах проти CNN з `train_log.csv`).
- **Виходи:** `handoff/UL.md` (допиши блок RT-DETR).
- **GPU:** ≈3.75 год. **Smoke:** обов'язковий.
