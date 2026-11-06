# T4. One-stage anchor-based (RetinaNet або SSD)

База плюс тюнінг: anchors на основі розмірів з EDA, параметри focal loss або IoU-пороги. Порівняння початкової й тюнінгованої моделі з найкращою two-stage.

**Готово, коли:** рани `2x` є в ноутбуці, а порівняння є окремим графіком чи таблицею.

## Оркестрація

- **Лейн TV · блок [T3+T4] (той самий воркер, SendMessage) · модель sonnet · хвиля W2 · залежить від `handoff/EDA.md` (anchors) і `tv_common` з T3.**
- **Власні файли:** `src/models/retina.py`, `configs/2?_*`.
- **Ідея:** torchvision `retinanet_resnet50_fpn`; anchors з k-means по розмірах з `handoff/EDA.md` (стиль YOLOv2); focal α/γ, IoU-пороги.
- **Рани:** `20_retina_base`, `21_retina_anchors`, `22_retina_focal`, `29_compare` (початкова й тюнінгована vs найкраща two-stage; дані двох родин читає з `results/`).
- **Виходи:** `handoff/TV.md`.
- **GPU:** ≈3.75 год. **Smoke:** обов'язковий.
