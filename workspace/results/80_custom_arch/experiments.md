# 80_custom_arch

- **База:** 30_yolo_base
- **Що змінено:** додано рівень детекції P2 (stride 4, 64 канали); P3-P5, ваги, епохи, batch, протокол без змін
- **Чому:** EDA: 73.5% об'єктів train малі після ресайзу до 960, медіана sqrt(площі) 22.8 px, p5 10.2 px, k-means якорі від 8x11 px; 24.0% об'єктів мають sqrt(площі) < 16 px, тобто менше за 2 клітинки P3
- **Гіпотеза:** AP_small зросте більше, ніж впаде fps; AP_large не зміниться; вартість менша за перехід на 1280 (31_yolo_imgsz)
- **Ключі конфігу, відмінні від бази:** arch, budget_minutes

## Результат (повний val)

| Метрика | База | Цей ран | Різниця |
|---|---|---|---|
| mAP | 0.1715 | 0.1783 | 0.0068 |
| AP50 | 0.3403 | 0.3414 | 0.0012 |
| AP_small | 0.0841 | 0.1129 | 0.0288 |
| AP_medium | 0.1534 | 0.1711 | 0.0177 |
| AP_large | 0.3237 | 0.2842 | -0.0395 |
| fps | 61.8057 | 45.0292 | -16.7765 |
| train_minutes | 57.3818 | 85.8464 | 28.4646 |
| epochs | 10 | 9 | -1.0000 |

## Фейли й відхилення

- oom_batch_reduced: 1
- oom_assigner: 1
- епох 9 замість 10 (сторож бюджету)
- епох не порівну з базою: 9 проти 10

## Перевірка, що зміна застосована

```json
{
 "arch": {
  "name": "yolo11s-p2",
  "transferred": 499,
  "source_tensors": 499,
  "new_tensors": 64,
  "strides": [
   8.0,
   16.0,
   32.0,
   4.0
  ],
  "detect_inputs": [
   128,
   256,
   512,
   64
  ],
  "box_hidden": 64,
  "cls_hidden": 128,
  "tal_stride_val": 16.0
 },
 "smoke": false,
 "ultralytics_log": {
  "oom_batch_reduced": 1,
  "oom_assigner": 1,
  "nms_time_limit": 0,
  "resume_ignored_args": 0,
  "transferred_line": "Transferred 563/563 items from pretrained weights"
 },
 "best_pt": {
  "layers": 27,
  "strides": [
   8.0,
   16.0,
   32.0,
   4.0
  ],
  "detect_inputs": [
   128,
   256,
   512,
   64
  ],
  "cls_hidden": 128,
  "params": 9569360,
  "gflops_at_imgsz": 65.253312
 }
}
```

## Висновок (пише автор)
