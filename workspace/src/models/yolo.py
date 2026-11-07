"""T5, anchor-free: сучасний YOLO (Ultralytics) через ul_common. Конфіги 30-32."""

from . import ul_common


def run(cfg, out_dir, data_root, smoke):
    ul_common.run(cfg, out_dir, data_root, smoke)


def load_predictor(cfg, weights):
    return ul_common.load_predictor(cfg, weights)
