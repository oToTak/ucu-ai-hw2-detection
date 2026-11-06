"""05_baseline: найменший YOLO зі спільним протоколом. Мета: перевірити Kaggle-ланцюжок до submission.csv."""

from . import ul_common


def run(cfg, out_dir, data_root, smoke):
    ul_common.run(cfg, out_dir, data_root, smoke)


def load_predictor(cfg, weights):
    return ul_common.load_predictor(cfg, weights)
