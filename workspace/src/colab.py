"""Логіка тонких ноутбуків workspace/colab/*.ipynb.

Цей модуль не імпортує common і pipeline зверху: вони читають env під час імпорту,
а env виставляє setup().
"""
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HUB = Path("/content/drive/MyDrive/hw2")
DRIVE_MOUNT = "/content/drive"
LOCAL_DATA = Path("/content/data")
LOCAL_WORK = Path("/content/hw2_work")
LOCAL_SMOKE = Path("/content/results/_smoke")
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
DATA_LAYOUT = "у корені zip: test_images.csv, train/, val/, test/"
# Те, що Colab уже має: якщо пакет є, його версія лишається, навіть транзитивно.
KEEP_INSTALLED = ["numpy", "pandas", "matplotlib", "torch", "torchvision", "pyyaml", "wandb"]


def _check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU немає. Runtime → Change runtime type → T4 GPU, потім Run all ще раз.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def _mount_drive():
    from google.colab import drive

    drive.mount(DRIVE_MOUNT)
    if not HUB.is_dir():
        raise FileNotFoundError(
            f"Немає папки {HUB}. Створи на Google Drive у «Мій диск» папку hw2 і поклади туди data.zip ({DATA_LAYOUT})."
        )
    if not (HUB / "data.zip").is_file():
        raise FileNotFoundError(f"Немає {HUB / 'data.zip'}. Поклади його в папку hw2 на Drive ({DATA_LAYOUT}).")


def _prepare_data():
    """Кадри читаємо лише з локального диска; розпакований каталог з'являється цілим."""
    if (LOCAL_DATA / "test_images.csv").exists():
        print(f"Дані вже є: {LOCAL_DATA}")
        return
    local_zip = LOCAL_DATA.with_name("data.zip")
    partial = LOCAL_DATA.with_name("data_partial")
    shutil.rmtree(partial, ignore_errors=True)
    print("Копіюю data.zip з Drive і розпаковую...")
    shutil.copyfile(HUB / "data.zip", local_zip)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extractall(partial)
    local_zip.unlink()
    missing = [name for name in ("test_images.csv", "train", "val", "test") if not (partial / name).exists()]
    if missing:
        shutil.rmtree(partial, ignore_errors=True)
        raise FileNotFoundError(f"data.zip має неправильну структуру, немає {missing}. Потрібно: {DATA_LAYOUT}.")
    os.replace(partial, LOCAL_DATA)
    print(f"Дані готові: {LOCAL_DATA}")


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_requirements():
    """Ставить решту requirements.txt; constraints не дають pip змінити наявні версії KEEP_INSTALLED."""
    kept = {_canonical(name): _installed_version(name) for name in KEEP_INSTALLED}
    kept = {name: version for name, version in kept.items() if version}
    lines = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#")[0].strip()
        if requirement and _canonical(re.split(r"[=<>!~\[ ;]", requirement)[0]) not in kept:
            lines.append(requirement)
    if not lines:
        return
    with tempfile.TemporaryDirectory() as tmp:
        requirements, constraints = Path(tmp) / "requirements.txt", Path(tmp) / "constraints.txt"
        requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
        constraints.write_text("".join(f"{n}=={v}\n" for n, v in kept.items()), encoding="utf-8")
        command = [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements), "-c", str(constraints)]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip не зміг поставити залежності:\n{result.stderr[-2000:]}")
    print(f"Поставлено: {lines}")


def _set_env():
    if "src.common" in sys.modules:
        print("УВАГА: src.common уже імпортований, після pull зроби Runtime → Restart session")
    (HUB / "results").mkdir(exist_ok=True)
    os.environ.update(
        HW2_RESULTS=str(HUB / "results"),
        HW2_WIP=str(HUB / "wip"),
        HW2_PACK_DIR=str(HUB),
        HW2_DATA=str(LOCAL_DATA),
        HW2_WORK=str(LOCAL_WORK),
        HW2_SMOKE_RESULTS=str(LOCAL_SMOKE),
    )


def setup(gpu=True):
    """GPU, Drive, дані на локальний диск, залежності, env. Повторний виклик безпечний.

    gpu=False: для CPU-ранів (W&B backfill, compare, summary), коли ліміт GPU вичерпано."""
    if gpu:
        _check_gpu()
    _mount_drive()
    _prepare_data()
    _install_requirements()
    _set_env()


def run(compute_list, smoke=False, force=False):
    """Рахує рани й показує результати. force: True або список шаблонів, рахувати заново."""
    from . import pipeline

    pipeline.configure(compute_list, smoke, force)
    pipeline.compute(*compute_list)
    pipeline.show(*compute_list)


def finish():
    """light.zip у HUB; після справжніх ранів дописує все на Drive і відмонтовує його."""
    from google.colab import drive

    from . import common

    common.pack_results()
    if not common.SMOKE:
        drive.flush_and_unmount()
