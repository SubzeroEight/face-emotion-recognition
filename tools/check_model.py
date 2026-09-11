"""Check dependencies, local model paths, and the CoAtNet classification head."""

import argparse
import importlib.util
import sys

import torch

from config import DEFAULT_CHECKPOINT_PATH, DEFAULT_IMAGE_DATASET_DIR, YOLO_FACE_WEIGHTS
from inference import CONFIG
from models.networks import coatnet_1, count_parameters


OPTIONAL_MODULES = ("cv2", "PIL", "timm", "torchvision", "PyQt5", "pandas", "sklearn", "seaborn")


def report_path(label, path, required=True):
    path = path.resolve() if hasattr(path, "resolve") else path
    exists = path.exists() if hasattr(path, "exists") else False
    if required:
        status = "OK" if exists else "MISSING"
    else:
        status = "FOUND" if exists else "optional"
    size = f" ({path.stat().st_size / 1024 / 1024:.1f} MB)" if exists and path.is_file() else ""
    print(f"[{status:7}] {label}: {path}{size}")
    return exists or not required


def main():
    parser = argparse.ArgumentParser(description="检查项目环境和模型文件")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--forward", action="store_true", help="执行一次随机输入前向计算")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    missing_modules = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is None]
    print(f"Optional modules: {'all available' if not missing_modules else ', '.join(missing_modules)}")

    checks = [
        report_path("Sample image", CONFIG["test_image_dir"] / "000001.png"),
        report_path("Emotion dataset", DEFAULT_IMAGE_DATASET_DIR),
        report_path("YOLOv5-face weights", YOLO_FACE_WEIGHTS),
        report_path("CoAtNet checkpoint", args.checkpoint),
    ]

    if args.forward:
        model = coatnet_1(num_classes=len(CONFIG["classes"]))
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]))
        print(f"Forward output: {tuple(output.shape)}")
        print(f"Trainable parameters: {count_parameters(model):,}")

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
