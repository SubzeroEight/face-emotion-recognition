"""Project-wide paths and model constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_IMAGE_DATASET_DIR = DATASETS_DIR / "emotion-domestic"
FER2013_CSV_PATH = DATASETS_DIR / "fer2013" / "fer2013.csv"
SAMPLE_IMAGES_DIR = DATASETS_DIR / "samples" / "images"
SAMPLE_VIDEOS_DIR = DATASETS_DIR / "samples" / "videos"

DEFAULT_BASELINE_RUN_DIR = RUNS_DIR / "baseline"
DEFAULT_DYNAMIC_RUN_DIR = RUNS_DIR / "dynamic"
DEFAULT_CHECKPOINT_PATH = DEFAULT_DYNAMIC_RUN_DIR / "weights" / "best_model.pth"

REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_CONFUSION_DIR = REPORTS_DIR / "confusion_matrix"
REPORTS_DATA_DIR = REPORTS_DIR / "data"

YOLO_FACE_WEIGHTS = PROJECT_ROOT / "models" / "yolov5_face" / "weights" / "yolov5n-face.pt"

EMOTION_CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
DEFAULT_IMAGE_SIZE = 224
