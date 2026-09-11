"""Inference configuration and reusable transforms."""

from torchvision import transforms

from config import DEFAULT_CHECKPOINT_PATH, DEFAULT_IMAGE_SIZE, EMOTION_CLASSES, SAMPLE_IMAGES_DIR


CONFIG = {
    "checkpoint_path": DEFAULT_CHECKPOINT_PATH,
    "test_image_dir": SAMPLE_IMAGES_DIR,
    "classes": EMOTION_CLASSES,
    "img_size": DEFAULT_IMAGE_SIZE,
    "save_csv": True
}


def get_transform(img_size):
    """最基础的数据预处理流程"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
