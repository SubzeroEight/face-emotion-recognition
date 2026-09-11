import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import time

from config import SAMPLE_IMAGES_DIR, YOLO_FACE_WEIGHTS
from inference import CONFIG
from models.networks import coatnet_1
from models.yolov5_face.detect_face import detect, load_model


def load_models(device, config=CONFIG):
    # ==================== 1. 加载人脸检测模型 ====================
    # 加载YOLOv5-face模型
    face_weights = str(YOLO_FACE_WEIGHTS)
    face_model = load_model(face_weights, device=device)

    # ==================== 2. 加载表情分类模型 ====================
    # 初始化CoatNet表情分类模型
    emotion_model = coatnet_1(num_classes=len(config["classes"]))
    emotion_model = emotion_model.to(device)
    emotion_model.eval()

    # 加载表情分类模型权重
    ckpt_path = str(config["checkpoint_path"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    emotion_model.load_state_dict(checkpoint["model_state_dict"])

    return face_model, emotion_model


def build_transform(img_size):
    t = []
    size = int((256 / 224) * img_size)
    t.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def add_annotation(image, faces, pred_classes, confidences):
    """在图像上添加人脸框和表情标注"""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (box, cls, conf) in enumerate(zip(faces, pred_classes, confidences)):
        # 转换为整数坐标
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{cls} ({conf:.2%})"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = x1
        text_y = max(y1 - text_height - 5, 5)

        draw.rectangle([(text_x - 2, text_y - 2), (text_x + text_width + 2, text_y + text_height + 2)], fill="red")
        draw.text((text_x, text_y), text, font=font, fill="white")

    return image


def detect_faces_and_emotions(face_model, emotion_model, device, image_path, config=CONFIG):
    # ==================== 初始化计时器 ====================
    total_start = time.time()
    timings = {
        "total": 0,
        "face_detection": 0,
        "face_cropping": 0,
        "emotion_preprocessing": 0,
        "emotion_inference": 0,
        "result_processing": 0,
        "annotation": 0
    }

    # ====================  执行人脸检测 ====================
    # 调用detect函数获取人脸检测结果
    face_start = time.time()
    detection_results = detect(
        model=face_model,
        source=image_path,
        device=device
    )
    timings["face_detection"] = time.time() - face_start
    # print('detection_results', detection_results)

    # ====================  处理检测结果 ====================
    # 读取原始图像用于裁剪
    img_load_start = time.time()
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(original_img_rgb)
    timings["face_cropping"] = time.time() - img_load_start

    # 存储最终结果
    final_faces = []
    emotion_prep_total = 0
    emotion_infer_total = 0
    result_proc_total = 0

    # 遍历每个检测到的人脸 (处理单张图片)
    for frame_data in detection_results:
        for face in frame_data["faces"]:
            # print('face', face)
            # ==================== 裁剪人脸区域 ====================
            crop_start = time.time()
            x1, y1, x2, y2 = [int(coord) for coord in face["bbox"]]  # 左上角坐标  右下角坐标
            face_region = pil_img.crop((x1, y1, x2, y2))
            emotion_prep_start = time.time()
            timings["face_cropping"] += emotion_prep_start - crop_start

            # ==================== 表情分类预处理 ====================
            # 使用与训练时相同的transform
            transform = build_transform(config["img_size"])
            input_tensor = transform(face_region).unsqueeze(0).to(device)
            emotion_infer_start = time.time()
            emotion_prep_total += emotion_infer_start - emotion_prep_start

            # ==================== 执行表情分类 ====================
            with torch.no_grad():
                outputs = emotion_model(input_tensor)

            result_proc_start = time.time()
            emotion_infer_total += result_proc_start - emotion_infer_start
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # 解析结果
            pred_idx = np.argmax(probabilities)
            pred_class = config["classes"][pred_idx]
            confidence = float(probabilities[pred_idx])
            result_proc_end = time.time()
            result_proc_total += result_proc_end - result_proc_start

            # ==================== 记录结果 ====================
            final_faces.append({
                "position": [x1, y1, x2, y2],
                "emotion": pred_class,
                "confidence": confidence,
                # "landmarks": face["landmarks"]  # 可选：保留关键点信息
            })

    timings["emotion_preprocessing"] = emotion_prep_total
    timings["emotion_inference"] = emotion_infer_total
    timings["result_processing"] = result_proc_total

    # ==================== 生成带标注的图像 ====================
    # 在原始图像上绘制结果
    annot_start = time.time()
    annotated_img = original_img.copy()
    for face in final_faces:
        x1, y1, x2, y2 = face["position"]
        # 绘制人脸框
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # 添加表情标签
        label = f"{face['emotion']} {face['confidence']:.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    timings["annotation"] = time.time() - annot_start

    # ==================== 计算总耗时 ====================
    total_end = time.time()
    timings["total"] = total_end - total_start

    # ==================== 打印详细耗时 ====================
    print(f"\n{'=' * 40} 性能分析 {'=' * 40}")
    print(f"总耗时: {timings['total']:.4f} 秒")
    print(
        f"人脸检测耗时: {timings['face_detection']:.4f} 秒 ({timings['face_detection'] / timings['total'] * 100:.1f}%)")
    print(f"人脸裁剪耗时: {timings['face_cropping']:.4f} 秒 ({timings['face_cropping'] / timings['total'] * 100:.1f}%)")
    print(
        f"表情预处理耗时: {timings['emotion_preprocessing']:.4f} 秒 ({timings['emotion_preprocessing'] / timings['total'] * 100:.1f}%)")
    print(
        f"表情推理耗时: {timings['emotion_inference']:.4f} 秒 ({timings['emotion_inference'] / timings['total'] * 100:.1f}%)")
    print(
        f"结果处理耗时: {timings['result_processing']:.4f} 秒 ({timings['result_processing'] / timings['total'] * 100:.1f}%)")
    print(f"标注耗时: {timings['annotation']:.4f} 秒 ({timings['annotation'] / timings['total'] * 100:.1f}%)")
    print(f"检测到人脸数: {len(final_faces)}")
    print('=' * 90)

    # ==================== 返回最终结果 ====================
    return {
        "filename": image_path,
        "annotated_image": annotated_img,
        "faces": final_faces
    }


def print_results(result):
    """打印详细结果"""
    if not result:
        return

    # OpenCV显示图像
    cv2.imshow("Annotated Image", result['annotated_image'])
    cv2.waitKey(0)  # 等待按键后关闭窗口
    cv2.destroyAllWindows()

    # 打印文本结果
    print(f"\n文件名: {result['filename']}")
    print(f"检测到 {len(result['faces'])} 个人脸")
    for i, face in enumerate(result['faces'], 1):
        print(f"\n人脸 {i}:")
        print(f"位置: {face['position']}")
        print(f"表情: {face['emotion']}")
        print(f"置信度: {face['confidence']}")


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='单张图片表情检测')
    parser.add_argument(
        '--image-path',
        default=str(SAMPLE_IMAGES_DIR / '000001.png'),
        type=str,
        required=False,
        help='输入图片路径'
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model, emotion_model = load_models(device, config=CONFIG)

    # 执行检测
    result = detect_faces_and_emotions(face_model, emotion_model, device, args.image_path, CONFIG)

    # 输出结果
    if result:
        print_results(result)
    else:
        print("未能完成图片检测")
