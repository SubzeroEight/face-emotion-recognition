"""绘制混淆矩阵"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torchvision import datasets

from config import DEFAULT_CHECKPOINT_PATH, DEFAULT_IMAGE_DATASET_DIR, REPORTS_CONFUSION_DIR
from data_loader.fer2013 import get_dataloaders
from inference import CONFIG
from inference.predictor import build_transform
from models.networks import coatnet_1


def plot_confusion_matrix_for_jaffe(
    all_labels,
    all_preds,
    datasets_name,
    class_names,
    output_dir=REPORTS_CONFUSION_DIR,
):
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 归一化处理
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建图形对象
    fig = plt.figure(figsize=(10, 8))  # 显式获取figure对象
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 先保存再显示
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'confusion_matrix_on_{datasets_name}.pdf', format='pdf', dpi=1200)
    plt.close(fig)  # 关闭图形释放内存
    plt.show()  # 如果需要显示可以保留

    print('完成')


def plot_confusion_matrix(
    model,
    data_loader,
    datasets_name,
    device,
    class_names,
    output_dir=REPORTS_CONFUSION_DIR,
):
    model.eval()
    all_preds = []
    all_labels = []

    remap_rule = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long).to(device)
    if datasets_name == "fer2013":
        remap_rule = torch.tensor([0, 1, 2, 3, 5, 6, 4], dtype=torch.long).to(device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)  # 根据数据集处理和dataloader而定，（bitch_size, 3，224，224）
            labels = labels.to(device)  # 根据数据集处理而定，（bitch_size, 1）

            remapped_labels = torch.take(remap_rule, labels)
            # torch.take 使用原始标签值作为索引，从 remap_rule 中取出对应值。如：当 labels=4 时，会取出 remap_rule[4] = 5

            outputs = model(inputs)  # 根据网络而定，（bitch_size, 7）
            _, preds = torch.max(outputs, 1)  # 返回：最大值，最大值索引

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(remapped_labels.cpu().numpy())

        # 生成混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        # 归一化处理
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建图形对象
        fig = plt.figure(figsize=(10, 8))  # 显式获取figure对象
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 先保存再显示
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'confusion_matrix_on_{datasets_name}.pdf', format='pdf', dpi=1200)
        plt.close(fig)  # 关闭图形释放内存
        plt.show()  # 如果需要显示可以保留

        print('完成')


def plot_emotion_domestic_confusion_matrix(path=None):
    path = Path(path) if path else DEFAULT_IMAGE_DATASET_DIR / 'val'
    test_transform = build_transform(img_size=CONFIG["img_size"])
    data_dataset = datasets.ImageFolder(root=path, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=64, shuffle=False)

    return data_loader


def plot_fer2013_confusion_matrix():
    _, _, data_loader_train = get_dataloaders(augment=False)
    # label = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return data_loader_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成表情分类混淆矩阵')
    parser.add_argument(
        '--dataset',
        choices=('emotion-domestic', 'fer2013'),
        default='emotion-domestic',
        help='要评估的数据集',
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument('--output-dir', type=Path, default=REPORTS_CONFUSION_DIR)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    emotion_model = coatnet_1(num_classes=len(CONFIG["classes"]))
    emotion_model = emotion_model.to(device)
    emotion_model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    emotion_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"train_acc: {checkpoint['train_acc']} val_acc: {checkpoint['val_acc']} epoch: {checkpoint['epoch']}")

    if args.dataset == 'emotion-domestic':
        datasets_name = 'emotion_domestic'
        data_loader = plot_emotion_domestic_confusion_matrix()
    else:
        datasets_name = 'fer2013'
        data_loader = plot_fer2013_confusion_matrix()

    plot_confusion_matrix(
        emotion_model,
        data_loader,
        datasets_name,
        device,
        CONFIG["classes"],
        args.output_dir,
    )

