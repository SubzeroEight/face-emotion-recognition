import os
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import InterpolationMode


class DynamicAugmentDataset(Dataset):
    def __init__(self, main_dataset, class_weights, base_transform, strong_transform, mode):
        """
        :param class_weights: 各类别权重（用于控制增强强度）
        :param base_transform: 基础增强（所有类使用）
        :param strong_transform: 强增强（低样本类使用）
        """
        self.main_dataset = main_dataset
        self.class_weights = class_weights
        self.base_transform = base_transform
        self.strong_transform = strong_transform
        assert mode in ["train", "val"], "mode must in ['train', 'val']"
        self.mode = mode

    def __getitem__(self, index):
        img, label = self.main_dataset[index]

        if self.mode == 'train':
            # 根据类别权重决定增强强度
            prob = self.class_weights[label]  # 权重越高，使用强增强的概率越大
            if torch.rand(1, device=prob.device) < prob:
                img = self.strong_transform(img)
        img = self.base_transform(img)

        return img, label

    def __len__(self):
        return len(self.main_dataset)


def get_dataloaders(config, weights):
    # 公共预处理
    common_preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC)
    ])

    # 基础的增强策略
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

    train_path = os.path.join(config.data_path, "train")
    val_path = os.path.join(config.data_path, "val")

    # 加载原始数据集
    train_dataset = datasets.ImageFolder(root=train_path, transform=common_preprocess)
    val_dataset = datasets.ImageFolder(root=val_path, transform=common_preprocess)

    train_dataset = DynamicAugmentDataset(train_dataset, class_weights=weights, base_transform=base_transform,
                                          strong_transform=strong_transform, mode="train")
    val_dataset = DynamicAugmentDataset(val_dataset, class_weights=weights, base_transform=base_transform,
                                         strong_transform=strong_transform, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def calculate_class_weights(opt):
    class_counts = {}
    dir_path = os.path.join(opt.data_path, "train")

    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if os.path.isdir(class_path) and not class_name.startswith('.'):
            # 统计图片数量（假设都是图片文件）
            num_images = len([f for f in os.listdir(class_path) if not f.startswith('.')])
            class_counts[class_name] = num_images

    counts = [v for k, v in class_counts.items()]

    # 计算权重（样本量倒数）
    weights = 1. / torch.tensor(counts, dtype=torch.float)
    return weights / weights.sum()  # 归一化


# 每个epoch结束后更新权重
def update_weights(model, dataloader):
    emotion_classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 在模型设备上初始化（GPU）
    class_correct = torch.zeros(len(emotion_classes), device=device)
    class_total = torch.zeros(len(emotion_classes), device=device)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # labels 已经是 GPU 张量

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]  # GPU 张量
                class_correct[label] += c[i]  # 直接操作 GPU 上的张量（无需 .item()）
                class_total[label] += 1

    # 新权重 = 1 - 准确率（错误率越高，权重越大）
    new_weights = 1 - (class_correct / class_total)
    new_weights = new_weights / new_weights.sum()
    return new_weights  # 已经是 GPU 张量