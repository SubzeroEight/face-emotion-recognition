"""
Adapted from https://github.com/usef-kh/fer/tree/master/data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os

from config import DATASETS_DIR, FER2013_CSV_PATH


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])
        # print("原始img", img.shape)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
            # print("transform后的img", img.shape)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def load_data(path=str(FER2013_CSV_PATH)):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48, 3), dtype=np.uint8)
    image_label = np.array(data['emotion'].astype(int))

    for i, row in enumerate(data.index):
        # 解析像素数据
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = image.reshape(48, 48)

        # 修改2: 直接扩展为三通道
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)  # 形状 (48,48,3)

        image_array[i] = image  # 正确存储

    return image_array, image_label


def get_dataloaders(path=str(FER2013_CSV_PATH), bs=64, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    # mu, st = 0, 255
    mu = [0.485, 0.456, 0.406]
    st = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),  # 调整尺寸到224x224
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),  # 转为灰度图
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader


# ----------------- 以下这些在模型训练中用不到 -----------------------------------
def inverse_normalize(tensor):
    """ 反归一化函数 """
    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    st = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
    return tensor * st + mu


def show_images(dataloader, num_images=4):
    """
    查看数据集
    :param dataloader: 要查看的数据集加载器
    :param num_images: 选择打印几张，一共 bitch_size 张
    :return:
    """
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    # 获取一个批次
    images, labels = next(iter(dataloader))
    print("图像张量形状:", images.shape)  # 输出: torch.Size([64, 3, 224, 224])
    print("标签张量形状:", labels.shape)  # 输出: torch.Size([64])

    # 反归一化
    images = inverse_normalize(images)

    # 转换为 numpy 格式
    images = images.numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)

    # 显示图像
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        emotion_name = emotion_mapping[labels[i].item()]
        plt.title(f"Label: {emotion_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_images(image_array, image_labels, save_path, emotion_mapping):
    """
    将图像数组按标签分类保存到对应文件夹
    :param image_array: 图像数组 (N, 48, 48, 3)
    :param image_labels: 标签数组 (N,)
    :param save_path: 保存根目录
    :param emotion_mapping: 标签映射字典
    """
    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),  # 双三次插值
        transforms.Lambda(lambda x: x.convert('RGB'))  # 确保RGB格式
    ])

    # 创建类别文件夹
    for emotion in emotion_mapping.values():
        class_path = os.path.join(save_path, emotion)
        os.makedirs(class_path, exist_ok=True)

    # 初始化计数器
    counters = {emotion: 0 for emotion in emotion_mapping.values()}

    # 遍历保存所有图像
    for idx, (image, label) in enumerate(zip(image_array, image_labels)):
        # 获取类别名称
        emotion_name = emotion_mapping[label]

        # 更新计数器
        counters[emotion_name] += 1

        # 生成文件名
        filename = f"{emotion_name}_{counters[emotion_name]:04d}.png"
        file_path = os.path.join(save_path, emotion_name, filename)

        # 转换图像
        img = Image.fromarray(image[..., 0])  # 取出单通道（原数据是灰度）
        img = transform(img)  # 转换到224x224 RGB

        # 保存图像
        img.save(file_path)
        if idx % 500 == 0:  # 每处理500张打印进度
            print(f"已保存 {idx + 1}/{len(image_array)} 张")


def convert2image(data_path=str(FER2013_CSV_PATH), save_path=str(DATASETS_DIR / 'fer2013_images')):
    """
    将 fer2013 从 （1，48，48）转为（3，224，224） 保存在 save_path/train_images
    :param data_path:
    :param save_path:
    :return:
    """
    fer2013, emotion_mapping = load_data(data_path)
    image_array, image_labels = prepare_data(fer2013[fer2013['Usage'] == 'Training'])

    # 保存训练集图像
    print("正在保存训练集...")
    save_images(image_array, image_labels, os.path.join(save_path, "train_images"), emotion_mapping)


# 测试数据处理的是否正确
if __name__ == "__main__":
    # --------------- 查看数据集----------------
    # # 获取无增强的数据加载器
    # trainloader, valloader, _ = get_dataloaders(augment=False)
    #
    # print("=== 训练集样本验证 ===")
    # show_images(trainloader)
    #
    # print("=== 验证集样本验证 ===")
    # show_images(valloader)

    # ---------------------- csv灰度图转图片保存---------------------
    convert2image()
