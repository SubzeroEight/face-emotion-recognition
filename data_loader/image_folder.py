import os
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from torchvision.transforms import InterpolationMode


def build_loader(config):
    dataset_train = build_dataset(is_train=True, config=config)
    dataset_val = build_dataset(is_train=False, config=config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker, pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker, pin_memory=True,
        drop_last=False
    )

    return data_loader_train, data_loader_val  # 未处理的dataset，loader_dataset


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    prefix = 'train' if is_train else 'val'
    root = os.path.join(config.data_path, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset  # 一个可迭代数据集，每个元素是包含(image_tensor, label)两个元素的元组


def build_transform(is_train, config):
    resize_im = config.image_size > 32
    if is_train:
        transform = create_transform(
            input_size=config.image_size, is_training=True, color_jitter=0.4, auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25, re_mode='pixel', re_count=1, interpolation='bicubic',
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        if config.test_crop:
            size = int((256 / 224) * config.image_size)
            t.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(config.image_size))
        else:
            t.append(
                transforms.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC)
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
