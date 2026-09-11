import os
from pathlib import Path

import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DEFAULT_IMAGE_DATASET_DIR, REPORTS_FIGURES_DIR
from inference import CONFIG
from inference.predictor import build_transform
from models.networks import coatnet_1


def plot_class_distribution(root_folder, save_name):
    """
    一个文件夹，文件夹下包含7个文件夹，每个文件夹中存放一类图片，文件夹的命名就是图片的种类。
    统计这7个文件夹下有多少图片，并绘制成柱状图
    :param root_folder: 文件夹路径
    :param picture_name: 保存的文件名称
    :return:
    """
    # 初始化存储数据的字典
    class_counts = {}

    # 遍历根文件夹下的所有子文件夹
    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)

        # 确保是文件夹且不是隐藏文件
        if os.path.isdir(class_path) and not class_name.startswith('.'):
            # 统计图片数量（假设都是图片文件）
            num_images = len([f for f in os.listdir(class_path) if not f.startswith('.')])
            class_counts[class_name] = num_images

    # 准备绘图数据
    class_names = [k for k, v in class_counts.items()]
    counts = [v for k, v in class_counts.items()]
    total = sum(class_counts.values())

    # 创建柱状图
    plt.figure(figsize=(10, 8))
    bars = plt.bar(class_names, counts)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100 if total != 0 else 0

        # 格式化显示：数值 + 百分比
        label = f"{height}\n({percentage:.1f}%)"

        plt.text(bar.get_x() + bar.get_width() / 2., height, label,  ha='center', va='bottom', fontsize=9)

        # 设置图表样式
    plt.title(f"{save_name}", fontsize=14, pad=20)
    plt.xlabel('Emotion Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    # plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = Path(save_name)
    if save_path.suffix.lower() != '.png':
        save_path = save_path.with_suffix('.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"种类统计柱状图已绘制完成，保存在：{save_path}")


def clean_image_dataset(folder_path, delete_ratio=None, keep_num=None):
    """
    图片数据集清理函数
    :param folder_path: 图片文件夹路径
    :param delete_ratio: 删除比例 (0-1)
    :param keep_num: 保留数量 (正整数)
    """
    # 参数校验
    if (delete_ratio is None and keep_num is None) or (delete_ratio and keep_num):
        raise ValueError("必须且只能指定 delete_ratio 或 keep_num 中的一个参数")

    # 获取所有图片文件（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    all_files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))
                 and os.path.splitext(f)[1].lower() in image_extensions
                 and not f.startswith('.')]

    total = len(all_files)
    print(f"发现 {total} 张图片")

    # 计算需要保留/删除的数量
    if delete_ratio is not None:
        if not 0 < delete_ratio < 1:
            raise ValueError("删除比例必须介于0和1之间")
        keep_num = int(total * (1 - delete_ratio))
    else:
        keep_num = min(keep_num, total)  # 不能超过总数

    # 随机选择要保留的文件
    keep_files = random.sample(all_files, keep_num)

    # 执行删除操作
    deleted = 0
    for filename in all_files:
        if filename not in keep_files:
            try:
                os.remove(os.path.join(folder_path, filename))
                deleted += 1
            except Exception as e:
                print(f"删除失败：{filename} - {str(e)}")

    print(f"\n操作完成，成功删除 {deleted}/{total} 张图片")
    print(f"当前剩余 {len(keep_files)} 张图片")


def select_save_class(model, data_path, device, class_name_save, save_path, delete_original=True):
    """
        从数据集 data_path 中挑选出 model 判断为 class_name_save 的数据，并保存到 save_path
    :param model:
    :param data_path:
    :param device:
    :param class_name_save:
    :param save_path:
    :return:
    """

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)

    # 获取数据集类别映射
    test_transform = build_transform(img_size=224)
    dataset = datasets.ImageFolder(root=data_path, transform=test_transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 获取类别索引映射（重要！）
    class_to_idx = dataset.class_to_idx
    if class_name_save not in class_to_idx:
        raise ValueError(f"Class '{class_name_save}' not found in dataset classes")
    target_class_idx = class_to_idx[class_name_save]

    # 定义反归一化转换（如果transform包含归一化）
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])

    counter = 0  # 文件名计数器

    # 获取原始文件路径列表
    original_files = [sample[0] for sample in dataset.samples]  # 所有原始文件的绝对路径
    # print(original_files)

    # 记录待删除文件的路径
    files_to_delete = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 获取当前批次对应的原始文件路径
            batch_start_idx = batch_idx * data_loader.batch_size
            batch_files = original_files[batch_start_idx: batch_start_idx + inputs.size(0)]

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 创建筛选掩码（同时满足两个条件）
            mask = (preds == labels) & (labels == target_class_idx)

            # 获取符合条件的索引（在当前batch中的相对位置）
            selected_indices = torch.nonzero(mask).squeeze(1).tolist()

            # 记录需要删除的原始文件路径
            for idx in selected_indices:
                files_to_delete.append(batch_files[idx])

            # 获取符合条件的样本
            selected_inputs = inputs[mask]
            selected_labels = labels[mask]

            # 保存符合条件的图像
            for i, (img_tensor, label) in enumerate(zip(selected_inputs, selected_labels)):
                # 反归一化并转换为PIL图像
                img = inverse_transform(img_tensor.cpu())

                # 生成文件名
                filename = f"{class_name_save}_{counter:04d}.png"
                filepath = os.path.join(save_path, filename)

                img.save(filepath)  # 保存图像
                counter += 1

            print(f"Processed batch {batch_idx + 1}/{len(data_loader)}, saved {counter} images so far")

    print(f"Total saved images: {counter} in {save_path}")

    # 新增：执行删除操作
    if delete_original and files_to_delete:
        confirm = input(f"即将删除 {len(files_to_delete)} 个原始文件，确认？(y/n): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败：{file_path} - {str(e)}")
            print(f"成功删除 {deleted_count}/{len(files_to_delete)} 个文件")
        else:
            print("删除操作已取消")
    elif delete_original:
        print("没有需要删除的文件")


if __name__ == '__main__':
    # 展示数据集中各类数据的数量
    data_folder = str(DEFAULT_IMAGE_DATASET_DIR / 'train')
    save_name = str(REPORTS_FIGURES_DIR / 'Original_Class_Distribution')
    plot_class_distribution(data_folder, save_name)

    # # 删除部分数据以保持数据的均衡性
    # deleted_folder_path = str(DEFAULT_IMAGE_DATASET_DIR / 'train' / 'happy')
    # clean_image_dataset(deleted_folder_path, keep_num=10)

    # # 预测fear并保存
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device:", device)
    #
    # emotion_model = coatnet_1(num_classes=len(CONFIG["classes"]))
    # emotion_model = emotion_model.to(device)
    # emotion_model.eval()
    #
    # ckpt_path = CONFIG["checkpoint_path"]
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # emotion_model.load_state_dict(checkpoint["model_state_dict"])
    # print(f"train_acc: {checkpoint['train_acc']} val_acc: {checkpoint['val_acc']} epoch: {checkpoint['epoch']}")
    #
    # class_name_save = "fear"
    # data_path = str(DEFAULT_IMAGE_DATASET_DIR / 'train')
    # save_path = str(DEFAULT_BASELINE_RUN_DIR / 'selected')
    # select_save_class(emotion_model, data_path, device, class_name_save, save_path)
