import argparse
import math
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import DEFAULT_BASELINE_RUN_DIR, DEFAULT_IMAGE_DATASET_DIR
from data_loader.image_folder import build_loader
from models.networks import coatnet_1
from training.engine import evaluate, train_one_epoch

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main(opt):
    print(opt)  # 1.读取一些配置参数，并且输出
    assert os.path.exists(opt.data_path), "{} dose not exists.".format(opt.data_path)

    weights_dir = os.path.join(opt.output, "weights")
    logs_dir = os.path.join(opt.output, "logs")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=logs_dir)  # 日志保存在 output/logs

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else "cpu")

    data_loader_train, data_loader_val = build_loader(opt)

    nw = min([os.cpu_count(), opt.batch_size, opt.num_worker if opt.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    #   3.2 网络搭建：model
    classes = opt.num_classes
    model = coatnet_1(num_classes=classes)
    model = model.to(device)
    print(model)

    #   3.3 优化器，学习率，更新策略,损失函数
    pg = [p for p in model.parameters() if p.requires_grad]
    if opt.optimizer.lower() == 'sgd':  # 优化器
        optimizer = torch.optim.SGD(pg, lr=opt.lr, momentum=0.9, weight_decay=5e-5)
    elif opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(pg, lr=opt.lr, weight_decay=1e-3)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(pg, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 调度器
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_function = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    if opt.resume:
        resume_path = os.path.join(weights_dir, f'ckpt_epoch_{opt.resume}.pth')
        assert os.path.exists(resume_path), "{} dose not exists.".format(opt.resume)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"---------------------------------Resuming form {opt.resume}---------------------------------")

    elif opt.resume_best_epoch:
        resume_path = os.path.join(weights_dir, f'best_model.pth')
        assert os.path.exists(resume_path), "{} dose not exists.".format(opt.resume)
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"---------------------------------Resuming form {opt.resume}---------------------------------")

    best_acc = checkpoint["val_acc"] if opt.resume or opt.resume_best_epoch else -np.inf
    print("best_acc:", best_acc)
    print("----------------------------------Start training-------------------------------------------------")
    for epoch in tqdm(range(start_epoch, opt.epochs)):
        # train
        train_loss, train_acc = train_one_epoch(model, data_loader_train, device, optimizer, loss_function, epoch=epoch)
        scheduler.step()
        #  eval
        val_loss, val_acc = evaluate(model, data_loader_val, device, loss_function, epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "images"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        batch_images = next(iter(data_loader_train))[0].to(device)
        batch_images = batch_images * std + mean  # 反归一化
        batch_images = torch.clamp(batch_images, 0, 1)  # 确保像素值在 [0, 1]
        tb_writer.add_images(tags[5], batch_images, epoch)

        model_path = os.path.join(weights_dir, f"ckpt_epoch_{epoch}.pth")
        save_state = {'model_state_dict': model.state_dict(),
                      'train_acc': train_acc,
                      'val_acc': val_acc,
                      'epoch': epoch,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(),
                      }
        torch.save(save_state, model_path)

        #   3.6 模型保存：save
        is_best = val_acc > best_acc  # val_acc是epoch中的平均val acc
        if is_best:
            best_acc = val_acc
            print(f"best_acc:{best_acc}, epoch:{epoch}")
            best_model_path = os.path.join(weights_dir, "best_model.pth")
            save_state = {
                'model_state_dict': model.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(save_state, best_model_path)
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=str(DEFAULT_IMAGE_DATASET_DIR), help='The data path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--num-classes', default=7, type=int, help="Number of classes")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume-best-epoch', default=False, help='resume from checkpoint')
    parser.add_argument('--image-size', default=224, type=int, help="Input image size")
    parser.add_argument('--test-crop', action='store_true', help="Enable test-time center crop")
    parser.add_argument('--mixup-active', default=False, type=bool, help="")
    parser.add_argument('--output', default=str(DEFAULT_BASELINE_RUN_DIR), type=str, metavar='PATH', help='root of output folder')

    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--optimizer', type=str, default='adamw')  # sgd,adam,adamw

    args = parser.parse_args()

    main(opt=args)
