import sys
import torch


def train_one_epoch(model, data_loader, device, optimizer, loss_function, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0  # 逐渐累计 batch_size
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  # bitch_size

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  # 分类正确的不断累加

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()  # 损失的累加

        print("train epoch {} step {}， train loss: {:.5f} train acc: {:.5f}， lr: {:.7f}".format(
            epoch,
            step + 1,
            accu_loss.item() / (step + 1),   # train loss（平均损失）
            accu_num.item() / sample_num,  # train acc  （平均正确率）
            optimizer.param_groups[0]["lr"]))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num  # 本次epoch的平均损失和正确率


@torch.no_grad()
def evaluate(model, data_loader, device, loss_function, epoch):  # 每个epoch启用一次
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0  # 逐渐累计 batch_size
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  # bitch_size

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        print("[valid epoch {} step {}] val loss: {:.5f}, val acc: {:.5f}".format(epoch, step + 1,
                                                                                  accu_loss.item() / (step + 1),
                                                                                  accu_num.item() / sample_num))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


