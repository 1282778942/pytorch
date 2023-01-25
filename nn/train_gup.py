# -*- coding: UTF-8 -*-
# @Author：MengKang
# @Date：2023/01/23 11:36
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# 准备数据集
train_set = torchvision.datasets.CIFAR10(root='../nn/data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root='../nn/data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

# 查看训练集和数据集的图片个数
train_set_size = len(train_set)
test_set_size = len(test_set)
print("训练集中图片张数为：{}".format(train_set_size))
print("测试集中图片张数为：{}".format(test_set_size))


# 搭建神经网络
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

net = NN()
net = net.cuda(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda(device)

# 优化器
learning_rate = 1e-2  # 1 * (10)^(-2)=1/100=0.01
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epoch = 10  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("/logs_train")

start_time = time.time()

for i in range(epoch):
    print("*****第{}轮训练开始*****".format(i+1))

    # 训练步骤开始
    net.train()
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.cuda(device)
        labels = labels.cuda(device)
        output = net(imgs)
        loss = loss_fn(output, labels)

        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练时间为：{}".format(end_time-start_time))
            print("训练次数：{}， Loss值为：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = imgs.cuda(device)
            labels = labels.cuda(device)
            output = net(imgs)
            loss = loss_fn(output, labels)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == labels).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss为：{}".format(total_test_loss))
    print("整体测试集上的正确率为：".format(total_accuracy/test_set_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_set_size, total_test_step)
    total_test_step += 1

    torch.save(net.state_dict(), "net_{}.pth".format(i))
    print("模型已保存")

writer.close()
