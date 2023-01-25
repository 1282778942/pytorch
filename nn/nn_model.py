# -*- coding: UTF-8 -*-
# @Author：MengKang
# @Date：2023/01/23 14:57
import torch
from torch import nn


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


if __name__ == '__main__':
    net = NN()
    input = torch.ones((64, 3, 32, 32))  # 64张图片，通道为3，大小为32*32
    output = net(input)
    print(output.shape)
