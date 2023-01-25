import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

# L1 Loss()：abs(target_i - input_i)/total nums
loss = nn.L1Loss()
result_L1 = loss(input, target)  # [(1-1)+(2-2)+(5-3)]/3=0.6667

# MSE Loss(mean squared error)：[(target_i - input_i)^2]/total nums
loss = nn.MSELoss()
result_MSE = loss(input, target)

# Cross Entropy Loss：用于多分类
# 下面例子为三分类，分别是Person, Dog, Cat，对应的类别号为0, 1, 2
x = torch.tensor([0.1, 0.2, 0.3])  # x为输入图片经过网络以后的输出，该图片的三个类别概率值分别为0.1，0.2，0.3
y = torch.tensor([1])  # y为正确结果对应的类别，这里正确的类别是Dog，类别号为1
x = torch.reshape(x, (1, 3))  # Input：Shape(N,C)，where N=batch size，C=number of classes
loss = nn.CrossEntropyLoss()
result_loss = loss(x, y)

print(result_loss)

