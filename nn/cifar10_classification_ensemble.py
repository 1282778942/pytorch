import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# 准备数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

# 随机获取部分训练数据
dataiter = iter(train_loader)
images, labels = next(dataiter)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 显示图像
# imshow(torchvision.utils.make_grid(images))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(" ".join('%5s' % classes[labels[j]] for j in range(4)))


# 构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36*6*6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x


# 构建网络
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(1296, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.app = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = x.view(-1, 36*6*6)
        # x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # 使用全局平均池化
        x = self.app(x)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)
        return x


# 构建网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net0 = Net()
net1 = CNNNet()
net2 = LeNet()
mlps = [net0.to(device), net1.to(device), net2.to(device)]


# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=0.001)

for epoch in range(100):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 权重参数梯度清零
        optimizer.zero_grad()
        # 正向和反向传播
        for mlp in mlps:
            mlp.train()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
        optimizer.step()


    # 测试模型
    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        for i, mlp in enumerate(mlps):
            mlp.eval()
            out = mlp(img)

            _, prediction = torch.max(out, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()
            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(16)]
        vote_correct += (result == label.cpu().numpy()).sum()
    print("epoch: " + str(epoch) + "集成模型的正确率" + str(vote_correct/len(testloader)))

    for idx, correct in enumerate(mlps_correct):
        print("模型" + str(idx) + "的正确率：" + str(correct/len(testloader)))
