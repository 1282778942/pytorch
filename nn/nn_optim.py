import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../nn/data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
net = NN()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data
        output = net(imgs)
        # print(output)
        output, target = output.to(device), target.to(device)
        loss_cross = loss(output, target)
        optim.zero_grad()  # 将梯度清零
        loss_cross.backward()  # 反向传播，计算梯度（选用合适的优化器），进而调整参数
        optim.step()
        running_loss += loss_cross
    print(running_loss)
