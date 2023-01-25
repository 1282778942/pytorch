import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0, drop_last=False)

# img, target = trainset[0]
# print(img)
# print("类别为：{}".format(trainset.classes[target]), "对应类别号为：{}".format(target))
#


class my_first_nn(nn.Module):

    def __init__(self):
        super(my_first_nn).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


mfn = my_first_nn()
x = torch.tensor(1.0)
output = mfn(x)
print(output)
