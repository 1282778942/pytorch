import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

# vgg16_true.classifier.add_module('add_module', nn.Linear(100, 10))
# print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
