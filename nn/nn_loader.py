import torch
import torchvision

# 加载方式1 -> 对应保存方式1
model = torch.load("vgg16_method1.pth")
print(model)

# 加载方式2 -> 对应保存方式2(官网推荐)
model = torch.load("vgg16_method2.pth")
vgg_16 = torchvision.models.vgg16(pretrained=False)
vgg_16.load_state_dict(model)
print(vgg_16)
