import torch
import torchvision

vgg_16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1
torch.save(vgg_16, "vgg16_method1.pth")

# 保存方式2：将参数保存成字典(官方推荐)
torch.save(vgg_16.state_dict(), "vgg16_method2.pth")
