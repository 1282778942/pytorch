import torch
import torchvision

if __name__ == '__main__':
    # 测试CUDA
    print("Support CUDA ?: ", torch.cuda.is_available())
    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)

    y = torch.randn(2, 3)
    y = y.cuda()
    print(y)

    z = x + y
    print(z)

    # 测试 CUDNN
    from torch.backends import cudnn
    print("Support cudnn ?: ", cudnn.is_available())

    # 查看版本号
    print("torch版本：", torch.__version__)
    print("torchvision版本：", torchvision.__version__)
    print("cuda版本：", torch.version.cuda)
    print("cudnn版本：", torch.backends.cudnn.version())
    print("GPU信息：", torch.cuda.get_device_properties(0))
