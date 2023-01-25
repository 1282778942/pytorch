import torch
# 根据list数据生成Tensor
tensor = torch.Tensor([1, 2, 3, 4, 5, 6])
print(tensor)
# 根据指定形状生成Tensor
torch.Tensor(2, 3)  # 两行三列
# 根据给定的Tensor的形状生成Tensor
t = torch.Tensor([[1, 2, 3],
                  [4, 5, 6]])
t = torch.Tensor(t.size())
print(t)
# torch.Tensor()与torch.tensor()
print(torch.Tensor(2))  # 返回一个大小为2的张量，它里面的值是随机初始化的值
print(torch.tensor(2))  # 返回固定值
# 根据一定的规则生成Tensor
t = torch.eye(2, 2)  # 生成两行两列的单位阵
print(t)
t = torch.zeros(2, 3)  # 生成两行三列全是零的矩阵
print(t)
t = torch.linspace(1, 10, 4)  # 切片操作
print(t)
# 满足标准分布随机数生成Tensor
torch.randn(2, 3)
# 满足均匀分布随机数生成Tensor
torch.rand(2, 3)
# 返回所给数据形状相同，值全为0的张量
t = torch.rand(2, 3)
t = torch.zeros_like(t)
print(t)
