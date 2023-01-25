import torch
a = torch.linspace(0, 10, 6)
a = a.view((2, 3))
print(a)
a_sum_0 = a.sum(dim=0)  # 按列划分，逐行相加
print(a_sum_0)
a_sum_1 = a.sum(dim=1)  # 按行划分，逐列相加
print(a_sum_1)
print(a.sum(dim=0, keepdims=True))
print(a.cumsum(dim=0))
# 比较操作，一般是进行逐元素的比较，有些时候按照指定方向比较
x = torch.linspace(0, 10, 6).view(2, 3)
print(torch.max(x))
print(torch.max(x, dim=0))  # 按列求最大值，并返回最大值的索引
# eq是逐元素进行比较，equal是张量比较
a = torch.Tensor([1, 2, 3])
b = torch.Tensor([1, 3, 3])
print(a.eq(b), a.equal(b))
# 矩阵操作
a = torch.tensor([2, 3])
b = torch.tensor([3, 4])
print(torch.dot(a, b))
x = torch.randint(10, (2, 3))  # 不超过10的两行三列张量
print(x)
# Numpy中不论数组维数，均可以用dot()函数进行相乘
# Pytorch中一维张量使用dot，二维张量使用mm，三维张量使用bmm
y = torch.randint(6, (3, 4))
print(torch.mm(x, y))
x = torch.randint(10, (2,2,3))
y = torch.randint(6, (2,3,4))
print(torch.bmm(x, y))
