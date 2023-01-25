import torch
# 查看Tensor的形状
x = torch.randn(2, 3)
print(x.size())
# 查看Tensor的维度
print(x.dim())
# 将x变形为3*2的矩阵
print(x.view(3, 2))
# 将x展平为1维向量
y = x.view(-1)
print(y)
# 添加一个维度
z = torch.unsqueeze(y, 0)
print("张量 y：", y, "张量 z：", z)
print(y.size(), z.size())
# 统计元素个数
print("张量 z 中元素个数为：", z.numel())
# 设置一个随机种子
torch.manual_seed(1)
x = torch.randn(2, 3)
print("张量 x为：", x)
# 根据索引获取第一行/列所有数据
print(x[0, :])  # 获取第一行所有数据
print(x[:, 0])  # 获取第一列所有数据
print(x[-1, :])  # 获取最后一行数据
print(x[:, -1])  # 获取最后一列数据
# 筛选出张量中大于零的元素
mask = x > 0
y = torch.masked_select(x, mask)
print(y)
# 获取张量中大、小于零元素的下标
y = torch.nonzero(mask)
print(y)
# 获取指定索引对应的值：Dim=1表示按照行号进行索引；Dim=0表示按照列号进行索引
index = torch.LongTensor([[0, 1, 1]])
y = torch.gather(x, dim=0, index=index)  # 第一列的第一个元素，第二列的第二个元素，第三列的第二个元素
print(y)
index = torch.LongTensor([[0, 1, 1],  # 第一行的第一个元素，第一行的第二个元素，第一行的第二个元素
                          [1, 1, 1]])  # 第一行的第二个元素，第二行的第二个元素，第二行的第二个元素
y = torch.gather(x, dim=1, index=index)
print("张量y：", y)
# scatter函数和gather函数的作用相反，scatter按照index将x放到y中
a = torch.randn(3, 3)
print("张量a：", a)
print("索引：", index)
y.scatter_(dim=1, index=index, src=a)
print(y)
