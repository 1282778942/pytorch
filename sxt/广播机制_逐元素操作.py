import torch
import numpy as np

A = np.arange(0, 40, 10).reshape(4, 1)
# print(A)
B = np.arange(0, 3)
# print(B)
# 把numpy数组转为Tensor

A1 = torch.from_numpy(A)
print(A1)
B1 = torch.from_numpy(B)
print(B1)

# 使用Tensor自动实现广播
C1 = A1+B1  # A1是四行一列，B1是一行三列，C1是四行三列
print(C1)

# 将t张量元素限制在指定区间，比如[0,1]区间
torch.manual_seed(10)
t = torch.randn(1, 3)
print(t)
t = torch.clamp(t, 0, 1)
print(t)
t = t.add_(2)
print(t)
