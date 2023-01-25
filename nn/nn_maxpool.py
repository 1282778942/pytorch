import torch
import torch.nn.functional as F
from torch import nn

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
print(input.shape)
input = torch.reshape(input, (-1, 1, 5, 5))  # batch_size, channels, height, width
print(input.shape)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


nn = NN()
output = nn(input)
print(output)
