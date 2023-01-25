import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))  # batch_size, channels, height, width


class NN(nn.Module):
    def __init__(self) -> None:
        super(NN, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


nn = NN()
output = nn(input)
print(output)
