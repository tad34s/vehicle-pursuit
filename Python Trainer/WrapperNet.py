from network import QNetwork
from typing import List
import torch
from torch.nn import Parameter
from torch.nn.functional import one_hot

class WrapperNet(torch.nn.Module):
    def __init__(self, qnet: QNetwork, discrete_output_sizes: List[int]):
        super(WrapperNet, self).__init__()
        self.qnet = qnet

        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        output_shape = torch.Tensor([discrete_output_sizes])
        self.discrete_shape = Parameter(output_shape, requires_grad=False)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        qnet_result = self.qnet(obs)

        qnet_result = torch.mul(qnet_result, mask)

        output = qnet_result.view(-1, 2)
        output = torch.argmax(output, dim=1, keepdim=True)
        output = one_hot(output, num_classes=2).view(-1, self.discrete_shape.shape[-1])

        return self.version_number, self.memory_size, output, self.discrete_shape, output