from network import QNetwork
from typing import List
import torch
from torch.nn import Parameter
from torch.nn.functional import one_hot


# NOTE: For use in unity, different input and output shape is needed
# Inputs:
# - shape: (-1, 64, 64, 1) -> Vis observation
# - shape: (-1, 1, 1, 3) -> Nonvis observation
# - shape: (-1, 1, 1, 4) -> Action mask
#
# Outputs:
# - version_number: shape (1, 1, 1, 1) = [3]
# - memory_size: shape (1, 1, 1, 1) = [0]
# - discrete_actions: shape (1, 1, 1, 4) = [[2, 2, 2, 2]]
# - discrete_action_output_shape: shape (1, 1, 1, 4) -> network.action_options
# - deterministic_discrete_actions: shape (1, 1, 1, 4) -> network.action_options


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
        qnet_result, action_index = self.qnet.get_actions(obs, use_tensor=True)
        action_options = torch.tensor([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
        ])
        output = action_options[action_index]
        output = torch.mul(output, mask).to(dtype=torch.int64).view((-1, 4)) # .permute(0, 3, 1, 2)

        return self.version_number, self.memory_size, output, self.discrete_shape, output