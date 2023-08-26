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
    def __init__(self, qnet: QNetwork):
        super(WrapperNet, self).__init__()
        self.qnet = qnet

    def forward(self, vis_obs: torch.Tensor, nonvis_obs: torch.Tensor):
        qnet_result, action_index = self.qnet.get_actions((vis_obs, nonvis_obs),temperature = 0, use_tensor=True)
        action_options = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1]
        ])
        output = action_options[action_index]

        return qnet_result, output