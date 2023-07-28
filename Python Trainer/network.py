import torch
from typing import Tuple
from math import floor
from torch.nn import Parameter

# NOTE: For use in unity, different input and output shape is needed
# Inputs:
# - shape: (-1, 64, 64, 1) -> Image
# - shape: (-1, 1, 1, 3) -> Additional info
# - shape: (-1, 1, 1, 8) -> IDK
#
# Outputs:
# - version_number: shape (1, 1, 1, 1) = [3]
# - memory_size: shape (1, 1, 1, 1) = [0]
# - discrete_actions: shape (1, 1, 1, 4) = [[2, 2, 2, 2]]
# - discrete_action_output_shape: shape (1, 1, 1, 4)
# - deterministic_discrete_actions: shape (1, 1, 1, 4)

class QNetwork(torch.nn.Module):

    def __init__(self, visual_input_shape,nonvis_input_shape, encoding_size,output_size):
        super(QNetwork, self).__init__()
        height = visual_input_shape[1]
        width = visual_input_shape[2]
        initial_channels = visual_input_shape[0]
        # calculating required size of the dense layer based on the conv layers
        conv_1_hw = self.conv_output_shape((height, width), 5, 1)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 3, 1)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
        # layers
        self.conv1 = torch.nn.Conv2d(initial_channels, 16,5)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        self.nonvis_dense = torch.nn.Linear(nonvis_input_shape,8)
        self.dense1 = torch.nn.Linear(self.final_flat + 8,encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size,output_size)

    def forward(self, observation: Tuple):
        visual_obs, nonvis_obs = observation
        visual_obs, nonvis_obs = torch.Tensor(visual_obs), torch.Tensor(nonvis_obs)
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        nonvis_dense = torch.relu(self.nonvis_dense(nonvis_obs))
        hidden = conv_2.reshape([-1, self.final_flat])
        hidden = torch.concat([hidden,nonvis_dense],dim=1)
        hidden = self.dense1(hidden)
        hidden = torch.relu(hidden)
        output = self.dense2(hidden)
        return output

    def get_actions(self,observation):
        """
        Get the q values, if positive we do the action
        :param observation:
        :return q_values:
        """
        self.eval()
        with torch.no_grad():
            q_values = self.forward(observation)
        action_values = q_values.numpy()
        action_values[action_values > 0.0] = 1.0
        return q_values.numpy(), action_values
    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
          ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
          ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w