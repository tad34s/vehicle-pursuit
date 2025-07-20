import torch

from follower_agent.network_pipeline import NetworkPipeline

# NOTE: For use in unity, different input and output shape is needed
# Inputs:
# - shape: (-1, 120, 160, 1) -> Vis observation
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
    def __init__(self, net: NetworkPipeline):
        super(WrapperNet, self).__init__()

        self.depth_net = net.depth_net
        self.qnet = net.qnet
        self.temperature = 0.0

        # Register action options as buffer (ensures correct device placement)
        self.register_buffer(
            "action_options",
            torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1]]),
            persistent=False,
        )

    def forward(
        self, visual_obs: torch.Tensor, nonvis_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process through depth_net for all batch elements
        t_ref_pred = self.depth_net(
            (
                visual_obs,
                nonvis_obs[:, 1],  # All batch elements, feature index 1
                nonvis_obs[:, 2],  # All batch elements, feature index 2
            )
        )

        nan_mask = torch.isnan(nonvis_obs[:, 3:6])

        condition = nan_mask.any(dim=1, keepdim=True)
        condition = condition.expand(-1, 3)  # Expand to [batch_size, 3]

        updated_nonvis = nonvis_obs.clone()
        updated_nonvis[:, 3:6] = torch.where(condition, t_ref_pred, nonvis_obs[:, 3:6])

        q_values = self.qnet(updated_nonvis).view((-1, self.qnet.output_shape[1]))

        action_index = torch.argmax(q_values, dim=1)  # [batch_size]

        output = self.action_options[action_index]  # Shape: [batch_size, 4]

        return q_values, output
