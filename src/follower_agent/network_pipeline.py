import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from follower_agent.buffer import ReplayBuffer, State
from follower_agent.hyperparameters import LEARNING_RATE_DEPTH_NET, LEARNING_RATE_QNET
from variables import ACTION_OPTIONS


class NetworkPipeline:
    def __init__(self, nonvis_input_shape, device, inject_correct_values=False):
        self.output_shape = (1, len(ACTION_OPTIONS))
        self.nonvis_input_shape = nonvis_input_shape
        self.device = device
        self.inject_correct_values = inject_correct_values

        with torch.device(self.device):
            self.depth_net = DepthNetwork()
            self.qnet = QNetwork(nonvis_input_shape)

    def get_actions(self, state: State, temperature: float) -> tuple[np.ndarray, np.ndarray, int]:
        t_ref_pred = None
        if not self.inject_correct_values:
            t_ref_pred = self.depth_net(torch.Tensor(state.img, device=self.device))
            t_ref = t_ref_pred
        else:
            t_ref = state.t_ref
        qnet_input = torch.Tensor(
            [[state.steer, state.speed, state.leader_speed, *t_ref]], device=self.device
        )
        q_values = self.qnet(qnet_input)
        q_values = q_values.flatten(1)

        if temperature == 0:
            action_index = torch.argmax(q_values, dim=1, keepdim=True)
        else:
            probs = torch.softmax(q_values / temperature, 1)
            action_index = random.choices(range(len(ACTION_OPTIONS)), weights=probs[0])
        q_values: np.ndarray = q_values.cpu().detach().numpy().flatten()

        return q_values, t_ref_pred, action_index[0]

    def fit(
        self, memory: ReplayBuffer, epochs_qnet: int = 1, epochs_depth_net: int = 10
    ) -> tuple[float, float]:
        dataset_qnet = memory.get_qnet_dataset(self.inject_correct_values)
        avg_loss_qnet = self.qnet.fit(dataset_qnet, epochs_qnet)
        dataset_depth_net = memory.get_depth_net_dataset()
        avg_loss_depth_net = self.depth_net.fit(dataset_qnet, epochs_depth_net)
        return avg_loss_qnet, avg_loss_depth_net


class DepthNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torchvision.models.alexnet(weights=True).features
        self.extra = torch.nn.MaxPool2d((2, 2))
        self.predict = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(258, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=LEARNING_RATE_DEPTH_NET,
            weight_decay=1e-7,
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)
        features = self.extra(features)
        features = features.view(-1, 256)
        features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis=1).to(
            torch.float32
        )
        preds = self.predict(features)
        return preds

    def fit(self, dataset, epochs=1) -> float:
        # train qnet
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        loss_sum = 0
        count = 0

        for _ in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                x, y = batch

                y_hat = self.qnet(x)
                loss = self.loss_fn(y_hat, y)
                print(f"loss {loss}")  # noqa: T201
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1

        return loss_sum / count


class QNetwork(torch.nn.Module):
    def __init__(self, nonvis_input_shape):
        self.nonvis_input_shape = nonvis_input_shape
        self.output_shape = (1, len(ACTION_OPTIONS))
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.nonvis_input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, self.output_shape[1]),
        )
        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=LEARNING_RATE_QNET,
            weight_decay=1e-7,
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def fit(self, dataset, epochs=1) -> float:
        # train qnet
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        loss_sum = 0
        count = 0

        for _ in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                x, y = batch

                y_hat = self.qnet(x)
                loss = self.loss_fn(y_hat, y)
                print(f"loss {loss}")  # noqa: T201
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1

        return loss_sum / count
