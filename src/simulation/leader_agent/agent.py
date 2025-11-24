import copy
from pathlib import Path

import numpy as np
import torch
import torch.onnx
from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from agent_interface import Agent
from leader_agent.buffer import Experience, ReplayBuffer, StateTargetValuesDataset
from leader_agent.hyperparameters import (
    BLUR_INTENSITY,
    LEARNING_RATE,
    NOISE_INTESITY,
    NOISE_OPACITY,
    REDUCE_TEMPERATURE,
    REWARD_MAX,
    START_TEMPERATURE,
)
from leader_agent.network import QNetwork
from leader_agent.wrapper_net import WrapperNet
from variables import ACTION_OPTIONS


class LeaderAgent(Agent):
    behavior_name = "CarBehavior?team=1"
    name = "Leader"

    def __init__(
        self,
        visual_input_shape: tuple,
        nonvis_input_shape: tuple,
        buffer_size: int,
        device: torch.device,
        num_agents: int = 1,
        writer: SummaryWriter | None = None,
    ) -> None:
        """
        Class that manages creating a dataset and fitting the model
        :param model:
        :param buffer_size:
        :param device:
        :param num_agents:
        """
        super().__init__()
        self.device = device
        self.writer = writer

        self.curr_episode = 0

        self.memory = ReplayBuffer(buffer_size)

        self.model = QNetwork(
            visual_input_shape,
            nonvis_input_shape,
            device,
        )

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-7,
        )
        self.num_agents = num_agents

        self.exps: dict[int, Experience] = {}
        self.temperature = START_TEMPERATURE
        self.episode_rewards = 0

    def train(self) -> float:
        """
        Create dataset, fit the model, delete dataset
        :param exploration_chance:
        :param env:
        :return rewards earned:
        """

        # clean not finished experiences

        for id, exp in self.exps.items():
            if len(exp.actions) == 0:
                continue
            exp.actions[-1] = None
            self.memory.add_exp(exp)
            self.episode_rewards += sum(exp.rewards)

        self.memory.flip_dataset()
        sample_exp = self.memory.buffer[int(self.memory.size() / 3)]
        sample_image = sample_exp.observations[int(len(sample_exp) / 3)][0]
        sample_q_values = sample_exp.predicted_values[int(len(sample_exp) / 3)]

        self.temperature = max(0.0, self.temperature - REDUCE_TEMPERATURE)

        if self.writer is not None:
            self.writer.add_image("Sample image leader", sample_image)

            # Add text
            steer = ""

            for s in sample_q_values:
                steer += f"{s:.2f} "

            self.writer.add_text("Sample Q leader values (steer)", steer, self.curr_episode)
            self.writer.add_scalar("Temperature Leader", self.temperature, self.curr_episode)

        self.fit(1)
        self.memory.wipe()

        final_episode_rewards = self.episode_rewards
        self.episode_rewards = 0
        return final_episode_rewards

    def submit_actions(self, steps: tuple[DecisionSteps, TerminalSteps]) -> ActionTuple | None:
        decision_steps, terminal_steps = steps

        dis_action_values = []
        cont_action_values = []

        if len(decision_steps) == 0:
            for agent_id in terminal_steps:
                exp = self.exps[agent_id]
                state, reward = self.get_state_and_reward(terminal_steps[agent_id])
                exp.add_instance(
                    state,
                    None,
                    np.zeros(self.model.output_shape[1]),
                    reward,
                )
                exp.rewards.pop(0)
                self.episode_rewards += sum(exp.rewards)
                self.memory.add_exp(exp)
                self.exps[agent_id] = Experience()
            return None
        else:
            for agent_id in decision_steps:
                state, reward = self.get_state_and_reward(decision_steps[agent_id])

                # Get the action
                q_values, action_index = self.model.get_actions(
                    state,
                    self.temperature,
                )

                dis_action_values.append(ACTION_OPTIONS[action_index][0])
                cont_action_values.append([])

                if agent_id not in self.exps.keys():
                    self.exps[agent_id] = Experience()

                self.exps[agent_id].add_instance(
                    state,
                    action_index,
                    q_values.copy(),
                    reward,
                )

            action_tuple = ActionTuple()
            final_dis_action_values = np.array(dis_action_values)
            final_cont_action_values = np.array(cont_action_values)
            action_tuple.add_discrete(final_dis_action_values)
            action_tuple.add_continuous(final_cont_action_values)

            return action_tuple

    def fit(self, epochs: int) -> None:
        temp_states, targets = self.memory.create_targets()

        states = [[torch.tensor(obs).to(self.device) for obs in state] for state in temp_states]

        targets = torch.tensor(targets).to(self.device)

        dataset = StateTargetValuesDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        loss_sum = 0
        count = 0

        bar = tqdm(total=epochs * len(dataloader), desc="Fitting leader")
        for _ in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                x, y = batch

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                count += 1
                bar.update()

        if self.writer is not None:
            self.writer.add_scalar("Loss/Episode Leader", loss_sum / count, self.curr_episode)
        self.curr_episode += 1

    def save_model(self, path: Path) -> None:
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model).cpu()),
            (
                # Vis observation
                torch.randn((1, *self.model.visual_input_shape)),
                # Non vis observation
                torch.randn((1, *self.model.nonvis_input_shape)),
            ),
            str(path),
            opset_version=9,
            input_names=["visual_obs", "nonvis_obs"],
            output_names=["prediction", "action"],
        )

    def get_state_and_reward(self, step) -> tuple[tuple[np.ndarray, np.ndarray], np.float32]:  # noqa: ANN001
        state_obs = (
            self.image_preprocessing(self.model.visual_input_shape, step.obs[0]),
            step.obs[1],
        )

        # step.reward is a number from 0 to 1 representing distance from center of the road

        reward = (  # sigmoid function for reward
            REWARD_MAX / (1 + np.exp((-10 * step.reward) + 4))
            if 0 <= step.reward <= 1
            else step.reward
        )

        return state_obs, reward

    @staticmethod
    def image_preprocessing(visual_input_shape, img: np.ndarray) -> np.ndarray:
        blurred = (
            gaussian_filter(img, sigma=BLUR_INTENSITY)
            + np.random.normal(0.2, NOISE_INTESITY, img.shape).astype("float32") * NOISE_OPACITY
        )

        # normalizing
        blurred[blurred > 1] = 1
        blurred[blurred < 0] = 0

        slice_starts = (
            blurred.shape[1] - visual_input_shape[1],
            blurred.shape[2] - visual_input_shape[2],
        )
        return blurred[0:, slice_starts[0] :, slice_starts[1] :]
