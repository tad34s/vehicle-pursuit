from mlagents_envs.environment import ActionTuple, BaseEnv
from typing import Dict
import random
import numpy as np
from typing import NamedTuple, List
from network import QNetwork
import torch
from torch.utils.data import Dataset, DataLoader


class Experience:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.predicted_values = []

    def add_instance(self, observations, action, predicted_values, reward):
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predicted_values.append(predicted_values)

    def calculate_targets(self):
        targets = []
        states = []
        for e, observation in enumerate(self.observations):
            if self.actions[e] is None:
                break
            reward_mask = self.actions[e] * self.rewards[e]
            target = self.predicted_values[e] + reward_mask
            observation = [arr.astype("float32") for arr in observation]
            target = target.astype("float32")
            states.append(observation)
            targets.append(target)

        return states, targets


class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add_exp(self, exp):
        if not self.is_full():
            self.buffer.append(exp)

    def is_full(self):
        return len(self.buffer) >= self.size

    def create_targets(self):
        state_dataset = []
        targets_dataset = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets()
            targets_dataset += targets
            state_dataset += states
        return state_dataset, targets_dataset

    def wipe(self):
        self.buffer = []


class StateTargetValuesDataset(Dataset):

    def __init__(self, states: list, targets: list):
        self.states = states
        self.targets = targets
        if len(states) != len(targets):
            raise ValueError

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], self.targets[index]


class Trainer:
    def __init__(self, model: QNetwork, buffer_size):
        """
        Class that manages creating a dataset and fitting the model
        :param model:
        :param buffer_size:
        """
        self.memory = ReplayBuffer(buffer_size)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.exploration_time = 10

    def train(self, env):
        """
        Create dataset, fit the model, delete dataset
        :param env:
        :return rewards earned:
        """
        env.reset()
        rewards_stat = self.create_dataset(env)
        self.fit(10)
        self.memory.wipe()
        return rewards_stat

    def create_dataset(self, env):
        behavior_name = list(env.behavior_specs)[0]
        all_rewards = 0
        # Read and store the Behavior Specs of the Environment
        print(self.memory.is_full())
        num_exp = 0

        while not self.memory.is_full():
            num_exp += 1
            exp = Experience()
            while True:

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                order = (0, 3, 1, 2)
                decision_steps.obs[0] = np.transpose(decision_steps.obs[0], order)
                terminal_steps.obs[0] = np.transpose(terminal_steps.obs[0], order)

                if len(terminal_steps) == 1 and len(decision_steps) == 0:
                    exp.add_instance(terminal_steps.obs, None, None, terminal_steps.reward)
                    break

                # Get the action
                if num_exp > self.exploration_time:
                    q_values, action_values = self.model.get_actions(decision_steps.obs)

                else:
                    q_values = np.zeros((1, 4))
                    action_values = np.random.randint(2, size=(1, 4))

                exp.add_instance(decision_steps.obs, action_values, q_values, decision_steps.reward)
                action_tuple = ActionTuple()
                action_tuple.add_discrete(action_values)
                env.set_actions(behavior_name, action_tuple)
                env.step()

            exp.rewards.pop(0)
            all_rewards += sum(exp.rewards)
            self.memory.add_exp(exp)

        return all_rewards

    def fit(self, epochs: int):

        states, targets = self.memory.create_targets()
        dataset = StateTargetValuesDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        for epoch in range(epochs):
            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                X, y = batch
                X = [X[0].view((-1, 1, 64, 64)), X[1].view((-1, 3))]
                y = y.view(-1, 4)
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                print("loss", loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
