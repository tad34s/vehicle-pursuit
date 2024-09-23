import numpy as np
from network import mirrored_actions
from torch.utils.data import Dataset
from variables import DISCOUNT, REWARD_SAME_ACTION


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

    def flip(self):
        new_observations = [(np.flip(vis, 2), nonvis) for vis, nonvis in self.observations]
        new_actions = [mirrored_actions[x] if x is not None else None for x in self.actions]
        new_predicted_values = [np.flip(x, 0) for x in self.predicted_values]

        new_exp = Experience()
        new_exp.observations = new_observations
        new_exp.actions = new_actions
        new_exp.rewards = self.rewards.copy()
        new_exp.predicted_values = new_predicted_values
        return new_exp

    def calculate_targets(self):
        targets = []
        states = []
        for e, observation in enumerate(self.observations):
            if self.actions[e] is None:
                break

            action_index = self.actions[e]
            reward = self.rewards[e]

            if e != 0:
                if self.actions[e] == self.actions[e - 1]:
                    reward += REWARD_SAME_ACTION

            # we take the matrix of predicted values and for the actions we had taken adjust the value by the reward
            # and the value of the next state
            target_matrix = self.predicted_values[e].copy()

            # adjust
            target_matrix[action_index] = reward + max(self.predicted_values[e + 1]) * DISCOUNT
            observation = [arr.astype("float32") for arr in observation]
            target_matrix = target_matrix.astype("float32")
            states.append(observation)
            targets.append(target_matrix)

        return states, targets

    def __len__(self):
        return len(self.observations)


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_exp(self, exp):
        if not self.is_full():
            self.buffer.append(exp)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size

    def __len__(self) -> int:
        return sum([len(x) for x in self.buffer])

    def size(self) -> int:
        return len(self.buffer)

    def create_targets(self) -> tuple[list, list]:
        state_dataset = []
        targets_dataset = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets()
            targets_dataset += targets
            state_dataset += states
        return state_dataset, targets_dataset

    def flip_dataset(self):
        """
        Mirrors the image and action data in dataset, effectively doubles it.
        :return:
        """
        new_exps = []
        for exp in self.buffer:
            new_exp = exp.flip()
            new_exps.append(new_exp)

        for new_exp in new_exps:
            self.buffer.append(new_exp)

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
