from copy import copy

import numpy as np
import torch
from torch.utils.data import Dataset

from leader_agent.hyperparameters import DISCOUNT, STEERING_DISCOUNT
from variables import MIRRORED_ACTIONS


class State:
    def __init__(self, obs: list[np.ndarray]) -> None:
        vis_obs, nonvis_obs = obs
        self.img: np.ndarray = vis_obs
        self.steer: float = nonvis_obs[0]
        self.speed: float = nonvis_obs[1]
        self.leader_speed: float = nonvis_obs[2]
        self.t_ref: tuple[float, float, float] = (nonvis_obs[3], nonvis_obs[4], nonvis_obs[5])

    def flip(self):
        new_state = copy(self)
        # fliping image
        new_state.img = np.flip(new_state.img, 2)
        # fliping steering
        new_state.steer = new_state.steer * -1
        # fliping x axes and leader rotation
        self.t_ref = (new_state.t_ref[0] * -1, new_state.t_ref[1], new_state.t_ref[2] * -1)

        return new_state


class StateTargetValuesDataset(Dataset):
    def __init__(self, states: list, targets: list) -> None:
        super().__init__()
        self.states = [torch.tensor(x) for x in states]
        self.targets = [torch.tensor(x) for x in targets]
        if len(states) != len(targets):
            raise ValueError

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.states[index], self.targets[index]


class ImageDepthDataset(Dataset):
    def __init__(self, imgs, leader_speeds, car_speeds, y) -> None:
        super().__init__()
        self.imgs = [torch.tensor(img.copy()) for img in imgs]
        self.leader_speeds = [torch.tensor(leader_speed) for leader_speed in leader_speeds]
        self.car_speeds = [torch.tensor(car_speed) for car_speed in car_speeds]
        self.y = [torch.tensor(val) for val in y]
        if len(self.imgs) != len(self.car_speeds):
            raise ValueError

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, float, float], np.ndarray]:
        return (self.imgs[index], self.leader_speeds[index], self.car_speeds[index]), self.y[index]


class Experience:
    def __init__(self) -> None:
        self.states: list[State] = []
        self.actions: list[int | None] = []
        self.rewards: list[float] = []
        self.t_ref_pred: list[np.ndarray] = []
        self.q_values_pred: list[np.ndarray] = []

    def add_instance(
        self,
        state: State,
        action: int | None,
        t_ref_pred: np.ndarray | None,
        predicted_values: np.ndarray,
        reward: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if t_ref_pred is not None:
            self.t_ref_pred.append(t_ref_pred)
        self.q_values_pred.append(predicted_values)

    def flip(self) -> "Experience":
        new_states = [x.flip() for x in self.states]

        new_actions = [MIRRORED_ACTIONS[x] if x is not None else None for x in self.actions]

        new_predicted_values = [np.flip(x) for x in self.q_values_pred]

        new_t_ref_pred = [
            np.array([x[0] * -1.0, x[1], x[2] * -1.0], dtype=np.float32) for x in self.t_ref_pred
        ]

        new_exp = Experience()
        new_exp.states = new_states
        new_exp.actions = new_actions
        new_exp.rewards = self.rewards.copy()
        new_exp.t_ref_pred = new_t_ref_pred
        new_exp.q_values_pred = new_predicted_values
        return new_exp

    def calculate_targets(
        self,
        inject_correct_values=False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        targets = []
        states = []
        for e, state in enumerate(self.states):
            if self.actions[e] is None:
                break

            action_index = self.actions[e]
            reward = self.rewards[e]

            if action_index != 1:  # is steering
                reward *= STEERING_DISCOUNT

            # we take the matrix of predicted values and for the actions
            # we had taken adjust the value by the reward and the value of the next state
            target_matrix = self.q_values_pred[e].copy()

            # adjust
            target_matrix[action_index] = reward + max(self.q_values_pred[e + 1]) * DISCOUNT
            targets.append(target_matrix.astype("float32"))

            # add state

            if not inject_correct_values and len(self.t_ref_pred) > e:
                t_ref = self.t_ref_pred[e]
            else:
                t_ref = state.t_ref
            states.append(
                np.array([state.steer, state.speed, state.leader_speed, *t_ref], dtype=np.float32)
            )

        return states, targets

    def get_depth_data(
        self,
    ) -> tuple[tuple[list[np.ndarray], list[float], list[float]], list[np.ndarray]]:
        x = (
            [state.img for state in self.states],
            [state.leader_speed for state in self.states],
            [state.speed for state in self.states],
        )
        y = [np.array([*state.t_ref], dtype=np.float32) for state in self.states]
        return x, y

    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer: list[Experience] = []

    def add_exp(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def __len__(self) -> int:
        return sum([len(x) for x in self.buffer])

    def size(self) -> int:
        return len(self.buffer)

    def create_qnet_targets(
        self, inject_correct_values=False
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        state_dataset = []
        targets_dataset = []
        for exp in self.buffer:
            states, targets = exp.calculate_targets(inject_correct_values)
            targets_dataset += targets
            state_dataset += states
        return state_dataset, targets_dataset

    def create_depth_net_targets(
        self,
    ) -> tuple[tuple[list[np.ndarray], list[float], list[float]], list[list[np.ndarray]]]:
        imgs = []
        leader_speeds = []
        car_speeds = []
        Y = []
        for exp in self.buffer:
            (imgs_exp, leader_speeds_exp, car_speeds_exp), y = exp.get_depth_data()
            imgs += imgs_exp
            leader_speeds += leader_speeds_exp
            car_speeds += car_speeds_exp
            Y += y
        return (imgs, leader_speeds, car_speeds), Y

    def flip_dataset(self) -> None:
        """
        Mirrors the image and action data in dataset, effectively doubles it.
        """
        new_exps = []
        for exp in self.buffer:
            new_exp = exp.flip()
            new_exps.append(new_exp)

        for new_exp in new_exps:
            self.buffer.append(new_exp)

    def get_qnet_dataset(self, inject_correct_values=False) -> StateTargetValuesDataset:
        states, targets = self.create_qnet_targets(inject_correct_values)
        return StateTargetValuesDataset(states, targets)

    def get_depth_net_dataset(self) -> ImageDepthDataset:
        (imgs, leaders_speeds, car_speeds), y = self.create_depth_net_targets()
        return ImageDepthDataset(imgs, leaders_speeds, car_speeds, y)

    def wipe(self) -> None:
        self.buffer = []
