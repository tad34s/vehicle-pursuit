from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
from mlagents_envs.base_env import DecisionStep, TerminalStep
from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps
from torch.utils.tensorboard.writer import SummaryWriter

from agent_interface import Agent
from depth_net.controller import ModelPredictiveControl
from follower_agent.buffer import State


class FollowerControllerAgent(Agent):
    behavior_name = "CarBehavior?team=0"
    name = "Follower"

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        num_agents: int = 1,
        writer: SummaryWriter | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.writer = writer

        self.num_agents = num_agents

        self.episode_rewards: float = 0
        self.curr_episode = 1

        # self.controllers = [ModelPredictiveControl() * num_agents]
        self.controllers: dict[int, ModelPredictiveControl] = {}
        self.inject_correct_values = True

        # load onnx
        self.depth_net = ort.InferenceSession(model_path)

    def submit_actions(self, steps: tuple[DecisionSteps, TerminalSteps]) -> ActionTuple | None:
        decision_steps, terminal_steps = steps

        dis_action_values = []
        cont_action_values = []

        if len(decision_steps) == 0:
            return None
            for agent_id in terminal_steps:
                self.controllers.pop(agent_id)

        for agent_id in decision_steps:
            if agent_id not in self.controllers.keys():
                self.controllers[agent_id] = ModelPredictiveControl()
            state, _ = self.get_state_and_reward(decision_steps[agent_id])

            # Get the action
            actions = self.get_actions(state, agent_id)

            dis_action_values.append([])
            cont_action_values.append(actions)

        action_tuple = ActionTuple()
        final_dis_action_values = np.array(dis_action_values)
        final_cont_action_values = np.array(cont_action_values)
        action_tuple.add_discrete(final_dis_action_values)
        action_tuple.add_continuous(final_cont_action_values)

        return action_tuple

    def save_model(self, path: Path) -> None:
        return super().save_model(path)

    def train(self) -> float:
        # clean not finished experiences
        return 0.0

    def get_actions(self, state: State, agent_id: int):
        if self.inject_correct_values:
            t_ref = state.t_ref
        else:
            t_ref = self.depth_net.run(None, {"input": state.img})  # mby unsqueeze

        actions = self.controllers[agent_id].optimize_controls(state.speed, t_ref)
        actions = actions[0], actions[1]
        print("actions", actions)
        return actions

    def get_state_and_reward(self, step: DecisionStep | TerminalStep) -> tuple[State, float]:
        state = State(step.obs)

        # img_rgb = state.img.transpose(1, 2, 0)

        # state.img = cv2.resize(state.img, (128, 128), interpolation=cv2.INTER_LANCZOS4)

        reward = step.reward
        return state, reward
