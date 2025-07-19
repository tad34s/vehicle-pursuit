import copy
from pathlib import Path

import numpy as np
import torch
import torch.onnx
from mlagents_envs.base_env import DecisionStep, TerminalStep
from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps
from torch.utils.tensorboard.writer import SummaryWriter

from agent_interface import Agent
from follower_agent.buffer import Experience, ReplayBuffer, State
from follower_agent.hyperparameters import REDUCE_TEMPERATURE, START_TEMPERATURE
from follower_agent.network_pipeline import NetworkPipeline
from follower_agent.wrapper_net import WrapperNet
from variables import ACTION_OPTIONS


class FollowerAgent(Agent):
    behavior_name = "CarBehavior?team=0"
    name = "Follower"

    def __init__(
        self,
        visual_input_shape: tuple,
        nonvis_input_shape: tuple,
        buffer_size: int,
        device: torch.device,
        num_agents: int = 1,
        writer: SummaryWriter | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.writer = writer

        self.num_agents = num_agents

        self.temperature = 0
        self.episode_rewards = 0
        self.memory = ReplayBuffer()

        self.exps: dict[int, Experience] = {}
        self.temperature = START_TEMPERATURE
        self.episode_rewards: float = 0
        self.curr_episode = 1

        self.model = NetworkPipeline(
            visual_input_shape,
            nonvis_input_shape,
            device,
            inject_correct_values=True,
        )

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
                q_values, t_ref_pred, action_index = self.model.get_actions(
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
                    t_ref_pred,
                    q_values.copy(),
                    reward,
                )

            action_tuple = ActionTuple()
            final_dis_action_values = np.array(dis_action_values)
            final_cont_action_values = np.array(cont_action_values)
            action_tuple.add_discrete(final_dis_action_values)
            action_tuple.add_continuous(final_cont_action_values)

            return action_tuple

    def train(self) -> float:
        # clean not finished experiences
        for id, exp in self.exps.items():
            if len(exp.actions) == 0:
                continue
            exp.actions[-1] = None
            self.memory.add_exp(exp)
            self.episode_rewards += float(sum(exp.rewards))

        self.memory.flip_dataset()
        sample_exp = self.memory.buffer[int(self.memory.size() / 2)]
        sample_image = sample_exp.states[int(len(sample_exp) / 2)].img
        sample_q_values = sample_exp.q_values_pred[int(len(sample_exp) / 2)]

        avg_loss_qnet, avg_loss_depth_net = self.model.fit(self.memory)
        self.temperature = max(0.0, self.temperature - REDUCE_TEMPERATURE)
        if self.writer is not None:
            self.writer.add_image("Sample image follower", sample_image)

            # Add text
            steer = ""

            for s in sample_q_values:
                steer += f"{s:.2f} "

            self.writer.add_text("Sample Q follower values (steer)", steer, self.curr_episode)

            self.writer.add_scalar("Loss/Epoch Q-Net", avg_loss_qnet, self.curr_episode)
            self.writer.add_scalar("Loss/Epoch Depth-Net", avg_loss_depth_net, self.curr_episode)
            self.writer.add_scalar("Temperature Follower ", self.temperature, self.curr_episode)

        self.memory.wipe()

        final_episode_rewards: float = self.episode_rewards
        self.episode_rewards = 0
        self.curr_episode += 1
        return final_episode_rewards

    def save_model(self, path: Path) -> None:
        torch.onnx.export(
            WrapperNet(copy.deepcopy(self.model)),
            (
                # Vis observation
                torch.randn((1, *self.model.visual_input_shape)),
                # Non vis observation
                torch.randn((1, *self.model.nonvis_input_shape)),
            ),
            str(path),
            input_names=["visual_obs", "nonvis_obs"],
            output_names=["q_values", "actions"],
            opset_version=11,  # Use at least opset 11
            dynamic_axes={
                "visual_obs": {0: "batch_size"},
                "nonvis_obs": {0: "batch_size"},
                "q_values": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )

    def get_state_and_reward(self, step: DecisionStep | TerminalStep) -> tuple[State, float]:
        state = State(step.obs)

        reward = step.reward
        return state, reward
