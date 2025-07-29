from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps


class Agent(ABC):
    behavior_name = ""
    name = ""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def submit_actions(self, steps: tuple[DecisionSteps, TerminalSteps]) -> ActionTuple | None:
        pass

    @abstractmethod
    def train(self) -> float:
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        pass

    @classmethod
    def from_onnyx(
        cls,
        agent_class: type,
        path: str,
        visual_input_shape,
        nonvis_input_shape,
        inject_correct: bool = False,
    ) -> "AgentOnnyx":
        return AgentOnnyx(
            agent_class.behavior_name,
            agent_class.name,
            path,
            visual_input_shape,
            nonvis_input_shape,
            inject_correct,
        )


class AgentOnnyx(Agent):
    def __init__(
        self,
        behavior_name: str,
        name: str,
        model_path: str,
        visual_input_shape,
        nonvis_input_shape,
        inject_correct: bool,
    ) -> None:
        super().__init__()
        self.model = ort.InferenceSession(model_path)
        self.inject_correct = inject_correct
        self.behavior_name = behavior_name
        self.name = name
        self.visual_input_shape = visual_input_shape
        self.nonvis_input_shape = nonvis_input_shape

    def submit_actions(self, steps: tuple[DecisionSteps, TerminalSteps]) -> ActionTuple | None:
        decision_steps, _ = steps
        cont_action_values = []
        dis_action_values = []
        if len(decision_steps) == 0:
            return None

        for agent_id in decision_steps:
            step = decision_steps[agent_id]
            if self.name == "Leader":
                from leader_agent.agent import LeaderAgent

                visual_ob = LeaderAgent.image_preprocessing(
                    self.visual_input_shape, step.obs[0]
                ).reshape(1, *self.visual_input_shape)
            else:
                visual_ob = step.obs[0].reshape(1, *self.visual_input_shape)

            nonvis_ob = step.obs[1].reshape(1, *self.nonvis_input_shape)

            if not self.inject_correct:
                nonvis_ob[0, 3:] = torch.nan

            print(self.name, nonvis_ob)
            outputs = self.model.run(None, {"visual_obs": visual_ob, "nonvis_obs": nonvis_ob})

            actions = outputs[1]
            dis_action_values.append(actions[0])
            cont_action_values.append([])

        action_tuple = ActionTuple()
        final_dis_action_values = np.array(dis_action_values)
        final_cont_action_values = np.array(cont_action_values)
        action_tuple.add_discrete(final_dis_action_values)
        action_tuple.add_continuous(final_cont_action_values)

        return action_tuple

    def train(self) -> float:
        return 0.0

    def save_model(self, path: Path) -> None:
        _ = path
        pass
