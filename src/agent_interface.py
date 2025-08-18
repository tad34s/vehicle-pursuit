from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps

from follower_agent.buffer import State
from variables import DATASET_LOCATION

counter = 0


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
        create_dataset: bool = False,
    ) -> "AgentOnnyx":
        return AgentOnnyx(
            agent_class.behavior_name,
            agent_class.name,
            path,
            visual_input_shape,
            nonvis_input_shape,
            inject_correct,
            create_dataset,
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
        create_dataset: bool = False,
    ) -> None:
        super().__init__()
        self.model = ort.InferenceSession(model_path)
        self.inject_correct = inject_correct
        self.behavior_name = behavior_name
        self.name = name
        self.visual_input_shape = visual_input_shape
        self.nonvis_input_shape = nonvis_input_shape
        self.save_dataset = create_dataset

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
                dim = (128, 128)
                visual_ob = step.obs[0]
                visual_ob = np.transpose(visual_ob, (1, 2, 0))
                visual_ob = cv2.resize(visual_ob, dsize=dim, interpolation=cv2.INTER_AREA)
                visual_ob = np.transpose(visual_ob, (2, 0, 1))
                visual_ob = visual_ob.reshape(1, *self.visual_input_shape)

            nonvis_ob = step.obs[1].reshape(1, *self.nonvis_input_shape)

            if not self.inject_correct:
                nonvis_ob[0, 3:] = torch.nan

            if self.save_dataset:
                self.save_state(State(step.obs))

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

    def save_state(self, state: State):
        global counter

        dataset_path = Path(DATASET_LOCATION)
        dataset_images = dataset_path / "images"
        dataset_t_ref = dataset_path / "t_ref"
        dataset_images.mkdir(parents=True, exist_ok=True)
        dataset_t_ref.mkdir(parents=True, exist_ok=True)

        img = np.transpose(state.img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            pass
        cv2.imwrite(str(dataset_images / f"{counter}.png"), img)
        np.save(str(dataset_t_ref / f"{counter}.npy"), state.t_ref)
        counter += 1
