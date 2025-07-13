from abc import ABC, abstractmethod
from pathlib import Path

from mlagents_envs.environment import ActionTuple, DecisionSteps, TerminalSteps


class Agent(ABC):
    def __init__(self, behavior_name: str) -> None:
        super().__init__()
        self.behavior_name = behavior_name

    @abstractmethod
    def submit_actions(self, steps: tuple[DecisionSteps, TerminalSteps]) -> ActionTuple | None:
        pass

    @abstractmethod
    def train(self) -> float:
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        pass
