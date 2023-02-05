from abc import ABC, abstractmethod

from deep_search.search.action import Action
from deep_search.search.state import State


class Agent(ABC):
    @abstractmethod
    def decide_action(self, state: State) -> Action:
        raise NotImplemented
